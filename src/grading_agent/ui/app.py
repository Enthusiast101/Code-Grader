"""Local Gradio interface for C assignment grading and FormAI-v2 evaluation."""
from __future__ import annotations

import json
import logging
import math
import random
from argparse import Namespace
from pathlib import Path
import shutil
import uuid
from typing import Any

import gradio as gr

from grading_agent.benchmark.formai_evaluate import evaluate_mode
from grading_agent.benchmark.formai_predict import run as run_formai_prediction
from grading_agent.benchmark.formai_manifest import validate_benchmark_split, read_source_manifest
from grading_agent.main import PROJECT_ROOT, grade_with_rules_file
from grading_agent.config import settings
from grading_agent.reporting.rationale import compact_rationale, evidence_summary, score_source_label

logger = logging.getLogger("grading_agent.ui")


def _upload_path(upload: Any, label: str) -> Path:
    if upload is None:
        raise ValueError(f"Upload the {label} file first.")
    raw = upload if isinstance(upload, (str, Path)) else getattr(upload, "name", None)
    if not raw:
        raise ValueError(f"Could not resolve the uploaded {label} file.")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Uploaded {label} file does not exist: {path}")
    return path

def _sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_source_sample(inputs_path: Path, output_path: Path, sample_count: int, seed: int) -> list[str]:
    """Write a reproducible source-only sample without reading label classes."""
    all_names = read_source_manifest(inputs_path)
    if sample_count <= 0 or sample_count >= len(all_names):
        shutil.copyfile(inputs_path, output_path)
        return list(all_names)
    selected = set(random.Random(seed).sample(all_names, sample_count))
    written: list[str] = []
    with inputs_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, raw_line in enumerate(source, 1):
            if not raw_line.strip():
                continue
            item = json.loads(raw_line)
            if item.get("file_name") in selected:
                target.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                written.append(item["file_name"])
    if set(written) != selected:
        raise ValueError("Could not materialise the requested source sample exactly.")
    return written


def _write_label_sample(labels_path: Path, output_path: Path, selected_names: list[str]) -> None:
    """Filter labels only after inference; labels never enter either predictor stage."""
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Ground-truth labels must be a JSON array.")
    by_name = {item.get("file_name"): item for item in payload if isinstance(item, dict)}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise ValueError(f"Selected source programs are missing ground-truth labels: {missing[:5]}")
    subset = [by_name[name] for name in selected_names]
    output_path.write_text(json.dumps(subset, indent=2), encoding="utf-8")


def _metric_summary_impl(report: dict[str, Any]) -> str:
    """Render the Gradio summary defensively; malformed optional fields never hide the report."""
    def number(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else default
        except (TypeError, ValueError):
            return default

    if not isinstance(report, dict):
        return "## Final grading result\n\nThe grader returned a non-object report. Check the JSON artifact."
    scoring = report.get("scoring", {})
    scoring = scoring if isinstance(scoring, dict) else {}
    lines = ["## Final grading result"]
    final_score = scoring.get("score")
    grade_label = f"{number(final_score):.2f}/100" if isinstance(final_score, (int, float)) else "N/A — insufficient grading evidence"
    lines.append(f"**Overall grade: {grade_label}**")
    lines.append(f"**Status:** `{scoring.get('status', 'completed_with_warnings')}`")
    active_problem = report.get("problem")
    if isinstance(active_problem, str) and active_problem.strip():
        lines.append("**Assignment used for this run:** " + active_problem.strip().replace("\n", " "))
    provenance = report.get("rules_provenance", {})
    if isinstance(provenance, dict) and provenance:
        origin = provenance.get("selection", {})
        origin = origin if isinstance(origin, dict) else {}
        source_label = origin.get("selected_from", "rules file")
        fingerprint = provenance.get("rules_sha256") or origin.get("selected_rules_sha256")
        lines.append(f"**Rules source:** `{source_label}`" + (f" · **Rules SHA-256:** `{fingerprint[:16]}…`" if isinstance(fingerprint, str) and fingerprint else ""))
        if origin.get("editor_was_stale"):
            lines.append("**Rules note:** The uploaded file changed after the editor was populated; the newer uploaded file was used to avoid stale assignment rules.")
    rsi_overview = report.get("llm_rsi_evaluation", {})
    rsi_overview = rsi_overview if isinstance(rsi_overview, dict) else {}
    summary = evidence_summary(report)
    if summary:
        lines.append("\n### Executive assessment")
        lines.append(summary)
    lines.append("\n### Metric scores")
    configuration = report.get("configuration", {})
    configuration = configuration if isinstance(configuration, dict) else {}
    enabled_metrics = configuration.get("enabled_metrics", [])
    enabled_metrics = enabled_metrics if isinstance(enabled_metrics, list) else []
    for metric in enabled_metrics:
        if not isinstance(metric, dict):
            continue
        name = str(metric.get("name", "unknown"))
        result = report.get(name, {})
        result = result if isinstance(result, dict) else {}
        metric_score = result.get("score")
        metric_score_label = f"{number(metric_score):.2f}/100" if isinstance(metric_score, (int, float)) else "N/A"
        source_label = score_source_label(result.get("score_origin"))
        lines.append(f"- **{name}: {metric_score_label}** _(source: {source_label})_")
        rationale = compact_rationale(result.get("score_rationale", ""), max_words=60, max_chars=420)
        if rationale:
            lines.append(f"  - Assessment: {rationale}")
        if name in {"correctness", "logical"}:
            rows = result.get("results", [])
            valid_rows = [row for row in rows if isinstance(row, dict) and isinstance(row.get("passed"), bool)] if isinstance(rows, list) else []
            if valid_rows:
                passed = sum(row["passed"] for row in valid_rows)
                lines.append(f"  - Tests: {passed}/{len(valid_rows)} passed")
                for row in [item for item in valid_rows if item.get("passed") is False][:3]:
                    lines.append(
                        f"  - Failed case {row.get('test_id', '?')}: input {row.get('input', '')!r}; "
                        f"expected {row.get('expected', '')!r}, got {row.get('actual', row.get('output', ''))!r}."
                    )

    understanding = report.get("program_understanding", {})
    if isinstance(understanding, dict) and (understanding.get("program_purpose") or understanding.get("actual_behavior")):
        lines.append("\n### Program understanding")
        actual = understanding.get("actual_behavior", understanding.get("program_purpose", ""))
        lines.append("**Actual source behavior:** " + str(actual).replace("\n", " "))
        if understanding.get("intended_behavior"):
            lines.append("**Expected assignment behavior:** " + str(understanding.get("intended_behavior")).replace("\n", " "))
        if understanding.get("contract_alignment"):
            lines.append("**Contract comparison:** " + str(understanding.get("contract_alignment")).replace("\n", " "))
        if understanding.get("algorithm_overview"):
            lines.append("**Actual algorithm:** " + str(understanding.get("algorithm_overview")).replace("\n", " "))

    findings = report.get("llm_rsi_findings", [])
    findings = findings if isinstance(findings, list) else []
    lines.append("\n### Whole-program RSI findings")
    if findings:
        for finding in findings[:12]:
            if not isinstance(finding, dict):
                continue
            location = f"line {finding.get('line_start')}" if finding.get("line_start") is not None else "location unresolved"
            description = str(finding.get("description", "")).replace("\n", " ")
            lines.append(
                f"- **{str(finding.get('severity', 'unknown')).upper()} — "
                f"{finding.get('metric', 'unmapped')} — {finding.get('title', 'Issue')}** ({location}): "
                f"{description}"
            )
    else:
        rsi_value = report.get("llm_rsi_evaluation", {})
        rsi_value = rsi_value if isinstance(rsi_value, dict) else {}
        rsi_status = rsi_value.get("status", "unavailable")
        if rsi_status in {"unavailable", "completed_with_fallback"}:
            lines.append("A complete LLM finding list was unavailable; deterministic analyzer findings and fallback scoring remain in the report. This does not mean the source is defect-free.")
        elif rsi_status == "completed_with_warnings":
            lines.append("No findings were retained from the latest review; one or more review/scoring stages used a fallback.")
        else:
            lines.append("No concrete, supported issues were retained by the successfully completed RSI review sequence.")

    rejected_findings = report.get("llm_rsi_rejected_findings", [])
    if isinstance(rejected_findings, list) and rejected_findings:
        lines.append("\n### Excluded RSI candidates (not used for scoring)")
        for item in rejected_findings[:12]:
            if not isinstance(item, dict):
                continue
            location = f"line {item.get('line_start')}" if item.get("line_start") is not None else "location unresolved"
            lines.append(
                f"- **{item.get('metric', 'unmapped')} — {item.get('title', 'Candidate')}** ({location}): "
                f"{str(item.get('reason', 'Not retained after validation.'))[:350]}"
            )

    rsi = report.get("llm_rsi_evaluation", {})
    rsi = rsi if isinstance(rsi, dict) else {}
    lines.append("\n### Program understanding + sequential whole-program RSI")
    lines.append(f"Status: `{rsi.get('status', 'completed_with_warnings')}`")
    if rsi.get("llm_calls") is not None:
        lines.append(f"LLM calls: {rsi.get('llm_calls')} (program understanding + sequential RSI + metric scoring)")
    latency = rsi.get("latency_seconds", {})
    latency = latency if isinstance(latency, dict) else {}
    if latency.get("total_llm") is not None:
        lines.append(f"LLM time: {number(latency.get('total_llm')):.2f} seconds")
        if isinstance(latency.get("program_understanding"), (int, float)):
            lines.append(f"  Program understanding: {number(latency.get('program_understanding')):.2f} seconds")
        for index, duration in enumerate(latency.get("rsi_requests", []) if isinstance(latency.get("rsi_requests"), list) else [], start=1):
            lines.append(f"  RSI request {index}: {number(duration):.2f} seconds")
        for index, duration in enumerate(latency.get("finding_validation_requests", []) if isinstance(latency.get("finding_validation_requests"), list) else [], start=1):
            lines.append(f"  Post-RSI validation {index}: {number(duration):.2f} seconds")
    if rsi.get("error"):
        lines.append(f"LLM note: {str(rsi.get('error')).replace(chr(10), ' ')[:1200]}")
    for diagnostic in rsi.get("diagnostics", [])[:4] if isinstance(rsi.get("diagnostics"), list) else []:
        lines.append(f"LLM diagnostic: {str(diagnostic).replace(chr(10), ' ')[:600]}")
    missing = rsi.get("metrics_without_llm_score", rsi.get("missing_metrics", []))
    if isinstance(missing, list) and missing:
        lines.append("Metrics using an explicit fallback score: " + ", ".join(str(item) for item in missing))

    compression = rsi.get("context_compression", {})
    if isinstance(compression, dict) and compression:
        status = str(compression.get("status", "not_needed"))
        model_calls = int(compression.get("same_model_calls", 0) or 0)
        if status == "same_model_used" and model_calls > 0:
            lines.append(f"Context compression: same grading model used ({model_calls} request(s)); model={compression.get('model', 'configured model')}")
        elif status == "deterministic_fallback":
            lines.append("Context compression: deterministic anchor-preserving compaction used; same-model compression was unavailable or rejected.")
        elif compression.get("enabled"):
            lines.append("Context compression: enabled; model compression was not needed for this run.")
        else:
            lines.append("Context compression: disabled.")

    warnings = scoring.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.append("\n### Scoring notes")
        lines.extend(f"- {str(warning)}" for warning in warnings[:8])
    if isinstance(report.get("sandbox"), dict) and report["sandbox"].get("status") == "unavailable":
        lines.append("\n**Docker was unavailable:** dynamic analyzer results are shown as unavailable; LLM/fallback evaluation may still use source and available evidence.")
    return "\n".join(lines)


def _metric_summary(report: dict[str, Any]) -> str:
    """Format a final UI result defensively; formatting must not hide the grade."""
    try:
        return _metric_summary_impl(report)
    except Exception as exc:
        import json
        if not isinstance(report, dict):
            return (
                "## Final grading result\n\n"
                "The grading pipeline returned a non-object report. Check the JSON artifact."
            )
        scoring = report.get("scoring", {})
        scoring = scoring if isinstance(scoring, dict) else {}
        try:
            score = float(scoring.get("score", 0.0))
            if not math.isfinite(score):
                score = 0.0
        except (TypeError, ValueError):
            score = 0.0
        lines = [
            "## Final grading result",
            f"**Overall grade: {score:.2f}/100**",
            f"**Status:** `{scoring.get('status', 'completed_with_warnings')}`",
            "",
            "The detailed Markdown summary used a fallback formatter; the JSON report remains the source of record.",
            f"Formatting diagnostic: `{type(exc).__name__}: {exc}`",
        ]
        try:
            llm = report.get("llm_rsi_evaluation", {})
            llm = llm if isinstance(llm, dict) else {}
            lines.append(f"LLM evaluation status: `{llm.get('status', 'unavailable')}`")
            for finding in report.get("llm_rsi_findings", [])[:12]:
                if isinstance(finding, dict):
                    lines.append(
                        f"- **{finding.get('severity', 'unknown')} — {finding.get('metric', 'unmapped')}**: "
                        f"{str(finding.get('title', 'Issue'))[:140]}"
                    )
        except Exception:
            pass
        return "\n".join(lines)


def load_rules_editor_from_upload(rules_upload: Any):
    """Populate the editor and remember which uploaded file it was based on.

    The source hash lets the submit handler detect a newly uploaded file while
    an older editor value is still present in the browser state.
    """
    default_time = float(settings.DOCKER_DEFAULT_TIMEOUT_S)
    default_memory = 512
    if rules_upload is None:
        return "", default_time, default_memory, "Upload a rules JSON file to populate the editor.", ""
    try:
        path = _upload_path(rules_upload, "rules JSON")
        content = path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Rules JSON root must be an object.")
        time_limit = float(parsed.get("time_limit_s", default_time))
        memory_limit = int(parsed.get("memory_limit_mb", default_memory))
        digest = _sha256_text(content)
        return content, time_limit, memory_limit, f"Loaded {path.name}; edit the JSON or use the controls below.", digest
    except Exception as exc:
        logger.warning("Could not load rules into Gradio editor: %s", exc)
        return "", default_time, default_memory, f"Could not load rules: {type(exc).__name__}: {exc}", ""


def grade_uploaded_submission(
    source_upload: Any,
    rules_upload: Any,
    rules_json_text: str,
    time_limit_s: float,
    memory_limit_mb: float,
    rsi_iterations: float,
    understanding_tokens: float,
    rsi_tokens: float,
    scoring_tokens: float,
    max_findings: float,
    source_chars: float,
    evidence_chars: float,
    temperature: float,
    model_name: str,
    read_timeout_s: float,
    docker_cpu_limit: float,
    enable_thinking: bool,
    run_sanitizers: bool,
    context_compression_enabled: bool = True,
    compression_threshold_chars: float = 1800,
    rules_editor_source_hash: str = "",
):
    """Stage files, apply per-run controls, and execute grading in isolation."""
    stage_dir: Path | None = None
    try:
        source_path = _upload_path(source_upload, "C source")
        if source_path.suffix.lower() != ".c":
            raise ValueError("The submission must be a .c file.")
        if source_path.stat().st_size > settings.UI_MAX_UPLOAD_BYTES:
            raise ValueError(f"The source file exceeds the {settings.UI_MAX_UPLOAD_BYTES} byte project limit.")

        editor_text = str(rules_json_text or "").strip()
        uploaded_path: Path | None = None
        uploaded_text = ""
        uploaded_hash = ""
        if rules_upload is not None:
            uploaded_path = _upload_path(rules_upload, "rules JSON")
            uploaded_text = uploaded_path.read_text(encoding="utf-8")
            uploaded_hash = _sha256_text(uploaded_text)

        # The editable panel is authoritative only when it was populated from
        # the currently uploaded file. If a new file has replaced that upload
        # but the browser still submits an older editor value, prefer the new
        # upload instead of silently grading against stale assignment rules.
        if uploaded_path is not None:
            if rules_editor_source_hash and rules_editor_source_hash == uploaded_hash and editor_text:
                raw_rules = editor_text
                selected_rules_source = "edited Gradio rules editor"
            else:
                raw_rules = uploaded_text
                selected_rules_source = "uploaded rules file"
        elif editor_text:
            raw_rules = editor_text
            selected_rules_source = "manually entered Gradio rules editor"
        else:
            raise ValueError("Upload a rules JSON file or enter rules in the editor.")

        rules = json.loads(raw_rules)
        if not isinstance(rules, dict) or not isinstance(rules.get("submission"), dict):
            raise ValueError("The rules JSON must contain a submission object.")
        selected_rules_hash = _sha256_text(raw_rules)
        rules["_run_metadata"] = {
            "selection": {
                "selected_from": selected_rules_source,
                "uploaded_file_name": uploaded_path.name if uploaded_path else None,
                "uploaded_file_sha256": uploaded_hash or None,
                "editor_base_sha256": rules_editor_source_hash or None,
                "selected_rules_sha256": selected_rules_hash,
                "editor_was_stale": bool(
                    uploaded_path is not None
                    and rules_editor_source_hash
                    and rules_editor_source_hash != uploaded_hash
                ),
            }
        }

        # The compact controls intentionally override just the corresponding
        # top-level rule values. All other metric-specific params remain fully
        # editable in the JSON editor.
        numeric_controls = {
            "time_limit_s": time_limit_s,
            "memory_limit_mb": memory_limit_mb,
        }
        for key, value in numeric_controls.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be a finite positive number.")
        rules["time_limit_s"] = float(time_limit_s)
        rules["memory_limit_mb"] = int(memory_limit_mb)
        if int(memory_limit_mb) != memory_limit_mb:
            raise ValueError("Docker memory limit must be a whole number of MB.")

        # Sanitizer control is reflected into the metric's validated params.
        for metric in rules.get("metrics", []):
            if isinstance(metric, dict) and metric.get("analyzer") in {"memory_safety", "valgrind_memcheck"}:
                params = metric.setdefault("params", {})
                if isinstance(params, dict):
                    params["run_sanitizers"] = bool(run_sanitizers)

        model_name = str(model_name or "").strip()
        if not model_name:
            raise ValueError("Model name cannot be empty.")
        if not 0.0 <= float(temperature) <= 1.0:
            raise ValueError("Temperature must be between 0 and 1.")
        if not 1 <= int(rsi_iterations) <= settings.LLM_REVIEW_MAX_ITERATIONS:
            raise ValueError(f"RSI iterations must be between 1 and {settings.LLM_REVIEW_MAX_ITERATIONS}.")
        if int(understanding_tokens) < 64 or int(rsi_tokens) < 128 or int(scoring_tokens) < 64:
            raise ValueError("Token budgets are too small; understanding/scoring must be >=64 and RSI >=128.")
        if not 1 <= int(max_findings) <= 50:
            raise ValueError("Maximum findings must be between 1 and 50.")
        if int(source_chars) < 1024 or int(evidence_chars) < 512:
            raise ValueError("Source/evidence prompt limits are too small (source >=1024, evidence >=512).")
        if not 10 <= float(read_timeout_s) <= 900:
            raise ValueError("LLM read timeout must be between 10 and 900 seconds.")
        if not 0.25 <= float(docker_cpu_limit) <= 16:
            raise ValueError("Docker CPU limit must be between 0.25 and 16 cores.")
        if not 256 <= int(compression_threshold_chars) <= 20000:
            raise ValueError("Context-compression threshold must be between 256 and 20,000 characters.")

        run_id = uuid.uuid4().hex[:12]
        stage_dir = settings.GRADIO_STAGING_DIR / run_id
        stage_dir.mkdir(parents=True, exist_ok=False)
        staged_source = stage_dir / "submission.c"
        shutil.copyfile(source_path, staged_source)
        rules["submission"]["source_file"] = str(staged_source.relative_to(PROJECT_ROOT))
        staged_rules = stage_dir / "rules.json"
        staged_rules.write_text(json.dumps(rules, indent=2), encoding="utf-8")

        runtime_options = {
            "MODEL": model_name,
            "TEMPERATURE": float(temperature),
            "LLAMA_REQUEST_TIMEOUT": (settings.LLAMA_REQUEST_CONNECT_TIMEOUT_S, float(read_timeout_s)),
            "LLM_REVIEW_MAX_ITERATIONS": int(rsi_iterations),
            "LLM_PROGRAM_UNDERSTANDING_MAX_TOKENS": int(understanding_tokens),
            "LLM_GLOBAL_RSI_MAX_TOKENS": int(rsi_tokens),
            "LLM_GLOBAL_SCORE_MAX_TOKENS": int(scoring_tokens),
            "LLM_GLOBAL_MAX_FINDINGS": int(max_findings),
            "LLM_PROGRAM_SOURCE_MAX_CHARS": int(source_chars),
            "LLM_GLOBAL_SOURCE_MAX_CHARS": int(source_chars),
            "LLM_MAX_EVIDENCE_CHARS": int(evidence_chars),
            "LLAMA_REASONING_ENABLED": True,
            "LLAMA_REASONING_FORMAT": "deepseek",
            "LLAMA_REASONING_BUDGET": settings.LLAMA_REASONING_BUDGET,
            "LLAMA_JSON_GRAMMAR_ENABLED": True,
            "LLM_CONTEXT_COMPRESSION_ENABLED": bool(context_compression_enabled),
            "LLM_CONTEXT_COMPRESSION_THRESHOLD_CHARS": int(compression_threshold_chars),
            "DOCKER_CPU_LIMIT": float(docker_cpu_limit),
        }
        report_dir = settings.GRADIO_REPORTS_DIR / run_id
        report, json_path, text_path = grade_with_rules_file(
            staged_rules,
            project_root=PROJECT_ROOT,
            report_dir=report_dir,
            runtime_options=runtime_options,
        )
        return _metric_summary(report), str(json_path), str(text_path)
    except Exception as exc:
        logger.exception("Gradio grading request failed")
        return f"## Could not complete grading\n`{type(exc).__name__}: {exc}`", None, None
    finally:
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)


def _prediction_stage_counts(prediction_path: Path) -> dict[str, int]:
    """Count completed/failed analysis stages for transparent UI feedback."""
    counts = {
        "samples": 0, "llm_completed": 0, "llm_failed": 0,
        "cppcheck_completed": 0, "cppcheck_unavailable": 0,
        "rsi_completed": 0, "rsi_unavailable": 0,
        "partial_predictions": 0, "inconclusive_predictions": 0,
    }
    with prediction_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            counts["samples"] += 1
            if item.get("llm_audit_status") in {"completed", "completed_with_warnings"}:
                counts["llm_completed"] += 1
            else:
                counts["llm_failed"] += 1
            if item.get("tool_status") == "completed":
                counts["cppcheck_completed"] += 1
            elif item.get("tool_status") in {"unavailable", "inconclusive"}:
                counts["cppcheck_unavailable"] += 1
            if item.get("rsi_review_status") in {"completed", "completed_with_warnings"}:
                counts["rsi_completed"] += 1
            elif item.get("rsi_review_status") in {"unavailable", "skipped"}:
                counts["rsi_unavailable"] += 1
            if item.get("prediction_status") == "completed_with_warnings":
                counts["partial_predictions"] += 1
            elif item.get("prediction_status") == "inconclusive":
                counts["inconclusive_predictions"] += 1
    return counts


def run_formai_combined(
    inputs_upload: Any,
    labels_upload: Any,
    mode: str,
    tool_timeout: float,
    max_llm_findings: float | int,
    rsi_iterations: float | int,
    memory_limit: str,
    cpu_limit: float,
    sample_count: float | int = 0,
):
    """Run LLM + cppcheck fusion, with optional evidence-review refinement."""
    try:
        inputs_path = _upload_path(inputs_upload, "source-only test JSONL")
        labels_path = _upload_path(labels_upload, "held-out labels JSON")
        if mode not in {"hybrid", "hybrid_rsi"}:
            raise ValueError("Choose a combined mode so both LLM and cppcheck are attempted.")
        # Filename alignment preflight prevents an accidental four-record run
        # from being reported as a 100-record benchmark. Labels remain isolated
        # from both prediction stages; only filenames/class counts are checked.
        manifest = validate_benchmark_split(inputs_path, labels_path)
        if tool_timeout <= 0 or int(max_llm_findings) < 1 or not 1 <= int(rsi_iterations) <= 3:
            raise ValueError("Invalid timeout, finding limit, or RSI iteration count.")
        if not 0.25 <= float(cpu_limit) <= 16:
            raise ValueError("Docker CPU limit must be between 0.25 and 16 cores.")
        sample_value = float(sample_count or 0)
        if not math.isfinite(sample_value) or not sample_value.is_integer():
            raise ValueError("Sample count must be a whole number (0 means all programs).")
        requested_samples = int(sample_value)
        if requested_samples < 0:
            raise ValueError("Sample count must be zero (all programs) or a positive integer.")
        if requested_samples > manifest["programs"]:
            raise ValueError(
                f"Requested {requested_samples} samples, but the uploaded split contains only {manifest['programs']} programs."
            )

        run_id = uuid.uuid4().hex[:12]
        out_dir = settings.FORM_AI_GRADIO_RUNS_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=False)
        prediction_path = out_dir / f"{mode}.jsonl"
        evaluation_path = out_dir / f"{mode}-evaluation.json"
        sampled_inputs_path = out_dir / "selected_test_inputs.jsonl"
        sampled_labels_path = out_dir / "selected_test_labels.json"
        selected_names = _write_source_sample(
            inputs_path, sampled_inputs_path, requested_samples, settings.FORM_AI_GRADIO_SAMPLE_SEED
        )
        selected_count = len(selected_names)
        args = Namespace(
            inputs=sampled_inputs_path,
            mode=mode,
            output=prediction_path,
            max_samples=None,
            tool_timeout=float(tool_timeout),
            max_llm_findings=int(max_llm_findings),
            rsi_iterations=int(rsi_iterations),
            memory_limit=memory_limit,
            cpu_limit=float(cpu_limit),
        )
        run_formai_prediction(args)
        # Labels are opened and filtered only after prediction generation.
        _write_label_sample(labels_path, sampled_labels_path, selected_names)
        evaluation = evaluate_mode(sampled_labels_path, prediction_path, line_tolerance=settings.FORM_AI_LINE_TOLERANCE)
        stage_counts = _prediction_stage_counts(prediction_path)
        stage_counts["expected_records"] = selected_count
        stage_counts["available_programs"] = manifest["programs"]
        stage_counts["sampled_programs"] = selected_count
        stage_counts["missing_prediction_records"] = max(0, selected_count - stage_counts["samples"])
        if stage_counts["samples"] != selected_count:
            logger.error(
                "Incomplete FormAI prediction run: wrote %d of %d selected records.",
                stage_counts["samples"], selected_count,
            )
        run_manifest_path = prediction_path.with_suffix(prediction_path.suffix + ".run.json")
        prediction_run_manifest = (
            json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if run_manifest_path.is_file() else None
        )
        evaluation_payload = {
            "dataset": "FormAI-v2",
            "mode": mode,
            "prediction_file": str(prediction_path),
            "evaluation": evaluation,
            "stage_counts": stage_counts,
            "input_manifest": manifest,
            "sample_selection": {
                "requested_count": requested_samples,
                "selected_count": selected_count,
                "available_count": manifest["programs"],
                "seed": settings.FORM_AI_GRADIO_SAMPLE_SEED,
                "strategy": "uniform random sample without replacement from source filenames; labels filtered only after prediction",
                "selected_file_names": selected_names,
            },
            "prediction_run_manifest": prediction_run_manifest,
            "fusion_policy": "cppcheck findings are retained; LLM candidates in hybrid_rsi require RSI adjudication against exact cited source lines; missing/inconclusive stages cannot be interpreted as safe.",
        }
        evaluation_path.write_text(json.dumps(evaluation_payload, indent=2), encoding="utf-8")
        confusion = evaluation["confusion_matrix"]
        if evaluation["metrics_finalized"]:
            metric_text = (
                f"**Precision / Recall / F1:** {evaluation['precision']} / {evaluation['recall']} / {evaluation['f1_score']}  \\n"
                f"**Accuracy / Balanced accuracy:** {evaluation['accuracy']} / {evaluation['balanced_accuracy']}  \\n"
            )
            evaluation_status = "**Benchmark status:** COMPLETE — metrics finalized for the selected sample."
        else:
            partial = evaluation["observed_subset_metrics"]
            metric_text = (
                "**Full-test-set metrics:** withheld because predictions are missing or inconclusive.  \\n"
                f"**Observed-subset precision / recall / F1:** {partial['precision']} / {partial['recall']} / {partial['f1_score']}  \\n"
                f"**Observed-subset accuracy / balanced accuracy:** {partial['accuracy']} / {partial['balanced_accuracy']}  \\n"
            )
            evaluation_status = "**Benchmark status:** INCOMPLETE — do not report observed-subset metrics as final FormAI results."
        summary = (
            "## FormAI-v2 combined evaluation\n"
            f"{evaluation_status}  \\n"
            f"**Mode:** `{mode}`  \\n"
            f"**Programs selected:** {selected_count} / {manifest['programs']} available  \\n"
            f"**Ground-truth programs evaluated:** {evaluation['ground_truth_records']}  \\n"
            f"**Prediction records emitted:** {evaluation['matched_prediction_records']} / {evaluation['ground_truth_records']}  \\n"
            f"**Prediction coverage:** {evaluation['prediction_coverage']}  \\n"
            f"**Classification coverage:** {evaluation['classification_coverage']}  \\n"
            f"{metric_text}"
            f"**Observed-subset TP / FP / FN / TN:** {confusion['tp']} / {confusion['fp']} / {confusion['fn']} / {confusion['tn']}  \\n"
            f"**Missing / inconclusive:** {evaluation['missing_predictions']} / {evaluation['inconclusive_predictions']}  \\n"
            f"**LLM audits completed:** {stage_counts['llm_completed']}/{stage_counts['expected_records']}  \\n"
            f"**cppcheck completed:** {stage_counts['cppcheck_completed']}/{stage_counts['expected_records']}  \\n"
            f"**Evidence reviews completed:** {stage_counts['rsi_completed']}/{stage_counts['expected_records']}  \\n"
            f"**Partial predictions:** {stage_counts['partial_predictions']}  \\n\n"
            "Ground-truth classes are not read by the prediction stages; only filenames are checked before execution to ensure both uploads are the same split."
        )
        return summary, str(prediction_path), str(evaluation_path)
    except ValueError as exc:
        logger.warning("FormAI benchmark preflight failed: %s", exc)
        return f"## Benchmark could not complete\n`{type(exc).__name__}: {exc}`", None, None
    except Exception as exc:
        logger.exception("FormAI Gradio request failed")
        return f"## Benchmark could not complete\n`{type(exc).__name__}: {exc}`", None, None


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Code Checking Assistant") as demo:
        gr.Markdown(
            "# Code Checking Assistant\n"
            "Rule-driven C grading with deterministic analyzer evidence and an independent local LLM review. "
            "Infrastructure failures are reported per stage; one unavailable analyzer does not stop the remaining pipeline."
        )
        with gr.Tab("Grade a C submission"):
            gr.Markdown("Upload the C source and the instructor's `rules.json`. Test files and similarity corpora remain resolved relative to the project root.")
            with gr.Row():
                source_file = gr.File(label="C source file", file_types=[".c"], type="filepath")
                rules_file = gr.File(label="Instructor rules JSON", file_types=[".json"], type="filepath")
            rules_status = gr.Markdown("Upload rules JSON to load an editable copy below.")
            gr.Markdown(
                "The editor is tied to the uploaded file it was loaded from. Edits are used for that same file; "
                "if you upload a different file while an old editor value is still present, the newer upload wins. "
                "The result displays the exact assignment statement and rules fingerprint used for grading."
            )
            rules_editor_source_hash = gr.State("")
            rules_editor = gr.Code(
                label="Editable grading rules (JSON)",
                language="json",
                interactive=True,
                lines=18,
                value="",
            )
            with gr.Accordion("Grading and analyzer parameters", open=True):
                gr.Markdown(
                    "These controls apply to this run only. For metric selection, rubric weights, "
                    "test-set paths, and analyzer-specific thresholds, edit the JSON above: "
                    "`metrics[].enabled`, `metrics[].weight`, and `metrics[].params`."
                )
                with gr.Row():
                    time_limit = gr.Slider(1, settings.DOCKER_MAX_TIMEOUT_S, value=settings.DOCKER_DEFAULT_TIMEOUT_S, step=1, label="Per-test execution timeout (seconds)")
                    memory_limit = gr.Slider(128, 8192, value=512, step=128, label="Docker memory limit (MB)")
                    docker_cpu = gr.Slider(0.25, 16, value=settings.DOCKER_CPU_LIMIT, step=0.25, label="Docker CPU limit (cores)")
                    run_sanitizers = gr.Checkbox(value=settings.ANALYZER_RUN_SANITIZERS, label="Run ASan/UBSan sanitizers")
                with gr.Row():
                    rsi_iterations_ui = gr.Slider(1, settings.LLM_REVIEW_MAX_ITERATIONS, value=settings.LLM_REVIEW_MAX_ITERATIONS, step=1, label="Whole-program RSI iterations")
                    understanding_tokens_ui = gr.Slider(64, 3072, value=settings.LLM_PROGRAM_UNDERSTANDING_MAX_TOKENS, step=64, label="Program-understanding token budget")
                    rsi_tokens_ui = gr.Slider(128, 4096, value=settings.LLM_GLOBAL_RSI_MAX_TOKENS, step=128, label="Each RSI pass token budget")
                    scoring_tokens_ui = gr.Slider(64, 4096, value=settings.LLM_GLOBAL_SCORE_MAX_TOKENS, step=64, label="Metric-scoring token budget")
                with gr.Row():
                    max_findings_ui = gr.Slider(1, 50, value=settings.LLM_GLOBAL_MAX_FINDINGS, step=1, label="Maximum retained findings")
                    gr.Markdown("Each LLM stage makes one schema-constrained request; repair retries are disabled.")
                    source_chars_ui = gr.Slider(1024, 30000, value=settings.LLM_GLOBAL_SOURCE_MAX_CHARS, step=1024, label="Maximum source characters per LLM stage")
                    evidence_chars_ui = gr.Slider(512, 12000, value=settings.LLM_MAX_EVIDENCE_CHARS, step=512, label="Maximum evidence characters in prompts")
                with gr.Row():
                    llm_temperature_ui = gr.Slider(0, 1, value=settings.TEMPERATURE, step=0.05, label="LLM temperature")
                    llm_timeout_ui = gr.Slider(10, 900, value=settings.LLAMA_REQUEST_READ_TIMEOUT_S, step=10, label="LLM request timeout (seconds)")
                with gr.Row():
                    model_name_ui = gr.Textbox(
                        value=settings.MODEL,
                        label="LLM model identifier",
                        info="Used when this app starts llama-server. If reusing an already-running server, enter the model loaded by that server.",
                    )
                    thinking_ui = gr.Checkbox(value=True, label="Enable model thinking", info="Enabled by default; thinking is bounded so the final JSON response has room.")
                with gr.Accordion("Same-model context compression", open=True):
                    gr.Markdown(
                        "Large RSI history is compressed by the same model currently selected above. "
                        "Source code, original line numbers, finding identities, analyzer values, and evidence IDs are protected. "
                        "If compression fails validation, loss-aware deterministic compaction is used."
                    )
                    with gr.Row():
                        context_compression_ui = gr.Checkbox(value=settings.LLM_CONTEXT_COMPRESSION_ENABLED, label="Enable same-model compression")
                        compression_threshold_ui = gr.Slider(256, 20000, value=settings.LLM_CONTEXT_COMPRESSION_THRESHOLD_CHARS, step=256, label="Compress history above (characters)")
            rules_file.change(
                load_rules_editor_from_upload,
                inputs=[rules_file],
                outputs=[rules_editor, time_limit, memory_limit, rules_status, rules_editor_source_hash],
            )
            grade_button = gr.Button("Run grading pipeline", variant="primary")
            grade_summary = gr.Markdown()
            with gr.Row():
                report_json = gr.File(label="Full JSON report")
                report_text = gr.File(label="Readable text report")
            grade_button.click(
                grade_uploaded_submission,
                inputs=[
                    source_file, rules_file, rules_editor, time_limit, memory_limit,
                    rsi_iterations_ui, understanding_tokens_ui, rsi_tokens_ui,
                    scoring_tokens_ui, max_findings_ui,
                    source_chars_ui, evidence_chars_ui, llm_temperature_ui, model_name_ui, llm_timeout_ui, docker_cpu,
                    thinking_ui, run_sanitizers, context_compression_ui,
                    compression_threshold_ui, rules_editor_source_hash,
                ],
                outputs=[grade_summary, report_json, report_text],
            )

        with gr.Tab("FormAI-v2 combined benchmark"):
            gr.Markdown(
                "Choose how many programs to evaluate from the uploaded held-out split. "
                "A seeded random sample is selected from source filenames; the matching labels are filtered only after prediction generation. "
                "The source-input and label filenames must match exactly."
            )
            with gr.Row():
                source_inputs = gr.File(label="test_inputs.jsonl (source only)", file_types=[".jsonl"], type="filepath")
                ground_truth = gr.File(label="test_labels.json (filename preflight + evaluator only)", file_types=[".json"], type="filepath")
            with gr.Row():
                benchmark_mode = gr.Radio(
                    choices=[("Hybrid: LLM + cppcheck", "hybrid"), ("Hybrid + evidence review (RSI)", "hybrid_rsi")],
                    value=settings.UI_DEFAULT_BENCHMARK_MODE, label="Combined prediction mode",
                )
            with gr.Row():
                tool_timeout = gr.Number(value=settings.UI_DEFAULT_TOOL_TIMEOUT_S, precision=0, label="cppcheck timeout (seconds)")
                max_findings = gr.Number(value=settings.UI_DEFAULT_MAX_FINDINGS, precision=0, label="Maximum LLM findings")
                rsi_iterations = gr.Slider(1, settings.LLM_REVIEW_MAX_ITERATIONS, value=settings.UI_DEFAULT_RSI_ITERATIONS, step=1, label="RSI iterations")
                memory_limit = gr.Dropdown(list(settings.UI_DOCKER_MEMORY_CHOICES), value=settings.UI_DEFAULT_MEMORY_LIMIT, label="Docker memory limit")
                cpu_limit = gr.Slider(0.25, 16, value=settings.DOCKER_CPU_LIMIT, step=0.25, label="Docker CPU limit (cores)")
                sample_count = gr.Number(
                    value=settings.UI_DEFAULT_FORM_AI_SAMPLE_COUNT,
                    minimum=0,
                    precision=0,
                    label="Number of samples (0 = all)",
                    info="Select a reproducible random subset; seed is configured in config/settings.py.",
                )
            benchmark_button = gr.Button("Run combined evaluation", variant="primary")
            benchmark_summary = gr.Markdown()
            with gr.Row():
                predictions_file = gr.File(label="Prediction JSONL")
                evaluation_file = gr.File(label="Evaluation JSON")
            benchmark_button.click(
                run_formai_combined,
                inputs=[source_inputs, ground_truth, benchmark_mode, tool_timeout, max_findings, rsi_iterations, memory_limit, cpu_limit, sample_count],
                outputs=[benchmark_summary, predictions_file, evaluation_file],
            )
        gr.Markdown(
            "**Scoring policy:** the LLM evaluates every enabled metric using the assignment, rubric, source, "
            "and analyzer evidence. Completed deterministic analyzer scores are fallbacks only; Python "
            "calculates the configured weighted total from the selected metric scores."
        )
    return demo


def main() -> None:
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT_APP)
    app = build_app()
    app.queue().launch(server_name=settings.UI_HOST, server_port=settings.UI_PORT, show_error=settings.UI_SHOW_ERROR)


if __name__ == "__main__":
    main()
