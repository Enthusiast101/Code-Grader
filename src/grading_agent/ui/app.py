"""Local Gradio interface for C assignment grading and FormAI-v2 evaluation."""
from __future__ import annotations

import json
import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
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
OCEAN_UI_CSS = """
.gradio-container {max-width: 1480px !important; margin: 0 auto !important;}
#app-header {padding: 0.5rem 0 0.2rem 0;}
#grade-button {min-height: 3.1rem; font-size: 1.05rem;}
#generate-tests-button {min-height: 3.1rem; font-size: 1.05rem;}
.compact-note {opacity: 0.78;}
#grading-progress:empty {display: none;}
.grade-progress-card {display: flex; gap: 1rem; align-items: flex-start; margin: 0.8rem 0 1rem; padding: 1rem 1.1rem; border: 1px solid rgba(25, 140, 160, .3); border-radius: 16px; background: linear-gradient(120deg, rgba(14, 116, 144, .1), rgba(20, 184, 166, .08)); overflow: hidden;}
.grade-progress-card.is-done {border-color: rgba(16, 140, 100, .35); background: linear-gradient(120deg, rgba(16, 140, 100, .1), rgba(20, 184, 166, .08));}
.grade-progress-icon {width: 2.4rem; height: 2.4rem; flex: 0 0 2.4rem; display: grid; place-items: center; border-radius: 50%; color: #fff; background: linear-gradient(140deg, #0891b2, #0f766e); font-size: 1.15rem; box-shadow: 0 0 0 6px rgba(8, 145, 178, .1);}
.is-done .grade-progress-icon {background: linear-gradient(140deg, #059669, #0f766e);}
.grade-progress-body {flex: 1; min-width: 0;}
.grade-progress-title {font-weight: 700; margin: 0 0 .2rem;}
.grade-progress-heading {display: flex; align-items: center; justify-content: space-between; gap: 1rem;}
.grade-progress-elapsed {font-variant-numeric: tabular-nums; white-space: nowrap; font-size: .88rem; opacity: .78;}
.grade-progress-description {opacity: .76; margin: 0 0 .75rem; font-size: .92rem;}
.grade-progress-track {height: .38rem; border-radius: 99px; overflow: hidden; background: rgba(100, 116, 139, .18);}
.grade-progress-track span {display: block; width: 38%; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #06b6d4, #14b8a6, #38bdf8); animation: grade-progress-slide 1.7s ease-in-out infinite;}
.is-done .grade-progress-track span {width: 100%; animation: none; background: linear-gradient(90deg, #10b981, #14b8a6);}
.grade-progress-stages {display: flex; flex-wrap: wrap; gap: .45rem; margin: 0 0 .75rem;}
.grade-progress-stage {padding: .2rem .55rem; border: 1px solid rgba(14, 116, 144, .18); border-radius: 99px; background: rgba(255, 255, 255, .35); font-size: .78rem; opacity: .85;}
@keyframes grade-progress-slide {0% {transform: translateX(-120%);} 100% {transform: translateX(360%);}}
@media (prefers-reduced-motion: reduce) {.grade-progress-track span {animation: none; width: 65%;}}
"""


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


def _normalize_legacy_metric_profile(rules: dict[str, Any]) -> dict[str, Any] | None:
    """Migrate the old weighted metric set to the current nine-metric profile."""
    supplied = rules.get("metrics")
    if not isinstance(supplied, list):
        return None
    supplied_by_name = {
        str(item.get("name")): item for item in supplied
        if isinstance(item, dict) and item.get("name")
    }
    legacy_names = {
        "correctness", "syntax", "structure", "modularity", "complexity",
        "logical", "runtime_efficiency", "memory_safety", "readability", "originality",
    }
    # Only migrate this application's known legacy metric profile. Arbitrary
    # instructor metric sets remain fully configurable.
    if "runtime_efficiency" not in supplied_by_name or not set(supplied_by_name).issubset(legacy_names):
        return None

    canonical_path = PROJECT_ROOT / "rules.json"
    try:
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        canonical_metrics = canonical.get("metrics")
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(canonical_metrics, list):
        return None

    migrated: list[dict[str, Any]] = []
    for template in canonical_metrics:
        if not isinstance(template, dict):
            continue
        name = str(template.get("name", ""))
        old = supplied_by_name.get(name, {})
        normalized = json.loads(json.dumps(template))
        params = dict(normalized.get("params", {}))
        if isinstance(old.get("params"), dict):
            params.update(old["params"])
        normalized["params"] = params
        migrated.append(normalized)

    if not migrated:
        return None
    removed = sorted(set(supplied_by_name) - {metric["name"] for metric in migrated})
    added = sorted({metric["name"] for metric in migrated} - set(supplied_by_name))
    rules["metrics"] = migrated
    # This project uses the packaged default metric guidance. Assignment-side
    # rubric overrides from the old editor must not resurrect the old profile.
    rules.pop("rubric", None)
    rules.pop("rubrics", None)
    return {
        "applied": True,
        "profile": "current_nine_metric_point_caps",
        "removed_metrics": removed,
        "added_metrics": added,
    }


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
    lines.append("\n### Metric points")
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
        max_points = number(metric.get("max_points", result.get("max_points", number(metric.get("weight")) * 100.0)))
        if not metric.get("grade_contribution", True):
            metric_score_label = "diagnostic only"
        elif isinstance(result.get("score_points"), (int, float)):
            metric_score_label = f"{number(result['score_points']):.2f}/{max_points:g}"
        elif isinstance(metric_score, (int, float)):
            metric_score_label = f"{number(metric_score) * max_points / 100.0:.2f}/{max_points:g}"
        else:
            metric_score_label = f"N/A (max {max_points:g})"
        source_label = score_source_label(result.get("score_origin"))
        lines.append(f"- **{name}: {metric_score_label}** _(source: {source_label})_")
        lines.append(f"  - Evidence class: {result.get('score_evidence_class', 'unavailable')}; analyzer status: {result.get('status', 'unknown')}")
        rationale = compact_rationale(result.get("score_rationale", ""), max_words=60, max_chars=420)
        if rationale:
            lines.append(f"  - Assessment: {rationale}")
        provenance_detail = result.get("score_provenance", {})
        if isinstance(provenance_detail, dict) and provenance_detail.get("missing_evidence"):
            lines.append("  - Missing evidence: " + "; ".join(str(item) for item in provenance_detail["missing_evidence"]))
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


def _grading_dashboard_cells(report: dict[str, Any], source_code: str):
    """Turn a grading report into concise, spreadsheet-like UI cells."""
    scoring = report.get("scoring", {}) if isinstance(report, dict) else {}
    scoring = scoring if isinstance(scoring, dict) else {}
    grade = scoring.get("score")
    grade_text = f"{float(grade):.2f}/100" if isinstance(grade, (int, float)) else "Not finalized"
    status = str(scoring.get("status", "unknown")).replace("_", " ").title()
    summary = ["## Grading result", f"**Overall grade:** {grade_text}  \n**Status:** {status}"]
    problem = report.get("problem") if isinstance(report, dict) else None
    if isinstance(problem, str) and problem.strip():
        summary.append("**Assignment:** " + " ".join(problem.split())[:220])

    config = report.get("configuration", {}) if isinstance(report, dict) else {}
    config = config if isinstance(config, dict) else {}
    metric_rows = []
    for metric in config.get("enabled_metrics", []):
        if not isinstance(metric, dict):
            continue
        name = str(metric.get("name", "unknown"))
        result = report.get(name, {})
        result = result if isinstance(result, dict) else {}
        maximum = float(metric.get("max_points", result.get("max_points", float(metric.get("weight", 0)) * 100)))
        if not metric.get("grade_contribution", True):
            grade_cell = "Diagnostic"
        elif isinstance(result.get("score_points"), (int, float)):
            grade_cell = f"{float(result['score_points']):.2f}/{maximum:g}"
        elif isinstance(result.get("score"), (int, float)):
            grade_cell = f"{float(result['score']) * maximum / 100:.2f}/{maximum:g}"
        else:
            grade_cell = f"N/A / {maximum:g}"
        metric_rows.append([
            name, grade_cell,
            str(result.get("status", "unknown")).replace("_", " ").title(),
            compact_rationale(result.get("score_rationale", ""), max_words=40, max_chars=260),
        ])

    observed: dict[str, dict[str, Any]] = {}
    observed_by_input: dict[str, dict[str, Any]] = {}
    for metric_name in ("correctness", "logical", "runtime_efficiency", "memory_safety"):
        result = report.get(metric_name, {})
        rows = result.get("results", []) if isinstance(result, dict) else []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("test_id") is not None:
                observed.setdefault(str(row["test_id"]), row)
                input_value = row.get("input")
                if isinstance(input_value, str):
                    observed_by_input.setdefault(input_value, row)

    test_cases = []
    generation = report.get("test_generation", {}) if isinstance(report, dict) else {}
    generation = generation if isinstance(generation, dict) else {}
    suite_path = generation.get("suite_path")
    if isinstance(suite_path, str):
        try:
            suite = json.loads(Path(suite_path).read_text(encoding="utf-8"))
            test_cases = suite.get("test_cases", []) if isinstance(suite, dict) else []
        except (OSError, json.JSONDecodeError):
            test_cases = []
    if not test_cases:
        for metric_name in ("correctness", "logical"):
            result = report.get(metric_name, {})
            rows = result.get("results", []) if isinstance(result, dict) else []
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        test_cases.append(row)

    case_rows = []
    for index, case in enumerate(test_cases, start=1):
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id", case.get("test_id", index)))
        case_input = str(case.get("input", ""))
        # When no frozen generated suite exists, the fallback case records are
        # themselves analyzer result rows. Prefer those direct outcomes, then
        # join suite cases by stable ID and finally by their exact input text.
        observed_row = (
            case if isinstance(case.get("passed"), bool)
            else observed.get(case_id) or observed_by_input.get(case_input, {})
        )
        passed = observed_row.get("passed")
        result_label = "PASS" if passed is True else "FAIL" if passed is False else "Not run"
        tags = case.get("tags", [])
        case_rows.append([
            case_id,
            "Instructor" if case.get("origin") == "instructor" else "LLM-generated",
            ", ".join(str(tag) for tag in tags) if isinstance(tags, list) else "",
            case_input,
            str(case.get("expected", "")),
            str(observed_row.get("actual", "")),
            result_label,
        ])

    total = len(case_rows)
    executed = sum(row[-1] in {"PASS", "FAIL"} for row in case_rows)
    if generation.get("status") == "completed":
        summary.append(
            f"**Test suite:** {total} cases · {generation.get('edge_test_count', 0)} edge cases · {executed} executed"
        )
    elif total:
        summary.append(f"**Test cases:** {total} configured · {executed} executed")
    else:
        summary.append("**Test suite:** No validated test suite was available for this run.")

    if generation.get("status") == "completed":
        generated_count = int(generation.get("generated_test_count", 0) or 0)
        requested_count = int(generation.get("requested_generated_test_count", generated_count) or 0)
        instructor_count = int(generation.get("instructor_test_count", 0) or 0)
        summary.append(
            f"**Generated cases:** {generated_count}/{requested_count} requested · "
            f"{instructor_count} instructor cases"
        )
        oracle_source = generation.get("reference_solution_source")
        if oracle_source and oracle_source != "llm_generated":
            summary.append("**Expected-output oracle:** trusted instructor reference")
        shortfall = int(generation.get("generated_test_count_shortfall", 0) or 0)
        if shortfall:
            summary.append(f"**Generation note:** {shortfall} requested cases were not produced.")
        rejected_candidates = int(generation.get("rejected_generated_test_count", 0) or 0)
        if rejected_candidates:
            summary.append(
                f"**Generation note:** {rejected_candidates} invalid LLM candidate(s) were rejected and replaced."
            )
        ambiguities = generation.get("unresolved_ambiguities", [])
        if isinstance(ambiguities, list) and ambiguities:
            summary.append(
                "**Oracle note:** expected outputs use the trusted instructor reference. "
                + "; ".join(str(item) for item in ambiguities[:2])[:300]
            )
    else:
        generation_error = generation.get("error") or generation.get("reason")
        if generation_error:
            summary.append(
                "**Automatic generation:** unavailable — "
                + str(generation_error).replace(chr(10), " ")[:300]
            )

    finding_rows = []
    findings = report.get("llm_rsi_findings", []) if isinstance(report, dict) else []
    if isinstance(findings, list):
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            start = finding.get("line_start")
            end = finding.get("line_end")
            location = f"{start}" if start is not None else "—"
            if end is not None and end != start:
                location += f"–{end}"
            finding_rows.append([
                str(finding.get("severity", "unknown")).upper(),
                str(finding.get("metric", "unmapped")),
                location,
                str(finding.get("title", "Issue")),
                compact_rationale(finding.get("description", ""), max_words=45, max_chars=300),
            ])
    return "\n\n".join(summary), metric_rows, case_rows, finding_rows, source_code


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
        migration = _normalize_legacy_metric_profile(parsed)
        time_limit = float(parsed.get("time_limit_s", default_time))
        memory_limit = int(parsed.get("memory_limit_mb", default_memory))
        digest = _sha256_text(content)
        editor_content = json.dumps(parsed, indent=2) if migration else content
        status = f"Loaded {path.name}; edit the JSON or use the controls below."
        if migration:
            status += " Legacy metrics updated to the current nine-metric profile; runtime_efficiency removed."
        return editor_content, time_limit, memory_limit, status, digest
    except Exception as exc:
        logger.warning("Could not load rules into Gradio editor: %s", exc)
        return "", default_time, default_memory, f"Could not load rules: {type(exc).__name__}: {exc}", ""


def load_source_editor_from_upload(source_upload: Any):
    """Load an uploaded C file into the syntax-highlighted editor."""
    if source_upload is None:
        return "", "", "Paste C source below or upload a `.c` file."
    try:
        path = _upload_path(source_upload, "C source")
        if path.suffix.lower() != ".c":
            raise ValueError("The submission must be a .c file.")
        if path.stat().st_size > settings.UI_MAX_UPLOAD_BYTES:
            raise ValueError(f"The source file exceeds the {settings.UI_MAX_UPLOAD_BYTES} byte project limit.")
        content = path.read_text(encoding="utf-8")
        return content, _sha256_text(content), f"Loaded `{path.name}` into the C editor. You can edit it before grading."
    except Exception as exc:
        return "", "", f"Could not load source: {type(exc).__name__}: {exc}"


def _select_submission_inputs(
    source_upload: Any,
    source_code_editor: str,
    source_editor_hash: str,
    rules_upload: Any,
    rules_json_text: str,
    rules_editor_source_hash: str,
) -> tuple[str, str, str, Path | None, str, bool]:
    """Resolve the exact source and rules content used by both UI stages."""
    uploaded_source_path = _upload_path(source_upload, "C source") if source_upload is not None else None
    if uploaded_source_path is not None and uploaded_source_path.suffix.lower() != ".c":
        raise ValueError("The submission must be a .c file.")
    if uploaded_source_path is not None and uploaded_source_path.stat().st_size > settings.UI_MAX_UPLOAD_BYTES:
        raise ValueError(f"The source file exceeds the {settings.UI_MAX_UPLOAD_BYTES} byte project limit.")
    editor_source = str(source_code_editor or "")
    if uploaded_source_path is not None:
        uploaded_source = uploaded_source_path.read_text(encoding="utf-8")
        source_code = (
            editor_source
            if source_editor_hash and source_editor_hash == _sha256_text(uploaded_source) and editor_source
            else uploaded_source
        )
    elif editor_source.strip():
        source_code = editor_source
    else:
        raise ValueError("Upload a C source file or paste C code into the editor.")
    if len(source_code.encode("utf-8")) > settings.UI_MAX_UPLOAD_BYTES:
        raise ValueError(f"The source code exceeds the {settings.UI_MAX_UPLOAD_BYTES} byte project limit.")

    editor_text = str(rules_json_text or "").strip()
    uploaded_rules_path: Path | None = None
    uploaded_rules_text = ""
    uploaded_rules_hash = ""
    if rules_upload is not None:
        uploaded_rules_path = _upload_path(rules_upload, "rules JSON")
        uploaded_rules_text = uploaded_rules_path.read_text(encoding="utf-8")
        uploaded_rules_hash = _sha256_text(uploaded_rules_text)

    if uploaded_rules_path is not None:
        if rules_editor_source_hash and rules_editor_source_hash == uploaded_rules_hash and editor_text:
            raw_rules = editor_text
            selected_rules_source = "edited Gradio rules editor"
        else:
            raw_rules = uploaded_rules_text
            selected_rules_source = "uploaded rules file"
    elif editor_text:
        raw_rules = editor_text
        selected_rules_source = "manually entered Gradio rules editor"
    else:
        raise ValueError("Upload a rules JSON file or enter rules in the editor.")

    stale_editor = bool(
        uploaded_rules_path is not None
        and rules_editor_source_hash
        and rules_editor_source_hash != uploaded_rules_hash
    )
    return (
        source_code, raw_rules, selected_rules_source,
        uploaded_rules_path, uploaded_rules_hash, stale_editor,
    )


def generate_test_cases_for_submission(
    source_upload: Any,
    source_code_editor: str,
    source_editor_hash: str,
    rules_upload: Any,
    rules_json_text: str,
    rules_editor_source_hash: str,
    time_limit_s: float,
    memory_limit_mb: float,
    docker_cpu_limit: float,
    temperature: float,
    model_name: str,
    read_timeout_s: float,
    max_generated_cases: float,
    additional_test_focus: str,
    enable_thinking: bool,
):
    """Run the explicit first UI step: understand source, then generate tests."""
    stage_dir: Path | None = None
    generation_process = None
    request_event_token = None
    generation_request_events: list[dict[str, Any]] = []
    try:
        (
            source_code, raw_rules, selected_rules_source,
            uploaded_rules_path, uploaded_rules_hash, stale_editor,
        ) = _select_submission_inputs(
            source_upload, source_code_editor, source_editor_hash,
            rules_upload, rules_json_text, rules_editor_source_hash,
        )
        rules = json.loads(raw_rules)
        if not isinstance(rules, dict) or not isinstance(rules.get("submission"), dict):
            raise ValueError("The rules JSON must contain a submission object.")
        _normalize_legacy_metric_profile(rules)
        if isinstance(max_generated_cases, bool) or not 1 <= int(max_generated_cases) <= 60:
            raise ValueError("Generated test-case target must be between 1 and 60. Set the slider to 0 to grade without generating cases.")
        if not isinstance(additional_test_focus, str) or len(additional_test_focus) > 500:
            raise ValueError("Test variation focus must be at most 500 characters.")
        for name, value in (("time_limit_s", time_limit_s), ("memory_limit_mb", memory_limit_mb), ("docker_cpu_limit", docker_cpu_limit)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number.")
        if int(memory_limit_mb) != memory_limit_mb:
            raise ValueError("Docker memory limit must be a whole number of MB.")
        if not 0.0 <= float(temperature) <= 1.0:
            raise ValueError("LLM temperature must be between 0 and 1.")
        if not 10 <= float(read_timeout_s) <= 900:
            raise ValueError("LLM read timeout must be between 10 and 900 seconds.")
        model_name = str(model_name or "").strip()
        if not model_name:
            raise ValueError("Model name cannot be empty.")

        test_generation = rules.setdefault("test_generation", {})
        if not isinstance(test_generation, dict):
            raise ValueError("test_generation in rules must be a JSON object.")
        test_generation["enabled"] = int(max_generated_cases) > 0
        test_generation["max_cases"] = int(max_generated_cases)
        test_generation["additional_focus"] = additional_test_focus.strip()
        rules["time_limit_s"] = float(time_limit_s)
        rules["memory_limit_mb"] = int(memory_limit_mb)
        rules["_run_metadata"] = {
            "selection": {
                "selected_from": selected_rules_source,
                "uploaded_file_name": uploaded_rules_path.name if uploaded_rules_path else None,
                "uploaded_file_sha256": uploaded_rules_hash or None,
                "selected_rules_sha256": _sha256_text(raw_rules),
                "editor_was_stale": stale_editor,
            }
        }

        run_id = uuid.uuid4().hex[:12]
        stage_dir = settings.GRADIO_STAGING_DIR / run_id
        stage_dir.mkdir(parents=True, exist_ok=False)
        staged_source = stage_dir / "submission.c"
        staged_source.write_text(source_code, encoding="utf-8")
        rules["submission"]["source_file"] = str(staged_source.relative_to(PROJECT_ROOT))
        staged_rules = stage_dir / "rules.json"
        staged_rules.write_text(json.dumps(rules, indent=2), encoding="utf-8")

        from grading_agent.config.rule_loader import load_rules
        from grading_agent.llm.server import initiate_server
        from grading_agent.sandbox.docker_sandbox import DockerSandbox
        from grading_agent.testing.test_generation import prepare_assignment_test_suite
        from grading_agent.llm.reviewer import _LLM_REQUEST_EVENTS, _request_json

        normalized_rules = load_rules(staged_rules, project_root=PROJECT_ROOT)
        runtime_options = {
            "MODEL": model_name,
            "TEMPERATURE": float(temperature),
            "LLAMA_REQUEST_TIMEOUT": (settings.LLAMA_REQUEST_CONNECT_TIMEOUT_S, float(read_timeout_s)),
            "LLAMA_REASONING_ENABLED": bool(enable_thinking),
            "DOCKER_CPU_LIMIT": float(docker_cpu_limit),
        }
        with settings.runtime_overrides(runtime_options):
            request_event_token = _LLM_REQUEST_EVENTS.set(generation_request_events)
            if not _generation_server_is_ready():
                generation_process = initiate_server()
            requested_cases = int(max_generated_cases)
            replacement_reserve = min(max(0, 80 - requested_cases), max(3, (requested_cases + 3) // 4))
            candidate_count = requested_cases + replacement_reserve
            generation_token_budget = max(
                int(settings.runtime_value("LLM_GLOBAL_SCORE_MAX_TOKENS") or 0),
                min(settings.LLAMA_CONTEXT_SIZE, 2048 + candidate_count * 192),
            )

            def generation_request(messages, schema, stage_name):
                parsed, _raw = _request_json(
                    messages,
                    max_tokens=generation_token_budget,
                    enable_thinking=None,
                    response_schema=schema,
                    schema_name=stage_name,
                )
                return parsed

            with DockerSandbox(
                memory_limit=f"{int(memory_limit_mb)}m",
                cpu_limit=float(docker_cpu_limit),
            ) as sandbox:
                result = prepare_assignment_test_suite(
                    normalized_rules,
                    sandbox=sandbox,
                    request_json=generation_request,
                    project_root=PROJECT_ROOT,
                    model_id=model_name,
                    generation_seed=settings.SEED,
                    temperature=float(temperature),
                    submission_source=source_code,
                    submission_filename=staged_source.name,
                )
            result["request_events"] = generation_request_events

        if result.get("status") != "completed":
            raise ValueError(result.get("reason") or "Test generation did not complete.")
        suite_path = Path(str(result.get("suite_path", ""))).resolve()
        if not suite_path.is_file() or not suite_path.is_relative_to(PROJECT_ROOT.resolve()):
            raise ValueError("The generated test-suite artifact is missing or outside the project directory.")
        frozen = json.loads(suite_path.read_text(encoding="utf-8"))
        cases = frozen.get("test_cases", []) if isinstance(frozen, dict) else []
        if not isinstance(cases, list) or not cases:
            raise ValueError("The generated test-suite artifact contains no test cases.")
        metric_paths = result.get("metric_testset_paths", {})
        if not isinstance(metric_paths, dict) or not metric_paths:
            raise ValueError("No execution metrics were bound to the generated test suite.")
        submission_analysis = result.get("submission_analysis")
        if (
            not isinstance(submission_analysis, dict)
            or not isinstance(submission_analysis.get("implementation_summary"), str)
            or not submission_analysis.get("implementation_summary", "").strip()
        ):
            raise ValueError("The LLM did not complete its analysis of the submitted C source. No grading was started.")
        manifest = {
            "status": "completed",
            "source_sha256": _sha256_text(source_code),
            "rules_sha256": _sha256_text(raw_rules),
            "requested_generated_test_count": int(max_generated_cases),
            "additional_test_focus": additional_test_focus.strip(),
            "assignment_fingerprint": result.get("assignment_fingerprint"),
            "suite_path": str(suite_path),
            "metric_testset_paths": {str(key): str(value) for key, value in metric_paths.items()},
            "instructor_test_count": int(result.get("instructor_test_count", 0)),
            "generated_test_count": int(result.get("generated_test_count", 0)),
            "total_test_count": int(result.get("total_test_count", len(cases))),
            "edge_test_count": int(result.get("edge_test_count", 0)),
            "reference_solution_source": result.get("reference_solution_source"),
            "oracle_kind": result.get("oracle_kind"),
            "oracle_caveat": result.get("oracle_caveat"),
            "source_analysis": submission_analysis,
        }
        if manifest["generated_test_count"] != int(max_generated_cases):
            raise ValueError(
                f"The LLM generated {manifest['generated_test_count']} additional cases, "
                f"but {int(max_generated_cases)} were requested. Please generate again."
            )

        preview_rows = []
        for index, case in enumerate(cases, 1):
            if not isinstance(case, dict):
                continue
            preview_rows.append([
                str(case.get("id", index)),
                "Instructor" if case.get("origin") == "instructor" else "LLM-generated",
                ", ".join(str(tag) for tag in case.get("tags", []) if isinstance(tag, str)),
                str(case.get("input", "")),
                str(case.get("expected", "")),
                "",
                "Ready to grade",
            ])
        implementation_summary = submission_analysis.get("implementation_summary", "").strip()
        likely_weak_points = submission_analysis.get("likely_weak_points", [])
        weak_point_text = "\n".join(
            f"- {item}" for item in likely_weak_points if isinstance(item, str) and item.strip()
        ) or "- No specific weakness was identified by the analysis stage."
        status = (
            f"## Test cases generated successfully\n\n"
            f"**{manifest['generated_test_count']} additional LLM-generated cases** + "
            f"**{manifest['instructor_test_count']} unique instructor cases** = "
            f"**{manifest['total_test_count']} cases ready for grading.**\n\n"
            "### What the LLM understood from the submitted C code\n\n"
            f"{implementation_summary}\n\n"
            f"**Potential weaknesses targeted:**\n{weak_point_text}\n\n"
            "Expected outputs were produced by the reference solution. Review the combined suite below, "
            "then click **Grade Submission**."
        )
        return status, manifest, preview_rows, gr.update(interactive=True)
    except Exception as exc:
        logger.exception("Test-case generation request failed")
        return (
            f"## Test-case generation failed\n\n`{type(exc).__name__}: {exc}`\n\n"
            "No grading was started. Correct the inputs or model settings and generate the suite again.",
            {},
            [],
            gr.update(interactive=False),
        )
    finally:
        if request_event_token is not None:
            try:
                from grading_agent.llm.reviewer import _LLM_REQUEST_EVENTS
                _LLM_REQUEST_EVENTS.reset(request_event_token)
            except Exception:
                pass
        if generation_process is not None:
            from grading_agent.main import _stop_llm_server
            _stop_llm_server(generation_process)
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)


def invalidate_generated_test_suite(max_generated_cases: float = 0):
    """Clear stale suite state and gate grading based on the requested case count."""
    return (
        (
            "No generated test cases requested. Grading can run with instructor-configured tests."
            if int(max_generated_cases or 0) == 0
            else "**The generated suite is no longer current.** Click **Generate Test Cases** again before grading."
        ),
        {},
        [],
        gr.update(interactive=int(max_generated_cases or 0) == 0),
    )


def _generation_server_is_ready() -> bool:
    """Small local readiness check used only by the explicit generate action."""
    import requests
    try:
        response = requests.get(settings.LLAMA_HEALTH_URL, timeout=settings.LLAMA_HEALTH_TIMEOUT_S)
        return response.status_code == 200
    except requests.RequestException:
        return False


def grade_uploaded_submission(
    source_upload: Any,
    source_code_editor: str,
    source_editor_hash: str,
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
    max_generated_cases: float = 60,
    additional_test_focus: str = "",
    generated_suite_state: dict[str, Any] | None = None,
):
    """Stage files and grade only against the suite created by the Generate button."""
    stage_dir: Path | None = None
    source_code = ""
    try:
        (
            source_code, raw_rules, selected_rules_source,
            uploaded_path, uploaded_hash, stale_rules_editor,
        ) = _select_submission_inputs(
            source_upload, source_code_editor, source_editor_hash,
            rules_upload, rules_json_text, rules_editor_source_hash,
        )

        rules = json.loads(raw_rules)
        if not isinstance(rules, dict) or not isinstance(rules.get("submission"), dict):
            raise ValueError("The rules JSON must contain a submission object.")
        metric_profile_migration = _normalize_legacy_metric_profile(rules)
        selected_rules_hash = _sha256_text(raw_rules)
        rules["_run_metadata"] = {
            "selection": {
                "selected_from": selected_rules_source,
                "uploaded_file_name": uploaded_path.name if uploaded_path else None,
                "uploaded_file_sha256": uploaded_hash or None,
                "editor_base_sha256": rules_editor_source_hash or None,
                "selected_rules_sha256": selected_rules_hash,
                "editor_was_stale": stale_rules_editor,
            },
            "metric_profile_migration": metric_profile_migration,
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

        if isinstance(max_generated_cases, bool) or not 0 <= int(max_generated_cases) <= 60:
            raise ValueError("Generated test-case target must be between 0 and 60.")
        if not isinstance(additional_test_focus, str) or len(additional_test_focus) > 500:
            raise ValueError("Test variation focus must be at most 500 characters.")
        test_generation = rules.setdefault("test_generation", {})
        if not isinstance(test_generation, dict):
            raise ValueError("test_generation in rules must be a JSON object.")
        test_generation["enabled"] = int(max_generated_cases) > 0
        test_generation["max_cases"] = int(max_generated_cases)
        test_generation["additional_focus"] = additional_test_focus.strip()

        if int(max_generated_cases) == 0:
            test_generation.pop("pre_generated_suite", None)
        else:
            # The UI uses two explicit actions. Never call the LLM generator from
            # the grading action: require and verify the artifact created earlier.
            if not isinstance(generated_suite_state, dict) or generated_suite_state.get("status") != "completed":
                raise ValueError("Generate test cases first. Select the requested count and click 'Generate Test Cases' before grading.")
            if generated_suite_state.get("source_sha256") != _sha256_text(source_code):
                raise ValueError("The C submission changed after test generation. Generate test cases again before grading.")
            if generated_suite_state.get("rules_sha256") != _sha256_text(raw_rules):
                raise ValueError("The assignment rules changed after test generation. Generate test cases again before grading.")
            if int(generated_suite_state.get("requested_generated_test_count", -1)) != int(max_generated_cases):
                raise ValueError("The selected testcase count changed after generation. Generate test cases again before grading.")
            if str(generated_suite_state.get("additional_test_focus", "")) != additional_test_focus.strip():
                raise ValueError("The test variation focus changed after generation. Generate test cases again before grading.")

            suite_path = Path(str(generated_suite_state.get("suite_path", ""))).resolve()
            if not suite_path.is_file() or not suite_path.is_relative_to(PROJECT_ROOT.resolve()):
                raise ValueError("The generated test suite is missing or outside the project directory. Generate it again.")
            metric_paths = generated_suite_state.get("metric_testset_paths")
            if not isinstance(metric_paths, dict) or not metric_paths:
                raise ValueError("The generated test suite has no metric bindings. Generate it again.")
            execution_analyzers = {"io_tests", "edge_case_runner", "memory_safety", "runtime_profile"}
            bound_paths: dict[str, str] = {}
            for metric in rules.get("metrics", []):
                if not isinstance(metric, dict) or not metric.get("enabled", True) or metric.get("analyzer") not in execution_analyzers:
                    continue
                metric_name = str(metric.get("name", ""))
                path_value = metric_paths.get(metric_name)
                if not isinstance(path_value, str):
                    raise ValueError(f"The generated suite has no testcase file for metric '{metric_name}'. Generate it again.")
                resolved_path = Path(path_value).resolve()
                if not resolved_path.is_file() or not resolved_path.is_relative_to(PROJECT_ROOT.resolve()):
                    raise ValueError(f"The generated testcase file for '{metric_name}' is missing or invalid.")
                relative_path = resolved_path.relative_to(PROJECT_ROOT.resolve()).as_posix()
                metric.setdefault("params", {})["testset_path"] = relative_path
                bound_paths[metric_name] = relative_path

            test_generation["pre_generated_suite"] = {
                "status": "completed",
                "assignment_fingerprint": generated_suite_state.get("assignment_fingerprint"),
                "suite_path": str(suite_path),
                "metric_testset_paths": bound_paths,
                "instructor_test_count": int(generated_suite_state.get("instructor_test_count", 0)),
                "generated_test_count": int(generated_suite_state.get("generated_test_count", 0)),
                "requested_generated_test_count": int(generated_suite_state.get("requested_generated_test_count", 0)),
                "total_test_count": int(generated_suite_state.get("total_test_count", 0)),
                "edge_test_count": int(generated_suite_state.get("edge_test_count", 0)),
                "reference_solution_source": generated_suite_state.get("reference_solution_source"),
                "oracle_kind": generated_suite_state.get("oracle_kind"),
                "oracle_caveat": generated_suite_state.get("oracle_caveat"),
                "source_analysis": generated_suite_state.get("source_analysis", {}),
            }
            rules["_run_metadata"]["generated_test_suite"] = test_generation["pre_generated_suite"]

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
        staged_source.write_text(source_code, encoding="utf-8")
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
            "LLAMA_REASONING_ENABLED": bool(enable_thinking),
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
        summary, metric_rows, case_rows, finding_rows, graded_source = _grading_dashboard_cells(report, source_code)
        return summary, metric_rows, case_rows, finding_rows, graded_source, str(json_path), str(text_path)
    except Exception as exc:
        logger.exception("Gradio grading request failed")
        return (
            f"## Could not complete grading\n\n{type(exc).__name__}: {exc}",
            [], [], [], source_code, None, None,
        )
    finally:
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)


def grade_uploaded_submission_with_progress(*args: Any):
    """Stream a live visual indicator, then the completed grading result."""
    started = time.monotonic()

    def elapsed_label() -> str:
        elapsed = max(0, int(time.monotonic() - started))
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def progress_html(*, done: bool = False, failed: bool = False) -> str:
        elapsed = elapsed_label()
        if failed:
            title = "Grading needs attention"
            description = "The run stopped before a complete result was produced. See the overview for details."
            icon, card_class = "!", ""
            progress = ""
        elif done:
            title = "Grading complete"
            description = "Test variations, analysis, and consolidated metric scoring have finished."
            icon, card_class = "✓", " is-done"
            progress = '<div class="grade-progress-track" aria-hidden="true"><span></span></div>'
        else:
            title = "Grading in progress"
            description = "Preparing generated test variations and evaluating the submission."
            icon, card_class = "↻", ""
            progress = (
                '<div class="grade-progress-stages" aria-label="Pipeline stages">'
                '<span class="grade-progress-stage">Test variations</span>'
                '<span class="grade-progress-stage">Sandbox runs</span>'
                '<span class="grade-progress-stage">Source review</span>'
                '<span class="grade-progress-stage">Metric points</span></div>'
                '<div class="grade-progress-track" aria-hidden="true"><span></span></div>'
            )
        return (
            f'<section class="grade-progress-card{card_class}" role="status" aria-live="polite">'
            f'<div class="grade-progress-icon" aria-hidden="true">{icon}</div>'
            '<div class="grade-progress-body">'
            '<div class="grade-progress-heading">'
            f'<div class="grade-progress-title">{title}</div>'
            f'<div class="grade-progress-elapsed">Elapsed&nbsp; {elapsed}</div>'
            '</div>'
            f'<p class="grade-progress-description">{description}</p>{progress}'
            '</div></section>'
        )

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="grading-ui") as executor:
        future = executor.submit(grade_uploaded_submission, *args)
        yield (progress_html(), *[gr.skip() for _ in range(7)])
        while not future.done():
            time.sleep(1)
            if not future.done():
                yield (progress_html(), *[gr.skip() for _ in range(7)])
        result = future.result()
    failed = isinstance(result[0], str) and result[0].startswith("## Could not complete grading")
    yield (progress_html(done=not failed, failed=failed), *result)


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
    with gr.Blocks(title="Code Grader") as demo:
        gr.Markdown(
            "# Code Grader\n"
            "Upload your C submission and assignment rules. Generate additional test cases if needed, or grade with the instructor-configured tests.",
            elem_id="app-header",
        )
        with gr.Tab("Grade a C submission"):
            with gr.Row():
                source_file = gr.File(label="C source file", file_types=[".c"], type="filepath")
                rules_file = gr.File(label="Instructor rules JSON", file_types=[".json"], type="filepath")
            source_status = gr.Markdown("Paste C code or upload a `.c` file.", elem_classes=["compact-note"])
            source_editor_hash = gr.State("")
            source_editor = gr.Code(
                label="Submission source", language="c", interactive=True,
                lines=24, max_lines=42, value="", elem_id="submission-source",
            )
            with gr.Row():
                max_generated_cases = gr.Slider(
                    0, 60, value=0, step=1, label="Generated test case target",
                    info="Requests this many additional LLM-generated cases. Existing instructor cases are kept and included too.",
                )
                additional_test_focus = gr.Textbox(
                    label="Test variation focus (optional)",
                    placeholder="For example: duplicate values, zero values, and boundary sizes",
                    lines=2, max_lines=3, max_length=500,
                )
            with gr.Row():
                gr.Markdown("Generate additional cases (optional):")
                generate_button = gr.Button(
                    "Generate Test Cases", variant="primary", elem_id="generate-tests-button"
                )
            rules_editor_source_hash = gr.State("")
            with gr.Accordion("Assignment rules", open=False):
                rules_status = gr.Markdown("Upload rules JSON to load an editable copy.", elem_classes=["compact-note"])
                rules_editor = gr.Code(
                    label="Rules JSON", language="json", interactive=True,
                    lines=18, max_lines=36, value="",
                )
                gr.Markdown(
                    "Rules and testcase paths resolve from the repository root.",
                    elem_classes=["compact-note"],
                )
            with gr.Accordion("Execution settings", open=False):
                with gr.Row():
                    time_limit = gr.Slider(1, settings.DOCKER_MAX_TIMEOUT_S, value=settings.DOCKER_DEFAULT_TIMEOUT_S, step=1, label="Per-test timeout (seconds)")
                    memory_limit = gr.Slider(128, 8192, value=512, step=128, label="Docker memory (MB)")
                    docker_cpu = gr.Slider(0.25, 16, value=settings.DOCKER_CPU_LIMIT, step=0.25, label="Docker CPU (cores)")
                    run_sanitizers = gr.Checkbox(value=settings.ANALYZER_RUN_SANITIZERS, label="Run ASan/UBSan")
            with gr.Accordion("LLM and context options", open=True):
                with gr.Row():
                    rsi_iterations_ui = gr.Slider(1, settings.LLM_REVIEW_MAX_ITERATIONS, value=settings.LLM_REVIEW_MAX_ITERATIONS, step=1, label="RSI iterations")
                    llm_temperature_ui = gr.Slider(0, 1, value=settings.TEMPERATURE, step=0.05, label="LLM temperature")
                    llm_timeout_ui = gr.Slider(10, 900, value=settings.LLAMA_REQUEST_READ_TIMEOUT_S, step=10, label="LLM timeout (seconds)")
                with gr.Row():
                    model_name_ui = gr.Textbox(value=settings.MODEL, label="LLM model identifier")
                    thinking_ui = gr.Checkbox(value=True, label="Enable model thinking")
                with gr.Row():
                    max_findings_ui = gr.Slider(1, 50, value=settings.LLM_GLOBAL_MAX_FINDINGS, step=1, label="Maximum findings")
                    source_chars_ui = gr.Slider(1024, 30000, value=settings.LLM_GLOBAL_SOURCE_MAX_CHARS, step=1024, label="Maximum source characters")
                    evidence_chars_ui = gr.Slider(512, 12000, value=settings.LLM_MAX_EVIDENCE_CHARS, step=512, label="Maximum evidence characters")
                with gr.Row():
                    understanding_tokens_ui = gr.Slider(64, 3072, value=settings.LLM_PROGRAM_UNDERSTANDING_MAX_TOKENS, step=64, label="Understanding token budget")
                    rsi_tokens_ui = gr.Slider(128, 4096, value=settings.LLM_GLOBAL_RSI_MAX_TOKENS, step=128, label="RSI token budget")
                    scoring_tokens_ui = gr.Slider(64, 4096, value=settings.LLM_GLOBAL_SCORE_MAX_TOKENS, step=64, label="Scoring token budget")
                with gr.Row():
                    context_compression_ui = gr.Checkbox(value=settings.LLM_CONTEXT_COMPRESSION_ENABLED, label="Enable same-model compression")
                    compression_threshold_ui = gr.Slider(256, 20000, value=settings.LLM_CONTEXT_COMPRESSION_THRESHOLD_CHARS, step=256, label="Compression threshold")
            generated_suite_state = gr.State({})
            generation_status = gr.Markdown(
                "No generated test cases requested. Grading can run with instructor-configured tests."
            )
            grade_button = gr.Button(
                "Grade Submission", variant="secondary", elem_id="grade-button", interactive=True,
            )
            grading_progress = gr.HTML(value="", elem_id="grading-progress")
            with gr.Tabs():
                with gr.Tab("Overview"):
                    grade_summary = gr.Markdown()
                    metrics_table = gr.Dataframe(
                        headers=["Metric", "Points awarded", "Status", "Assessment"],
                        datatype=["str", "str", "str", "str"],
                        value=[], interactive=False, wrap=True, show_search="filter",
                        label="Metric results", max_chars=360,
                    )
                with gr.Tab("Test cases"):
                    with gr.Accordion("Generated and graded test cases", open=False):
                        test_cases_table = gr.Dataframe(
                            headers=["Case", "Origin", "Tags", "Input", "Expected", "Actual", "Result"],
                            datatype=["str", "str", "str", "str", "str", "str", "str"],
                            value=[], interactive=False, wrap=True, show_search="filter",
                            label="Case details", max_chars=1200,
                        )
                with gr.Tab("Findings"):
                    findings_table = gr.Dataframe(
                        headers=["Severity", "Metric", "Lines", "Finding", "Evidence"],
                        datatype=["str", "str", "str", "str", "str"],
                        value=[], interactive=False, wrap=True, show_search="filter",
                        label="Validated source findings", max_chars=420,
                    )
                with gr.Tab("Submitted source"):
                    graded_source = gr.Code(
                        label="Exact C source graded", language="c", interactive=False,
                        lines=24, max_lines=48,
                    )
                with gr.Tab("Report files"):
                    with gr.Row():
                        report_json = gr.File(label="JSON report")
                        report_text = gr.File(label="Readable report")
            source_file.change(
                load_source_editor_from_upload,
                inputs=[source_file],
                outputs=[source_editor, source_editor_hash, source_status],
                show_progress="hidden",
            ).then(
                invalidate_generated_test_suite,
                inputs=[max_generated_cases],
                outputs=[generation_status, generated_suite_state, test_cases_table, grade_button],
                show_progress="hidden",
            )
            source_editor.change(
                invalidate_generated_test_suite,
                inputs=[max_generated_cases],
                outputs=[generation_status, generated_suite_state, test_cases_table, grade_button],
                show_progress="hidden",
            )
            rules_file.change(
                load_rules_editor_from_upload,
                inputs=[rules_file],
                outputs=[rules_editor, time_limit, memory_limit, rules_status, rules_editor_source_hash],
                show_progress="hidden",
            ).then(
                invalidate_generated_test_suite,
                inputs=[max_generated_cases],
                outputs=[generation_status, generated_suite_state, test_cases_table, grade_button],
                show_progress="hidden",
            )
            rules_editor.change(
                invalidate_generated_test_suite,
                inputs=[max_generated_cases],
                outputs=[generation_status, generated_suite_state, test_cases_table, grade_button],
                show_progress="hidden",
            )
            max_generated_cases.change(
                invalidate_generated_test_suite,
                inputs=[max_generated_cases],
                outputs=[generation_status, generated_suite_state, test_cases_table, grade_button],
                show_progress="hidden",
            )
            additional_test_focus.change(
                invalidate_generated_test_suite,
                inputs=[max_generated_cases],
                outputs=[generation_status, generated_suite_state, test_cases_table, grade_button],
                show_progress="hidden",
            )
            generate_button.click(
                generate_test_cases_for_submission,
                inputs=[
                    source_file, source_editor, source_editor_hash,
                    rules_file, rules_editor, rules_editor_source_hash,
                    time_limit, memory_limit, docker_cpu,
                    llm_temperature_ui, model_name_ui, llm_timeout_ui,
                    max_generated_cases, additional_test_focus, thinking_ui,
                ],
                outputs=[generation_status, generated_suite_state, test_cases_table, grade_button],
                show_progress="full",
                scroll_to_output=True,
            )
            grade_button.click(
                grade_uploaded_submission_with_progress,
                inputs=[
                    source_file, source_editor, source_editor_hash,
                    rules_file, rules_editor, time_limit, memory_limit,
                    rsi_iterations_ui, understanding_tokens_ui, rsi_tokens_ui,
                    scoring_tokens_ui, max_findings_ui, source_chars_ui, evidence_chars_ui,
                    llm_temperature_ui, model_name_ui, llm_timeout_ui, docker_cpu,
                    thinking_ui, run_sanitizers, context_compression_ui,
                    compression_threshold_ui, rules_editor_source_hash,
                    max_generated_cases, additional_test_focus,
                    generated_suite_state,
                ],
                outputs=[grading_progress, grade_summary, metrics_table, test_cases_table, findings_table, graded_source, report_json, report_text],
                show_progress="hidden", scroll_to_output=True,
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

    return demo


def main() -> None:
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT_APP)
    app = build_app()
    app.queue().launch(
        server_name=settings.UI_HOST,
        server_port=settings.UI_PORT,
        show_error=settings.UI_SHOW_ERROR,
        theme=gr.themes.Ocean(),
        css=OCEAN_UI_CSS,
    )


if __name__ == "__main__":
    main()
