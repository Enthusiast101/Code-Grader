"""Local Gradio interface for C assignment grading and FormAI-v2 evaluation."""
from __future__ import annotations

import contextvars
import html
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import uuid
from collections.abc import Callable, Sequence
from typing import Any

import gradio as gr

from grading_agent.main import PROJECT_ROOT, grade_with_rules_file
from grading_agent.config import settings
from grading_agent.reporting.rationale import compact_rationale, evidence_summary, score_source_label
from grading_agent.ui.formai_tab import build_formai_tab, preview_allocation

logger = logging.getLogger("grading_agent.ui")
OCEAN_UI_CSS = """
/* Fixed width: without width:100% the container shrinks to its content and
   jumps wider when an uploaded file name or table appears. */
.gradio-container {width: 100% !important; max-width: 1480px !important; margin: 0 auto !important;}
#settings-open-button {min-height: 3.1rem;}
#settings-done-button {margin-top: .75rem;}
#app-header {padding: 0.5rem 0 0.2rem 0;}
#grade-button, #generate-tests-button {min-height: 3.1rem; font-size: 1.05rem; font-weight: 600;}
.compact-note, .compact-note p {opacity: 0.8; font-size: .9rem;}
.input-panel {gap: .5rem !important;}
.input-panel h3 {margin: 0 0 .1rem !important; font-size: 1.05rem;}

/* Stage-aware progress card (test generation and grading). */
.progress-host:empty {display: none;}
.progress-card {margin: .6rem 0 .8rem; padding: 1rem 1.1rem; border-radius: 14px;
  border: 1px solid var(--border-color-accent, var(--color-accent)); background: var(--block-background-fill);}
.progress-card.is-done {border-color: #059669;}
.progress-card.is-failed {border-color: #dc2626;}
.progress-head {display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; flex-wrap: wrap;}
.progress-title {font-weight: 700;}
.progress-meta {font-size: .88rem; opacity: .8; font-variant-numeric: tabular-nums; white-space: nowrap;}
.progress-detail {margin: .25rem 0 .7rem; font-size: .92rem; opacity: .85;}
.progress-track {height: .5rem; border-radius: 99px; overflow: hidden; background: var(--border-color-primary);}
.progress-fill {height: 100%; border-radius: inherit; background: var(--color-accent); transition: width .6s ease;
  background-image: linear-gradient(90deg, transparent 0, rgba(255,255,255,.35) 50%, transparent 100%);
  background-size: 200% 100%; animation: progress-shimmer 1.6s linear infinite;}
.is-done .progress-fill {background-color: #059669; animation: none; background-image: none;}
.is-failed .progress-fill {background-color: #dc2626; animation: none; background-image: none;}
.progress-stages {display: flex; flex-wrap: wrap; gap: .4rem; margin: .75rem 0 0; padding: 0; list-style: none;}
.progress-stages li {padding: .18rem .6rem; border-radius: 999px; font-size: .8rem; border: 1px solid var(--border-color-primary); opacity: .65;}
.progress-stages li.done {opacity: .9; border-color: #059669;}
.progress-stages li.done::before {content: "✓ "; color: #059669; font-weight: 700;}
.progress-stages li.current {opacity: 1; font-weight: 600; border-color: var(--color-accent); color: var(--color-accent);}
/* Result tables: honour column_widths and break long unspaced tokens
   (hex dumps, long numeric inputs) instead of stretching the column. */
.results-table table {table-layout: fixed; width: 100%;}
.results-table td, .results-table th {overflow-wrap: anywhere; word-break: break-word; vertical-align: top;}
.io-table td:nth-child(4), .io-table td:nth-child(5), .io-table td:nth-child(6) {
  font-family: var(--font-mono); font-size: .85rem;}
@keyframes progress-shimmer {0% {background-position: 200% 0;} 100% {background-position: -200% 0;}}
@media (prefers-reduced-motion: reduce) {.progress-fill {animation: none; transition: none;}}

/* Tables: keep overflow inside the table frame instead of widening the page. */
.gradio-container .column, .gradio-container .tabitem {min-width: 0;}
.table-wrap {max-width: 100%; overflow-x: auto;}
.table-wrap td span {overflow-wrap: anywhere;}
.prose table {display: block; max-width: 100%; overflow-x: auto;}
.prose pre {max-width: 100%; overflow-x: auto;}
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
    """Replace the retired runtime_efficiency metric with the similarity metric."""
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

    # runtime_efficiency is replaced by the similarity metric with the same
    # weight and point cap. A legacy "originality" entry contributes its
    # comparison directory and is folded into the new metric.
    runtime = supplied_by_name["runtime_efficiency"]
    originality = supplied_by_name.get("originality", {})
    old_params = originality.get("params", {}) if isinstance(originality.get("params"), dict) else {}
    similarity = {
        "name": "similarity",
        "analyzer": "similarity_flag",
        "weight": runtime.get("weight", 0.1),
        "enabled": True,
        "params": {
            "corpus_dir": old_params.get("corpus_dir"),
            "low_threshold": old_params.get("low_threshold", settings.ANALYZER_SIMILARITY_LOW_THRESHOLD),
            "flag_threshold": old_params.get("flag_threshold", settings.ANALYZER_SIMILARITY_THRESHOLD),
            "shingle_size": old_params.get("shingle_size", settings.ANALYZER_SIMILARITY_SHINGLE_SIZE),
        },
        "max_points": runtime.get("max_points", 10),
        "grade_contribution": runtime.get("grade_contribution", True),
    }
    migrated: list[dict[str, Any]] = []
    for item in supplied:
        if not isinstance(item, dict) or item.get("name") == "originality":
            continue
        migrated.append(similarity if item.get("name") == "runtime_efficiency" else item)
    removed = sorted(set(supplied_by_name) - {metric["name"] for metric in migrated})
    added = sorted({metric["name"] for metric in migrated} - set(supplied_by_name))
    rules["metrics"] = migrated
    # This project uses the packaged default metric guidance. Assignment-side
    # rubric overrides from the old editor must not resurrect the old profile.
    rules.pop("rubric", None)
    rules.pop("rubrics", None)
    return {
        "applied": True,
        "profile": "runtime_efficiency_replaced_by_similarity",
        "removed_metrics": removed,
        "added_metrics": added,
    }


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
        verification = generation.get("expected_output_verification", {})
        if isinstance(verification, dict) and verification:
            summary.append(
                f"**Expected-output verification:** {verification.get('verified', 0)} generated pair(s) accepted"
                + (f" — {generation.get('oracle_caveat')}" if generation.get("oracle_caveat") else "")
            )
        shortfall = int(generation.get("generated_test_count_shortfall", 0) or 0)
        if shortfall:
            summary.append(f"**Generation note:** {shortfall} requested cases were not produced.")
        rejected_candidates = int(generation.get("rejected_generated_test_count", 0) or 0)
        duplicate_candidates = int(generation.get("duplicate_generated_test_count", 0) or 0)
        invalid_candidates = int(generation.get("invalid_generated_test_count", rejected_candidates - duplicate_candidates) or 0)
        if invalid_candidates or duplicate_candidates:
            summary.append(
                "**Generation note:** "
                + (f"{invalid_candidates} invalid LLM candidate(s) were discarded. " if invalid_candidates else "")
                + (f"{duplicate_candidates} repeated candidate(s) were skipped." if duplicate_candidates else "")
            )
        ambiguities = generation.get("unresolved_ambiguities", [])
        if isinstance(ambiguities, list) and ambiguities:
            summary.append(
                "**Expected-output note:** generated pairs were rechecked independently against the assignment specification. "
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
            status += " runtime_efficiency was replaced by the similarity metric (same weight and points)."
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
            generation_token_budget = min(
                max(4096, 2048 + requested_cases * 128),
                max(1024, int(settings.LLAMA_CONTEXT_SIZE) // 2),
            )

            def generation_request(messages, schema, stage_name):
                stage_token_budget = generation_token_budget
                if stage_name == "assignment_submission_understanding_single_pass":
                    stage_token_budget = min(
                        generation_token_budget,
                        max(512, int(settings.LLM_PROGRAM_UNDERSTANDING_MAX_TOKENS)),
                    )
                parsed, diagnostic = _request_json(
                    messages,
                    max_tokens=stage_token_budget,
                    enable_thinking=None,
                    response_schema=schema,
                    schema_name=stage_name,
                    fit_context=True,
                )
                if parsed is None:
                    detail = str(diagnostic or "No final assistant content was returned.")
                    detail = " ".join(detail.split())[:1_200]
                    logger.error(
                        "Structured LLM response was not valid JSON (stage=%s): %s",
                        stage_name,
                        detail,
                    )
                    raise RuntimeError(
                        f"{stage_name} did not return a valid JSON object. "
                        f"llama-server diagnostic: {detail}"
                    )
                return parsed

            # A sandbox lets reference programs run so expected outputs are
            # computed, not guessed. Without Docker, generation still works
            # and the result is flagged as not execution-checked.
            oracle_sandbox_manager = None
            oracle_sandbox = None
            try:
                from grading_agent.sandbox.docker_sandbox import DockerSandbox
                oracle_sandbox_manager = DockerSandbox(
                    memory_limit=f'{normalized_rules["memory_limit_mb"]}m',
                    cpu_limit=float(docker_cpu_limit),
                )
                oracle_sandbox = oracle_sandbox_manager.__enter__()
            except Exception as exc:
                logger.warning("Sandbox unavailable for reference execution during test generation: %s", exc)
                oracle_sandbox_manager = None
                oracle_sandbox = None
            try:
                result = prepare_assignment_test_suite(
                    normalized_rules,
                    sandbox=oracle_sandbox,
                    request_json=generation_request,
                    project_root=PROJECT_ROOT,
                    model_id=model_name,
                    generation_seed=settings.SEED,
                    temperature=float(temperature),
                    submission_source=source_code,
                    submission_filename=staged_source.name,
                )
            finally:
                if oracle_sandbox_manager is not None:
                    try:
                        oracle_sandbox_manager.__exit__(None, None, None)
                    except Exception:
                        logger.warning("Could not close the test-generation sandbox cleanly.")
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
        analysis_skipped_reason = str(result.get("submission_analysis_skipped_reason") or "").strip()
        if analysis_skipped_reason:
            submission_analysis = {}
        elif (
            not isinstance(submission_analysis, dict)
            or not isinstance(submission_analysis.get("implementation_summary"), str)
            or not submission_analysis.get("implementation_summary", "").strip()
        ):
            raise ValueError("The LLM did not complete its analysis of the submitted C source. No grading was started.")
        manifest = {
            "status": "completed",
            "source_sha256": _sha256_text(source_code),
            "rules_sha256": _sha256_text(raw_rules),
            "requested_generated_test_count": int(
                result.get("requested_generated_test_count", max_generated_cases)
            ),
            "configured_max_generated_cases": int(max_generated_cases),
            "stdin_unused": bool(result.get("stdin_unused", False)),
            "additional_test_focus": additional_test_focus.strip(),
            "assignment_fingerprint": result.get("assignment_fingerprint"),
            "suite_path": str(suite_path),
            "metric_testset_paths": {str(key): str(value) for key, value in metric_paths.items()},
            "instructor_test_count": int(result.get("instructor_test_count", 0)),
            "generated_test_count": int(result.get("generated_test_count", 0)),
            "total_test_count": int(result.get("total_test_count", len(cases))),
            "edge_test_count": int(result.get("edge_test_count", 0)),
            "rejected_generated_test_count": int(result.get("rejected_generated_test_count", 0)),
            "duplicate_generated_test_count": int(result.get("duplicate_generated_test_count", 0)),
            "invalid_generated_test_count": int(
                result.get("invalid_generated_test_count", result.get("rejected_generated_test_count", 0))
            ),
            "expected_output_verification": result.get("expected_output_verification"),
            "oracle_kind": result.get("oracle_kind"),
            "oracle_caveat": result.get("oracle_caveat"),
            "source_analysis": submission_analysis,
        }
        if manifest["generated_test_count"] == 0 and manifest["requested_generated_test_count"] > 0:
            raise ValueError(
                "The LLM did not produce any valid additional test cases. Instructor cases are preserved, "
                "but grading was not enabled for this generation request. Check the generation diagnostics "
                "and try again with a smaller generated-case target."
            )
        if manifest["generated_test_count"] == 0 and manifest["instructor_test_count"] == 0:
            raise ValueError("No valid instructor or generated test cases remained after validation.")

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
        generated_count = manifest["generated_test_count"]
        rejected_count = manifest["rejected_generated_test_count"]
        duplicate_count = int(manifest.get("duplicate_generated_test_count", 0))
        invalid_count = int(manifest.get("invalid_generated_test_count", rejected_count - duplicate_count))
        parts = [
            "## Test suite ready\n\n"
            f"**{generated_count} LLM-generated** + **{manifest['instructor_test_count']} instructor** = "
            f"**{manifest['total_test_count']} case(s) ready for grading.**"
            + (f" {invalid_count} invalid generated candidate(s) were discarded." if invalid_count else "")
            + (f" {duplicate_count} repeated candidate(s) were skipped." if duplicate_count else "")
            + "\n\n"
        ]
        if manifest["stdin_unused"]:
            parts.append(
                "**This assignment reads no stdin.** Generated cases are stdin-robustness edge cases: "
                "a correct program must ignore the supplied input and print the same fixed output.\n\n"
            )
        if analysis_skipped_reason:
            parts.append(
                "Source analysis was skipped because no test inputs needed to be designed. "
                "Memory and runtime defects are measured during grading by the configured analyzers.\n\n"
            )
        else:
            implementation_summary = submission_analysis.get("implementation_summary", "").strip()
            weak_point_text = "\n".join(
                f"- {item}" for item in submission_analysis.get("likely_weak_points", [])
                if isinstance(item, str) and item.strip()
            ) or "- None suggested."
            parts.append(
                "### Test-design notes from the submitted C code\n\n"
                f"{implementation_summary}\n\n"
                "**Areas the generator was asked to probe** (unverified LLM notes used only to choose test "
                "inputs; they are not grading findings):\n"
                f"{weak_point_text}\n\n"
            )
        if generated_count:
            caveat = str(manifest.get("oracle_caveat") or "").strip()
            unexecuted = "NOT confirmed by executing" in caveat
            parts.append(("**Warning:** " if unexecuted else "") + (caveat or "Generated expected outputs were checked.") + " ")
        if manifest["instructor_test_count"]:
            parts.append("Instructor cases are used exactly as provided. ")
        parts.append("Review the suite below, then click **Grade Submission**.")
        status = "".join(parts)
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
                "expected_output_verification": generated_suite_state.get("expected_output_verification"),
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


GENERATION_STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Preparing", ()),
    ("Analysing source", ("assignment_submission_understanding",)),
    ("Building reference", ("assignment_reference_program",)),
    ("Generating cases", ("assignment_test_generation",)),
    ("Verifying outputs", ("assignment_expected_output_verification",)),
)
GRADING_STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Tests and analyzers", ()),
    ("Understanding program", ("program_understanding",)),
    ("Source review", ("whole_program_rsi_findings",)),
    ("Validating findings", ("post_rsi_finding_validation",)),
    ("Scoring metrics", ("rubric_metric_scores",)),
)


def _elapsed_label(started: float) -> str:
    elapsed = max(0, int(time.monotonic() - started))
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _progress_card_html(
    *, title: str, detail: str, stages: Sequence[tuple[str, tuple[str, ...]]],
    index: int, started: float, state: str = "running",
) -> str:
    """Accessible, stage-aware progress card shared by generation and grading."""
    total = len(stages)
    if state == "done":
        percent = 100
    elif state == "failed":
        percent = max(8, int(100 * index / total))
    else:
        # The current stage is in progress: show it half filled.
        percent = max(4, min(96, int(100 * (index + 0.5) / total)))
    chips = []
    for position, (label, _prefixes) in enumerate(stages):
        if state == "done" or position < index:
            css = "done"
        elif position == index and state == "running":
            css = "current"
        else:
            css = ""
        chips.append(f'<li class="{css}">{html.escape(label)}</li>')
    card_class = {"done": " is-done", "failed": " is-failed"}.get(state, "")
    return (
        f'<section class="progress-card{card_class}" role="status" aria-live="polite">'
        '<div class="progress-head">'
        f'<span class="progress-title">{html.escape(title)}</span>'
        f'<span class="progress-meta">{percent}% · elapsed {_elapsed_label(started)}</span>'
        '</div>'
        f'<p class="progress-detail">{html.escape(detail)}</p>'
        f'<div class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" '
        f'aria-valuenow="{percent}" aria-label="{html.escape(title)}">'
        f'<div class="progress-fill" style="width:{percent}%"></div></div>'
        f'<ul class="progress-stages" aria-label="Stages">{"".join(chips)}</ul>'
        '</section>'
    )


def _run_with_stage_progress(
    fn: Callable[..., tuple],
    args: tuple,
    *,
    stages: Sequence[tuple[str, tuple[str, ...]]],
    running_title: str,
    done_title: str,
    failed_title: str,
    is_failure: Callable[[tuple], bool],
    pending_outputs: int,
    busy_updates: tuple = (),
    final_updates: tuple = (),
):
    """Run ``fn`` in a worker, streaming a progress card driven by real model stages.

    The worker runs in a copied context whose STAGE_LISTENER records the most
    advanced stage reached; the card reads it once per second. Yields
    (card, *outputs, *button_updates).
    """
    from grading_agent.llm.reviewer import STAGE_LISTENER

    started = time.monotonic()
    tracker = {"index": 0, "detail": stages[0][0]}

    def listener(stage_name: str) -> None:
        for position, (label, prefixes) in enumerate(stages):
            if any(stage_name.startswith(prefix) for prefix in prefixes):
                if position >= tracker["index"]:
                    tracker["index"] = position
                    tracker["detail"] = label
                return

    worker_context = contextvars.copy_context()
    worker_context.run(STAGE_LISTENER.set, listener)
    skipped = tuple(gr.skip() for _ in range(pending_outputs))

    def card(state: str = "running") -> str:
        if state == "done":
            return _progress_card_html(title=done_title, detail="All stages finished.", stages=stages,
                                       index=len(stages), started=started, state="done")
        if state == "failed":
            return _progress_card_html(title=failed_title, detail="Stopped during: " + tracker["detail"] + ". See the message below.",
                                       stages=stages, index=tracker["index"], started=started, state="failed")
        return _progress_card_html(title=running_title, detail="Current stage: " + tracker["detail"],
                                   stages=stages, index=tracker["index"], started=started)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="grader-ui") as executor:
        future = executor.submit(worker_context.run, fn, *args)
        yield (card(), *skipped, *busy_updates)
        while not future.done():
            time.sleep(1)
            if not future.done():
                yield (card(), *skipped, *busy_updates)
        result = future.result()
    failed = is_failure(result)
    yield (card("failed" if failed else "done"), *result, *final_updates)


def generate_test_cases_with_progress(*args: Any):
    """Generate the test suite while showing stage-aware progress."""
    yield from _run_with_stage_progress(
        generate_test_cases_for_submission, args,
        stages=GENERATION_STAGES,
        running_title="Generating test cases",
        done_title="Test suite ready",
        failed_title="Test-case generation needs attention",
        is_failure=lambda result: isinstance(result[0], str) and result[0].startswith("## Test-case generation failed"),
        # Outputs: status, suite state, table, grade button, generate button.
        # While running, the grade and generate buttons are both disabled; the
        # final result sets the grade button itself and re-enables generate.
        pending_outputs=3,
        busy_updates=(gr.update(interactive=False), gr.update(interactive=False)),
        final_updates=(gr.update(interactive=True),),
    )


def grade_uploaded_submission_with_progress(*args: Any):
    """Grade the submission while showing stage-aware progress."""
    yield from _run_with_stage_progress(
        grade_uploaded_submission, args,
        stages=GRADING_STAGES,
        running_title="Grading in progress",
        done_title="Grading complete",
        failed_title="Grading needs attention",
        is_failure=lambda result: isinstance(result[0], str) and result[0].startswith("## Could not complete grading"),
        pending_outputs=7,
        busy_updates=(gr.update(interactive=False), gr.update(interactive=False)),
        final_updates=(gr.update(interactive=True), gr.update(interactive=True)),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Code Grader") as demo:
        gr.Markdown(
            "# Code Grader\n"
            "Upload the C submission and the assignment rules, optionally generate extra test cases, then grade.",
            elem_id="app-header",
        )
        # Run settings live in a right-hand drawer: it slides in from the
        # Settings button and slides back out on "Done", keeping the main
        # grading page uncluttered. Values persist while the drawer is closed.
        with gr.Sidebar(label="Settings", open=False, position="right", width=440,
                        elem_id="settings-drawer") as settings_drawer:
            gr.Markdown("## Grading settings")
            with gr.Accordion("Execution", open=True):
                time_limit = gr.Slider(1, settings.DOCKER_MAX_TIMEOUT_S, value=settings.DOCKER_DEFAULT_TIMEOUT_S, step=1,
                                       label="Per-test timeout (seconds)")
                memory_limit = gr.Slider(128, 8192, value=512, step=128, label="Docker memory (MB)")
                docker_cpu = gr.Slider(0.25, 16, value=settings.DOCKER_CPU_LIMIT, step=0.25, label="Docker CPU (cores)")
                run_sanitizers = gr.Checkbox(value=settings.ANALYZER_RUN_SANITIZERS, label="Run ASan/UBSan")
            with gr.Accordion("LLM", open=True):
                model_name_ui = gr.Textbox(value=settings.MODEL, label="LLM model identifier")
                llm_temperature_ui = gr.Slider(0, 1, value=settings.TEMPERATURE, step=0.05, label="LLM temperature",
                                               info="0 gives repeatable grades.")
                llm_timeout_ui = gr.Slider(10, 900, value=settings.LLAMA_REQUEST_READ_TIMEOUT_S, step=10,
                                           label="LLM timeout (seconds)")
                rsi_iterations_ui = gr.Slider(1, settings.LLM_REVIEW_MAX_ITERATIONS, value=settings.LLM_REVIEW_MAX_ITERATIONS,
                                              step=1, label="RSI iterations", info="Stops early when a pass changes nothing.")
                max_findings_ui = gr.Slider(1, 50, value=settings.LLM_GLOBAL_MAX_FINDINGS, step=1, label="Maximum findings")
                thinking_ui = gr.Checkbox(value=True, label="Enable model thinking",
                                          info="Structured grading stages run without thinking so the JSON schema applies.")
            with gr.Accordion("Token budgets", open=False):
                understanding_tokens_ui = gr.Slider(64, 3072, value=settings.LLM_PROGRAM_UNDERSTANDING_MAX_TOKENS, step=64,
                                                    label="Understanding token budget")
                rsi_tokens_ui = gr.Slider(128, 4096, value=settings.LLM_GLOBAL_RSI_MAX_TOKENS, step=128, label="RSI token budget")
                scoring_tokens_ui = gr.Slider(64, 4096, value=settings.LLM_GLOBAL_SCORE_MAX_TOKENS, step=64, label="Scoring token budget")
            with gr.Accordion("Context limits", open=False):
                source_chars_ui = gr.Slider(1024, 30000, value=settings.LLM_GLOBAL_SOURCE_MAX_CHARS, step=1024,
                                            label="Maximum source characters")
                evidence_chars_ui = gr.Slider(512, 12000, value=settings.LLM_MAX_EVIDENCE_CHARS, step=512,
                                              label="Maximum evidence characters")
                context_compression_ui = gr.Checkbox(value=settings.LLM_CONTEXT_COMPRESSION_ENABLED,
                                                     label="Enable same-model compression",
                                                     info="Used only when a prompt would not fit the context.")
                compression_threshold_ui = gr.Slider(256, 20000, value=settings.LLM_CONTEXT_COMPRESSION_THRESHOLD_CHARS,
                                                     step=256, label="Compression threshold")
            settings_done_button = gr.Button("Done", variant="primary", elem_id="settings-done-button")

        with gr.Tab("Grade a C submission"):
            source_editor_hash = gr.State("")
            rules_editor_source_hash = gr.State("")
            generated_suite_state = gr.State({})

            # Submission and rules side by side; each editor is collapsed
            # behind its own bar so the page stays short until needed.
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300, elem_classes=["input-panel"]):
                    gr.Markdown("### Submission")
                    source_file = gr.File(label="C source file", file_types=[".c"], type="filepath", height=140)
                    source_status = gr.Markdown("Paste C code or upload a `.c` file.", elem_classes=["compact-note"])
                    with gr.Accordion("C source (view / edit)", open=False):
                        source_editor = gr.Code(
                            label="Submission source", language="c", interactive=True,
                            lines=22, max_lines=40, value="", elem_id="submission-source",
                        )
                with gr.Column(scale=1, min_width=300, elem_classes=["input-panel"]):
                    gr.Markdown("### Assignment rules")
                    rules_file = gr.File(label="Instructor rules JSON", file_types=[".json"], type="filepath", height=140)
                    rules_status = gr.Markdown("Upload rules JSON to load an editable copy.", elem_classes=["compact-note"])
                    with gr.Accordion("rules.json (view / edit)", open=False):
                        rules_editor = gr.Code(
                            label="Rules JSON", language="json", interactive=True,
                            lines=22, max_lines=40, value="",
                        )
                        gr.Markdown(
                            "Rules and test-case paths resolve from the repository root.",
                            elem_classes=["compact-note"],
                        )

            # Step: optional test generation.
            with gr.Row(equal_height=False):
                max_generated_cases = gr.Slider(
                    0, 60, value=0, step=1, label="Generated test case target",
                    info="Up to this many additional LLM-generated cases. Invalid candidates are discarded; instructor cases are always kept.",
                    scale=1, min_width=300,
                )
                additional_test_focus = gr.Textbox(
                    label="Test variation focus (optional)",
                    placeholder="For example: duplicate values, zero values, and boundary sizes",
                    lines=2, max_lines=3, max_length=500, scale=1, min_width=300,
                )
            with gr.Row():
                settings_open_button = gr.Button("⚙ Settings", variant="secondary", elem_id="settings-open-button",
                                                 scale=0, min_width=140)
                generate_button = gr.Button("Generate Test Cases", variant="secondary", elem_id="generate-tests-button", scale=1)
                grade_button = gr.Button("Grade Submission", variant="primary", elem_id="grade-button", interactive=True, scale=1)
            generation_progress = gr.HTML(value="", elem_classes=["progress-host"])
            generation_status = gr.Markdown(
                "No generated test cases requested. Grading can run with instructor-configured tests."
            )
            grading_progress = gr.HTML(value="", elem_id="grading-progress", elem_classes=["progress-host"])

            with gr.Tabs():
                with gr.Tab("Overview"):
                    grade_summary = gr.Markdown("Results appear here after grading.")
                    metrics_table = gr.Dataframe(
                        headers=["Metric", "Points awarded", "Status", "Assessment"],
                        datatype=["str", "str", "str", "str"],
                        value=[], interactive=False, wrap=True, show_search="filter",
                        label="Metric results", max_chars=360,
                        column_widths=["20%", "12%", "12%", "56%"], elem_classes=["results-table"],
                    )
                with gr.Tab("Test cases"):
                    test_cases_table = gr.Dataframe(
                        headers=["Case", "Origin", "Tags", "Input", "Expected", "Actual", "Result"],
                        datatype=["str", "str", "str", "str", "str", "str", "str"],
                        value=[], interactive=False, wrap=True, show_search="filter",
                        label="Generated and graded test cases", max_chars=220,
                        column_widths=["7%", "10%", "13%", "22%", "20%", "20%", "8%"],
                        elem_classes=["results-table", "io-table"],
                    )
                with gr.Tab("Findings"):
                    findings_table = gr.Dataframe(
                        headers=["Severity", "Metric", "Lines", "Finding", "Evidence"],
                        datatype=["str", "str", "str", "str", "str"],
                        value=[], interactive=False, wrap=True, show_search="filter",
                        label="Validated source findings", max_chars=420,
                        column_widths=["9%", "13%", "7%", "24%", "47%"], elem_classes=["results-table"],
                    )
                with gr.Tab("Submitted source"):
                    graded_source = gr.Code(label="Exact C source graded", language="c", interactive=False,
                                            lines=24, max_lines=48)
                with gr.Tab("Report files"):
                    with gr.Row():
                        report_json = gr.File(label="JSON report")
                        report_text = gr.File(label="Readable report")

            invalidation_outputs = [generation_status, generated_suite_state, test_cases_table, grade_button]
            source_file.change(
                load_source_editor_from_upload, inputs=[source_file],
                outputs=[source_editor, source_editor_hash, source_status], show_progress="hidden",
            ).then(invalidate_generated_test_suite, inputs=[max_generated_cases], outputs=invalidation_outputs, show_progress="hidden")
            source_editor.change(invalidate_generated_test_suite, inputs=[max_generated_cases], outputs=invalidation_outputs, show_progress="hidden")
            rules_file.change(
                load_rules_editor_from_upload, inputs=[rules_file],
                outputs=[rules_editor, time_limit, memory_limit, rules_status, rules_editor_source_hash], show_progress="hidden",
            ).then(invalidate_generated_test_suite, inputs=[max_generated_cases], outputs=invalidation_outputs, show_progress="hidden")
            rules_editor.change(invalidate_generated_test_suite, inputs=[max_generated_cases], outputs=invalidation_outputs, show_progress="hidden")
            max_generated_cases.change(invalidate_generated_test_suite, inputs=[max_generated_cases], outputs=invalidation_outputs, show_progress="hidden")
            additional_test_focus.change(invalidate_generated_test_suite, inputs=[max_generated_cases], outputs=invalidation_outputs, show_progress="hidden")

            generate_button.click(
                generate_test_cases_with_progress,
                inputs=[
                    source_file, source_editor, source_editor_hash,
                    rules_file, rules_editor, rules_editor_source_hash,
                    time_limit, memory_limit, docker_cpu,
                    llm_temperature_ui, model_name_ui, llm_timeout_ui,
                    max_generated_cases, additional_test_focus, thinking_ui,
                ],
                outputs=[generation_progress, generation_status, generated_suite_state, test_cases_table, grade_button, generate_button],
                show_progress="hidden",
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
                outputs=[grading_progress, grade_summary, metrics_table, test_cases_table, findings_table,
                         graded_source, report_json, report_text, grade_button, generate_button],
                show_progress="hidden",
            )

        settings_open_button.click(lambda: gr.Sidebar(open=True), outputs=settings_drawer,
                                   queue=False, show_progress="hidden")
        settings_done_button.click(lambda: gr.Sidebar(open=False), outputs=settings_drawer,
                                   queue=False, show_progress="hidden")

        formai_controls = build_formai_tab()
        # Fill the allocation preview for the default slider value on page load.
        demo.load(preview_allocation, inputs=[formai_controls[0], formai_controls[1], formai_controls[2],
                                              formai_controls[3], formai_controls[4]],
                  outputs=[formai_controls[5], formai_controls[6]], show_progress="hidden")

    return demo


def app_theme() -> gr.themes.Base:
    """The Ocean theme; fonts left at its defaults."""
    return gr.themes.Ocean()


def main() -> None:
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT_APP)
    app = build_app()
    app.queue().launch(
        server_name=settings.UI_HOST,
        server_port=settings.UI_PORT,
        show_error=settings.UI_SHOW_ERROR,
        theme=app_theme(),
        css=OCEAN_UI_CSS,
    )


if __name__ == "__main__":
    main()
