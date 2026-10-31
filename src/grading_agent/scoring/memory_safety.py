"""Deterministically combine Valgrind and LLM source-audit evidence."""

from __future__ import annotations

import re
from typing import Any


VALID_SEVERITIES = {"critical", "high", "medium", "low"}


def _valgrind_component_score(report: dict[str, Any]) -> dict[str, Any]:
    """Convert completed Valgrind test runs into a 0--100 component score.

    A missing log, timeout, failed compilation, truncated log, or absent
    ERROR SUMMARY makes this component inconclusive rather than safe.
    """
    results = report.get("results")
    total = report.get("tests_total")

    if (
        not report.get("compiled")
        or not isinstance(results, list)
        or not isinstance(total, int)
        or total <= 0
        or len(results) != total
        or report.get("analysis_error")
    ):
        return {
            "status": "inconclusive",
            "score": None,
            "tests_clean": 0,
            "tests_total": total,
            "reason": "Valgrind did not produce complete, valid evidence for every test.",
        }

    for result in results:
        summary = result.get("parsed_summary") or {}
        if (
            result.get("analysis_error")
            or result.get("timed_out")
            or result.get("log_truncated")
            or result.get("log_read_error")
            or summary.get("error_count") is None
        ):
            return {
                "status": "inconclusive",
                "score": None,
                "tests_clean": report.get("tests_clean", 0),
                "tests_total": total,
                "reason": "At least one Valgrind run was incomplete or inconclusive.",
            }

    clean = sum(1 for item in results if item.get("memory_safe") is True)
    score = round((clean / total) * 100.0, 2)

    return {
        "status": "completed",
        "score": score,
        "tests_clean": clean,
        "tests_total": total,
        "basis": "fraction of test executions with no reported Memcheck errors or leak findings",
    }


def _finding_identity(finding: dict[str, Any]) -> tuple[str, str, str]:
    """Build a stable key to avoid penalizing the same finding repeatedly."""
    category = str(finding.get("category", "unspecified")).strip().casefold()
    line = str(finding.get("line", "unknown")).strip()
    function = str(finding.get("function", "unknown")).strip().casefold()
    description = re.sub(
        r"\s+", " ", str(finding.get("description", "")).strip().casefold()
    )
    # Prefer location+category, but use normalized description when no line exists.
    return category, line, function if line != "unknown" else description


def _llm_component_score(
    audit: dict[str, Any],
    severity_penalties: dict[str, float],
) -> dict[str, Any]:
    """Map structured LLM findings to a deterministic rubric score.

    The LLM identifies findings and severity. Python applies the instructor's
    configured penalties; the model never returns the numeric component score.
    """
    if audit.get("status") not in {"completed", "completed_with_warnings"}:
        return {
            "status": "inconclusive",
            "score": None,
            "reason": audit.get("error", "LLM source audit did not complete."),
        }

    payload = audit.get("audit")
    if not isinstance(payload, dict) or not isinstance(payload.get("findings"), list):
        return {
            "status": "inconclusive",
            "score": None,
            "reason": "LLM source audit did not return a valid findings array.",
        }

    unique_findings: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, finding in enumerate(payload["findings"], start=1):
        if not isinstance(finding, dict):
            return {
                "status": "inconclusive",
                "score": None,
                "reason": f"LLM finding {index} is not a JSON object.",
            }

        severity = str(finding.get("severity", "")).casefold().strip()
        if severity not in VALID_SEVERITIES:
            return {
                "status": "inconclusive",
                "score": None,
                "reason": f"LLM finding {index} has unsupported severity '{severity}'.",
            }

        if not finding.get("description"):
            return {
                "status": "inconclusive",
                "score": None,
                "reason": f"LLM finding {index} is missing a description.",
            }

        key = _finding_identity(finding)
        # If the same location/category is repeated, retain the highest severity.
        previous = unique_findings.get(key)
        severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        if (
            previous is None
            or severity_order[severity] > severity_order[previous["severity"]]
        ):
            unique_findings[key] = {**finding, "severity": severity}

    penalties_applied = [
        {
            "category": finding.get("category", "unspecified"),
            "line": finding.get("line"),
            "severity": finding["severity"],
            "penalty": float(severity_penalties[finding["severity"]]),
        }
        for finding in unique_findings.values()
    ]
    total_penalty = min(
        100.0, sum(item["penalty"] for item in penalties_applied)
    )
    score = round(max(0.0, 100.0 - total_penalty), 2)

    return {
        "status": "completed_with_warnings" if audit.get("status") == "completed_with_warnings" or payload.get("rejected_findings") else "completed",
        "score": score,
        "unique_findings": len(unique_findings),
        "rejected_unscored_findings": len(payload.get("rejected_findings", [])),
        "penalties_applied": penalties_applied,
        "total_penalty": round(total_penalty, 2),
        "basis": "100 minus configured severity penalties for unique source-level findings",
    }


def calculate_memory_safety_score(
    valgrind_report: dict[str, Any],
    llm_audit: dict[str, Any],
    *,
    valgrind_weight: float,
    llm_audit_weight: float,
    severity_penalties: dict[str, float],
) -> dict[str, Any]:
    """Combine both mandatory checks using instructor-provided weights.

    If either audit is unavailable or inconclusive, the composite score is
    withheld. It is never silently renormalized using only the available side.
    """
    if not (0 < valgrind_weight < 1 and 0 < llm_audit_weight < 1):
        raise ValueError("Both memory-safety component weights must be between 0 and 1.")
    if abs((valgrind_weight + llm_audit_weight) - 1.0) > 1e-6:
        raise ValueError("Memory-safety component weights must sum to 1.0.")

    valgrind = _valgrind_component_score(valgrind_report)
    runtime_tools = valgrind_report.get("runtime_tools")
    if isinstance(runtime_tools, dict):
        runtime_component = {
            "status": runtime_tools.get("status", "inconclusive"),
            "score": runtime_tools.get("score"),
            "valgrind_score": runtime_tools.get("valgrind_score"),
            "asan_ubsan_score": runtime_tools.get("asan_ubsan_score"),
            "basis": runtime_tools.get("scoring_note"),
        }
    else:
        runtime_component = valgrind
    llm = _llm_component_score(llm_audit, severity_penalties)

    if runtime_component["score"] is None or llm["score"] is None:
        missing_reasons = []
        if runtime_component["score"] is None:
            runtime_reason = runtime_component.get("reason") or valgrind.get("reason")
            missing_reasons.append(
                "Runtime memory checks are incomplete"
                + (f": {runtime_reason}" if runtime_reason else ".")
            )
        if llm["score"] is None:
            audit_reason = llm.get("reason") or llm_audit.get("error")
            missing_reasons.append(
                "LLM source audit is incomplete"
                + (f": {audit_reason}" if audit_reason else ".")
            )
        return {
            "status": "incomplete",
            "score": None,
            "scale": "0-100",
            "weights": {
                "valgrind": valgrind_weight,
                "llm_source_audit": llm_audit_weight,
            },
            "runtime_tools": runtime_component,
            "valgrind": valgrind,
            "llm_source_audit": llm,
            "reason": "; ".join(missing_reasons),
        }

    score = round(
        valgrind_weight * runtime_component["score"]
        + llm_audit_weight * llm["score"],
        2,
    )

    warning_status = (
        runtime_component.get("status") == "completed_with_warnings"
        or llm.get("status") == "completed_with_warnings"
    )
    return {
        "status": "completed_with_warnings" if warning_status else "completed",
        "score": score,
        "scale": "0-100",
        "weights": {
            "valgrind": valgrind_weight,
            "llm_source_audit": llm_audit_weight,
        },
        "runtime_tools": runtime_component,
        "valgrind": valgrind,
        "llm_source_audit": llm,
        "formula": "valgrind_weight * runtime_tools_score + llm_audit_weight * llm_score",
    }


def finalize_memory_safety_result(
    memory_result: dict[str, Any],
    llm_audit: dict[str, Any],
    *,
    valgrind_weight: float,
    llm_audit_weight: float,
    severity_penalties: dict[str, float],
) -> dict[str, Any]:
    """Attach the composite result and align the metric status with its score.

    Runtime tools can complete while the required source audit remains
    inconclusive. The top-level metric must then be incomplete, while nested
    runtime-tools evidence remains available as completed.
    """
    composite = calculate_memory_safety_score(
        memory_result,
        llm_audit,
        valgrind_weight=valgrind_weight,
        llm_audit_weight=llm_audit_weight,
        severity_penalties=severity_penalties,
    )
    memory_result["llm_source_audit"] = llm_audit
    memory_result["composite_score"] = composite
    memory_result["score"] = composite.get("score")
    memory_result["score_scale"] = "0-100"

    if composite.get("score") is None:
        memory_result["status"] = "incomplete"
        memory_result["reason"] = composite.get(
            "reason", "Memory-safety composite could not be calculated."
        )
    elif composite.get("status") == "completed_with_warnings":
        memory_result["status"] = "completed_with_warnings"
        memory_result["reason"] = composite.get("reason", "Composite score includes reviewed, unscored LLM audit claims.")
    else:
        memory_result["status"] = "completed"
        memory_result.pop("reason", None)
    return memory_result
