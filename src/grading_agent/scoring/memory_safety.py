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
    if audit.get("status") != "completed":
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
        "status": "completed",
        "score": score,
        "unique_findings": len(unique_findings),
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
    llm = _llm_component_score(llm_audit, severity_penalties)

    if valgrind["score"] is None or llm["score"] is None:
        return {
            "status": "incomplete",
            "score": None,
            "scale": "0-100",
            "weights": {
                "valgrind": valgrind_weight,
                "llm_source_audit": llm_audit_weight,
            },
            "valgrind": valgrind,
            "llm_source_audit": llm,
            "reason": "Both Valgrind and LLM source audit must complete before a composite score is issued.",
        }

    score = round(
        valgrind_weight * valgrind["score"]
        + llm_audit_weight * llm["score"],
        2,
    )

    return {
        "status": "completed",
        "score": score,
        "scale": "0-100",
        "weights": {
            "valgrind": valgrind_weight,
            "llm_source_audit": llm_audit_weight,
        },
        "valgrind": valgrind,
        "llm_source_audit": llm,
        "formula": "valgrind_weight * valgrind_score + llm_audit_weight * llm_score",
    }
