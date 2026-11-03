"""Deterministic metric selection and weighted grade calculation.

Analyzer outputs are the sole numerical authority. The LLM may review every
metric and provide an advisory score/rationale, but its output never changes
the score used in the weighted grade.
"""
from __future__ import annotations

import math
from typing import Any

from grading_agent.reporting.rationale import compact_rationale
from grading_agent.scoring.score_bands import score_from_band


def _valid_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= 100.0 else None


def apply_metric_scores(
    metrics: list[dict[str, Any]],
    report: dict[str, Any],
    llm_evaluation: dict[str, Any] | None,
) -> None:
    """Attach deterministic analyzer scores and keep LLM scores advisory.

    Missing, unavailable, and inconclusive analyzer measurements remain
    missing. No neutral/default score is invented.
    """
    llm_by_name = {
        item.get("name"): item
        for item in (llm_evaluation or {}).get("metrics", [])
        if isinstance(item, dict) and item.get("name")
    }

    for metric in metrics:
        if not metric.get("enabled", True):
            continue
        name = metric["name"]
        result = report.setdefault(name, {})
        if not isinstance(result, dict):
            result = {}
            report[name] = result

        analyzer_status = str(result.get("status", "")).casefold()
        analyzer_score = _valid_score(result.get("score"))
        if analyzer_status != "completed":
            analyzer_score = None

        llm_item = llm_by_name.get(name, {})
        is_actual_llm_review = isinstance(llm_item, dict) and llm_item.get("origin", "llm") == "llm"
        # Only a documented band can be retained as an LLM advisory value;
        # arbitrary numeric values are deliberately ignored even if a caller
        # bypasses the constrained response schema.
        llm_proposal = (
            score_from_band(llm_item.get("score_band"))
            if is_actual_llm_review else None
        )

        # The deterministic analyzer is the only scoring authority.
        result["analyzer_score"] = analyzer_score
        result["llm_review_score"] = llm_proposal
        result["score"] = round(analyzer_score, 2) if analyzer_score is not None else None
        result["score_scale"] = "0-100"
        result["score_origin"] = (
            "deterministic_analyzer" if analyzer_score is not None else "unmeasured"
        )

        analyzer_explanation = (
            result.get("summary") or result.get("rationale")
            or result.get("message") or result.get("reason")
        )
        llm_explanation = (
            llm_item.get("rationale", "") if isinstance(llm_item, dict) else ""
        )
        explanation = analyzer_explanation or llm_explanation
        if analyzer_score is None:
            explanation = (
                result.get("reason")
                or result.get("error")
                or f"{name.replace('_', ' ')} was not measured; no grade was assigned."
            )
        result["score_rationale"] = compact_rationale(
            explanation, max_words=18, max_chars=120
        )

        # Preserve the LLM's metric-by-metric review for transparency, but
        # never feed its proposed numeric score into grade calculation.
        if is_actual_llm_review:
            result["llm_review"] = {
                "proposed_score": llm_proposal,
                "score_band": llm_item.get("score_band"),
                "rationale": llm_item.get("rationale", ""),
                "evidence_ids": llm_item.get("evidence_ids", []),
                "iterations_requested": llm_item.get("iterations_requested", 1),
                "iterations_completed": llm_item.get("iterations_completed", 1),
                "iteration_status": llm_item.get("iteration_status", "completed"),
            }
        else:
            result.pop("llm_review", None)

        result["scoring_status"] = (
            "measured" if analyzer_score is not None else "inconclusive"
        )


def calculate_overall_score(
    metrics: list[dict[str, Any]],
    metric_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Calculate a weighted grade only when every enabled metric is measured."""
    enabled = [item for item in metrics if item.get("enabled", True)]
    if not enabled:
        return {
            "status": "failed_configuration",
            "score": None,
            "scale": "0-100",
            "warnings": ["No metrics are enabled."],
            "formula": "sum(metric_weight * deterministic_analyzer_score)",
        }

    total_weight = sum(float(item["weight"]) for item in enabled)
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Enabled metric weights must sum to 1.0, got {total_weight:.6f}."
        )

    missing: list[dict[str, str]] = []
    for metric in enabled:
        name = metric["name"]
        result = metric_results.get(name)
        score = _valid_score(result.get("score")) if isinstance(result, dict) else None
        status = str(result.get("status", "missing")) if isinstance(result, dict) else "missing"
        if score is None or status != "completed":
            reason = "No result was produced."
            if isinstance(result, dict):
                reason = str(
                    result.get("reason") or result.get("error")
                    or result.get("analysis_error") or status
                )
            missing.append({
                "metric": name,
                "analyzer": str(metric.get("analyzer", "unknown")),
                "status": status,
                "reason": reason[:500],
            })

    if missing:
        return {
            "status": "failed_incomplete",
            "score": None,
            "scale": "0-100",
            "weights_sum": round(total_weight, 6),
            "contributions": [],
            "missing_metrics": missing,
            "warnings": [
                "No final grade was calculated because one or more enabled metrics "
                "could not be measured. Resolve the listed causes and rerun."
            ],
            "formula": "sum(metric_weight * deterministic_analyzer_score)",
            "metric_fusion": (
                "Only completed deterministic analyzer scores are authoritative. "
                "LLM scores are advisory and cannot fill missing measurements."
            ),
        }

    contributions: list[dict[str, Any]] = []
    weighted_total = 0.0
    for metric in enabled:
        name = metric["name"]
        result = metric_results[name]
        score = _valid_score(result.get("score"))
        if score is None:  # Defensive: missing scores were rejected above.
            raise AssertionError(f"Validated metric '{name}' lost its score.")
        weight = float(metric["weight"])
        contribution = weight * score
        weighted_total += contribution
        contributions.append({
            "metric": name,
            "analyzer": metric.get("analyzer"),
            "weight": weight,
            "score": round(score, 2),
            "contribution": round(contribution, 2),
            "score_origin": "deterministic_analyzer",
            "llm_review_score": _valid_score(result.get("llm_review_score")),
        })

    return {
        "status": "completed",
        "score": round(weighted_total, 2),
        "scale": "0-100",
        "weights_sum": round(total_weight, 6),
        "contributions": contributions,
        "warnings": [],
        "formula": "sum(metric_weight * deterministic_analyzer_score)",
        "metric_fusion": (
            "Deterministic analyzer scores determine the final grade; "
            "LLM per-metric scores and explanations are advisory only."
        ),
    }
