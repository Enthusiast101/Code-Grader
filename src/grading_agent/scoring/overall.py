"""Simple deterministic metric selection and weighted grade calculation."""
from __future__ import annotations

import math
from typing import Any

from grading_agent.reporting.rationale import compact_rationale


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
    """Choose the RSI metric score; fall back to the analyzer score.

    The analyzers collect objective evidence. The LLM evaluates that evidence
    against the rubric and supplies the metric score, as shown in the project
    flowchart. If an LLM score is unavailable, a completed analyzer score is
    used. If neither source measured the metric, a neutral provisional score
    is used and clearly identified rather than inventing a zero.
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
        if analyzer_status in {"inconclusive", "not_configured", "unavailable", "skipped", "error", "failed"}:
            analyzer_score = None
        llm_item = llm_by_name.get(name, {})
        score_origin = str(llm_item.get("origin", llm_item.get("score_origin", "llm"))) if isinstance(llm_item, dict) else "llm"
        candidate_llm_score = _valid_score(llm_item.get("score")) if isinstance(llm_item, dict) else None
        llm_score = candidate_llm_score if score_origin == "llm" else None
        fallback_score = candidate_llm_score if score_origin != "llm" else None

        if llm_score is not None:
            final_score = llm_score
            final_origin = "llm"
        elif analyzer_score is not None:
            final_score = analyzer_score
            final_origin = "analyzer_fallback"
        elif fallback_score is not None:
            final_score = fallback_score
            final_origin = score_origin if score_origin != "llm" else "evidence_fallback"
        else:
            # Missing evidence is not proof of a failed submission. Keep a
            # clearly tagged neutral provisional value rather than inventing 0.
            final_score = 70.0
            final_origin = "neutral_provisional"

        result["analyzer_score"] = analyzer_score
        result["llm_rsi_score"] = llm_score
        result["score"] = round(final_score, 2)
        result["score_scale"] = "0-100"
        result["score_origin"] = final_origin
        if llm_score is not None and isinstance(llm_item, dict):
            explanation = llm_item.get("rationale", "")
        elif final_origin == "analyzer_fallback":
            explanation = (
                result.get("summary") or result.get("rationale") or result.get("message")
                or f"LLM score unavailable; the completed {name.replace('_', ' ')} analyzer supplied the fallback score."
            )
        else:
            explanation = (
                llm_item.get("rationale", "") if isinstance(llm_item, dict) else ""
            ) or f"Insufficient measured evidence for {name.replace('_', ' ')}; a provisional score is shown."
        result["score_rationale"] = compact_rationale(explanation, max_words=18, max_chars=120)
        # Remove the old verbose score-basis field from generated artifacts.
        result.pop("scoring_basis", None)
        if isinstance(llm_item, dict) and llm_item and llm_score is not None:
            result["llm_rsi"] = {
                "rationale": llm_item.get("rationale", ""),
                "evidence_ids": llm_item.get("evidence_ids", []),
                "iterations_requested": llm_item.get("iterations_requested", 1),
                "iterations_completed": llm_item.get("iterations_completed", 1),
                "iteration_status": llm_item.get("iteration_status", "completed"),
                "iteration_history": llm_item.get("iteration_history", []),
                "score_origin": score_origin,
            }
        elif llm_score is None:
            result.pop("llm_rsi", None)
        result["status"] = "completed" if llm_score is not None else "completed_with_fallback"


def calculate_overall_score(
    metrics: list[dict[str, Any]],
    metric_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Calculate Wt = sum(Wi * Si) for every enabled metric."""
    enabled = [item for item in metrics if item.get("enabled", True)]
    if not enabled:
        return {
            "status": "completed",
            "score": 0.0,
            "scale": "0-100",
            "warnings": ["No metrics are enabled."],
            "formula": "sum(metric_weight * final_metric_score)",
        }

    total_weight = sum(float(item["weight"]) for item in enabled)
    if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
        raise ValueError(f"Enabled metric weights must sum to 1.0, got {total_weight:.6f}.")

    contributions: list[dict[str, Any]] = []
    warnings: list[str] = []
    weighted_total = 0.0

    for metric in enabled:
        name = metric["name"]
        result = metric_results.get(name)
        score = _valid_score(result.get("score")) if isinstance(result, dict) else None
        score_origin = str(result.get("score_origin", "unknown")) if isinstance(result, dict) else "unknown"
        if score is None:
            score = 70.0
            score_origin = "neutral_provisional"
            warnings.append(f"{name}: no valid score was available; a neutral provisional score was used.")
        elif score_origin == "neutral_provisional":
            warnings.append(f"{name}: evidence was insufficient for a measured score; a neutral provisional score was used.")

        weight = float(metric["weight"])
        contribution = weight * score
        weighted_total += contribution
        contributions.append({
            "metric": name,
            "analyzer": metric.get("analyzer"),
            "weight": weight,
            "score": round(score, 2),
            "contribution": round(contribution, 2),
            "score_origin": score_origin,
        })

    return {
        "status": "completed",
        "score": round(weighted_total, 2),
        "scale": "0-100",
        "weights_sum": round(total_weight, 6),
        "contributions": contributions,
        "warnings": warnings,
        "formula": "sum(metric_weight * final_metric_score)",
        "metric_fusion": "LLM score is authoritative; completed deterministic analyzer scores are fallbacks only; unmeasured metrics receive a labeled neutral provisional score.",
    }
