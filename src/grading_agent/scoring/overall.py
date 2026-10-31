"""Simple deterministic metric selection and weighted grade calculation."""
from __future__ import annotations

import math
from typing import Any


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
    flowchart. If an LLM score is unavailable, the deterministic analyzer's
    score is used. If neither exists, the metric receives zero.
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

        analyzer_score = _valid_score(result.get("score"))
        llm_item = llm_by_name.get(name, {})
        llm_score = _valid_score(llm_item.get("score")) if isinstance(llm_item, dict) else None

        if llm_score is not None:
            final_score = llm_score
            basis = "LLM RSI score based on rubric, source code, and analyzer evidence"
        elif analyzer_score is not None:
            final_score = analyzer_score
            basis = "deterministic analyzer score (LLM RSI score unavailable)"
        else:
            final_score = 0.0
            basis = "no analyzer or LLM score available; zero used"

        result["analyzer_score"] = analyzer_score
        result["llm_rsi_score"] = llm_score
        result["score"] = round(final_score, 2)
        result["score_scale"] = "0-100"
        result["scoring_basis"] = basis
        if isinstance(llm_item, dict) and llm_item:
            result["llm_rsi"] = {
                "rationale": llm_item.get("rationale", ""),
                "basis": llm_item.get("basis", "limited_evidence"),
                "evidence_ids": llm_item.get("evidence_ids", []),
            }
        result["status"] = "completed" if analyzer_score is not None or llm_score is not None else "completed_with_fallback"


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
        if score is None:
            score = 0.0
            warnings.append(f"{name}: no score was available; zero was used.")
        elif isinstance(result, dict) and result.get("scoring_basis", "").startswith("no analyzer or LLM score"):
            warnings.append(f"{name}: no analyzer or LLM score was available; zero was used.")

        weight = float(metric["weight"])
        contribution = weight * score
        weighted_total += contribution
        contributions.append({
            "metric": name,
            "analyzer": metric.get("analyzer"),
            "weight": weight,
            "score": round(score, 2),
            "contribution": round(contribution, 2),
            "scoring_basis": result.get("scoring_basis", "no analyzer or LLM score") if isinstance(result, dict) else "no analyzer or LLM score",
        })

    return {
        "status": "completed",
        "score": round(weighted_total, 2),
        "scale": "0-100",
        "weights_sum": round(total_weight, 6),
        "contributions": contributions,
        "warnings": warnings,
        "formula": "sum(metric_weight * final_metric_score)",
        "metric_fusion": "LLM RSI score per metric; deterministic analyzer score is the fallback; zero only when both are unavailable.",
    }
