"""Deterministic weighted aggregation for all enabled grading metrics."""
from __future__ import annotations

import math
from typing import Any


def calculate_overall_score(
    metrics: list[dict[str, Any]],
    metric_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Calculate a 0-100 weighted grade without renormalizing missing metrics."""
    enabled = [item for item in metrics if item.get("enabled", True)]
    if not enabled:
        return {"status": "incomplete", "score": None, "reason": "No enabled metrics."}

    total_weight = sum(float(item["weight"]) for item in enabled)
    if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
        raise ValueError(f"Enabled metric weights must sum to 1.0, got {total_weight:.6f}.")

    contributions = []
    incomplete = []
    weighted_total = 0.0

    for metric in enabled:
        name = metric["name"]
        result = metric_results.get(name)
        if not isinstance(result, dict):
            incomplete.append({"metric": name, "reason": "No analyzer result was produced."})
            continue

        score = result.get("score")
        if result.get("status") in {"inconclusive", "unavailable", "incomplete"} or score is None:
            incomplete.append({
                "metric": name,
                "reason": result.get("reason") or result.get("error") or f"Analyzer status: {result.get('status', 'unknown')}.",
            })
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
            incomplete.append({"metric": name, "reason": f"Invalid score value: {score!r}."})
            continue

        contribution = float(metric["weight"]) * float(score)
        weighted_total += contribution
        contributions.append({
            "metric": name,
            "analyzer": metric["analyzer"],
            "weight": float(metric["weight"]),
            "score": round(float(score), 2),
            "contribution": round(contribution, 2),
        })

    if incomplete:
        return {
            "status": "incomplete",
            "score": None,
            "scale": "0-100",
            "weights_sum": round(total_weight, 6),
            "contributions_available": contributions,
            "incomplete_metrics": incomplete,
            "policy": "No grade is issued when any enabled metric is missing or inconclusive; weights are not renormalized.",
        }

    return {
        "status": "completed",
        "score": round(weighted_total, 2),
        "scale": "0-100",
        "weights_sum": round(total_weight, 6),
        "contributions": contributions,
        "formula": "sum(metric_weight * metric_score)",
    }
