"""Evidence-first metric selection and weighted grade calculation.

Executed correctness/logical tests determine their pass-rate scores. Validated
LLM rubric scores assess other metrics and criteria without executed tests;
completed analyzers are the fallback when an LLM score is unavailable.
"""
from __future__ import annotations

import math
from typing import Any

from grading_agent.reporting.rationale import compact_rationale
from grading_agent.scoring.score_bands import band_for_score


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
    """Use test outcomes for correctness/logical, then LLM and analyzer evidence."""
    llm_metrics = (llm_evaluation or {}).get("metrics", [])
    llm_by_name = {
        item.get("name"): item for item in llm_metrics
        if isinstance(item, dict) and item.get("name")
    } if isinstance(llm_metrics, list) else {}

    for metric in metrics:
        if not metric.get("enabled", True):
            continue
        name = metric["name"]
        result = report.setdefault(name, {})
        if not isinstance(result, dict):
            result = {}
            report[name] = result

        analyzer_score = _valid_score(result.get("score"))
        if str(result.get("status", "")).casefold() != "completed":
            analyzer_score = None
        llm_item = llm_by_name.get(name, {})
        llm_score = _valid_score(llm_item.get("score")) if isinstance(llm_item, dict) and llm_item.get("origin", "llm") == "llm" else None
        if result.get("requires_executed_testset") is True:
            # An LLM proposal cannot replace missing execution evidence for a
            # metric whose configured analyzer requires assignment testcases.
            if llm_score is not None:
                result["unvalidated_llm_proposal"] = llm_score
            llm_score = None

        selected = llm_score if llm_score is not None else analyzer_score
        origin = "llm" if llm_score is not None else "deterministic_analyzer" if analyzer_score is not None else "unmeasured"
        test_cap_applied = False
        test_score_override_applied = False
        test_total = result.get("tests_total")
        test_passed = result.get("tests_passed")
        test_rows = result.get("results")
        if isinstance(test_rows, list) and test_rows:
            valid_rows = [row for row in test_rows if isinstance(row, dict) and isinstance(row.get("passed"), bool)]
            if valid_rows:
                # Row-level outcomes are authoritative when present; aggregate
                # counts may be stale after merged test variations.
                test_total = len(valid_rows)
                test_passed = sum(row.get("passed") is True for row in valid_rows)
        observed_test_score = None
        if (
            isinstance(test_total, int) and not isinstance(test_total, bool) and test_total > 0
            and isinstance(test_passed, int) and not isinstance(test_passed, bool)
            and 0 <= test_passed <= test_total
        ):
            observed_test_score = round(100.0 * test_passed / test_total, 2)
        if (
            name in {"correctness", "logical"}
            and str(result.get("status", "")).casefold() == "completed"
            and observed_test_score is not None
        ):
            # For these two test-defined metrics, executed outcomes are the
            # score. Keep any model proposal in llm_review_score for audit.
            selected = observed_test_score
            origin = "deterministic_test_results"
            test_score_override_applied = llm_score is not None and not math.isclose(
                llm_score, observed_test_score, rel_tol=0.0, abs_tol=1e-9
            )
            test_cap_applied = llm_score is not None and observed_test_score < llm_score
        result.update({
            "analyzer_score": analyzer_score,
            "llm_review_score": llm_score,
            "llm_rsi_score": llm_score,
            "score": round(selected, 2) if selected is not None else None,
            "score_scale": "0-100",
            "score_origin": origin,
            "scoring_status": "measured" if selected is not None else "inconclusive",
            "score_band": band_for_score(selected) if selected is not None else None,
            "test_score_cap_applied": test_cap_applied,
            "test_score_override_applied": test_score_override_applied,
        })
        explanation = (
            llm_item.get("rationale") if origin == "llm" else
            result.get("summary") or result.get("rationale") or result.get("message") or result.get("reason")
        ) if isinstance(llm_item, dict) else (result.get("summary") or result.get("rationale") or result.get("reason"))
        if not explanation:
            explanation = f"{name.replace('_', ' ')} could not be measured from available evidence."
        if origin == "deterministic_test_results":
            explanation = (
                f"Configured {name} tests passed {test_passed}/{test_total}; the score is the observed pass rate "
                f"({observed_test_score:.2f}/100). "
                + (f"The LLM proposed {llm_score:.2f}/100 and its proposal is retained for audit. " if llm_score is not None else "")
                + str(explanation)
            )
        result["score_rationale"] = compact_rationale(explanation, max_words=60, max_chars=420)
        if isinstance(llm_item, dict) and llm_item:
            result["llm_review"] = {
                "proposed_score": llm_score, "effective_score": selected,
                "test_score_cap_applied": test_cap_applied, "rationale": llm_item.get("rationale", ""),
                "evidence_ids": llm_item.get("evidence_ids", []),
                "confidence": llm_item.get("confidence"),
                "iterations_requested": llm_item.get("iterations_requested", 1),
                "iterations_completed": llm_item.get("iterations_completed", 1),
                "iteration_status": llm_item.get("iteration_status", "completed"),
            }


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
            "formula": "sum(metric_weight * validated_metric_score)",
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
        result = result if isinstance(result, dict) else {}
        score = _valid_score(result.get("score"))
        status = str(result.get("status", "missing"))
        score_origin = str(result.get("score_origin") or ("deterministic_analyzer" if status == "completed" else ""))
        if score is None or score_origin not in {"llm", "llm_test_capped", "deterministic_test_results", "deterministic_analyzer"}:
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
            "formula": "sum(metric_weight * validated_metric_score)",
            "metric_fusion": (
                "Executed correctness/logical test pass rates are authoritative; validated LLM rubric scores are primary for other metrics; completed analyzer scores are fallback. "
                "No neutral or zero score is fabricated when evidence is unavailable."
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
            "score_origin": result.get("score_origin") or "deterministic_analyzer",
            "test_score_cap_applied": result.get("test_score_cap_applied", False),
            "test_score_override_applied": result.get("test_score_override_applied", False),
            "llm_review_score": _valid_score(result.get("llm_review_score")),
        })

    return {
        "status": "completed",
        "score": round(weighted_total, 2),
        "scale": "0-100",
        "weights_sum": round(total_weight, 6),
        "contributions": contributions,
        "warnings": [],
        "formula": "sum(metric_weight * validated_metric_score)",
        "metric_fusion": (
            "Executed correctness/logical test pass rates determine those metrics; validated LLM metric scores determine other metrics when present; deterministic analyzer scores are fallback."
        ),
    }
