"""Evidence-first metric selection and weighted grade calculation.

Executed correctness/logical tests determine their pass-rate scores. Validated
LLM rubric scores assess other metrics and criteria without executed tests;
completed analyzers are the fallback when an LLM score is unavailable.
"""
from __future__ import annotations

import math
import re
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
    deterministic_metrics = {
        "io_tests", "edge_case_runner", "runtime_profile", "memory_safety",
        "compiler_diagnostics",
    }

    for metric in metrics:
        if not metric.get("enabled", True):
            continue
        name = metric["name"]
        result = report.setdefault(name, {})
        if not isinstance(result, dict):
            result = {}
            report[name] = result

        analyzer_name = str(result.get("analyzer") or metric.get("analyzer") or "")
        result.setdefault("weight", metric.get("weight"))
        result.setdefault("max_points", metric.get("max_points", float(metric.get("weight", 0.0)) * 100.0))
        result.setdefault("grade_contribution", metric.get("grade_contribution", True))
        analyzer_score = _valid_score(result.get("score"))
        if str(result.get("status", "")).casefold() != "completed":
            analyzer_score = None
        llm_item = llm_by_name.get(name, {})
        llm_score_10 = None
        if isinstance(llm_item, dict) and llm_item.get("origin", "llm") == "llm":
            candidate = llm_item.get("score")
            if (
                not isinstance(candidate, bool)
                and isinstance(candidate, (int, float))
                and math.isfinite(float(candidate))
                and 0.0 <= float(candidate) <= 10.0
            ):
                llm_score_10 = float(candidate)
        proposed_score_10 = llm_score_10
        llm_score = llm_score_10 * 10.0 if llm_score_10 is not None else None
        analyzer_requires_execution = (
            analyzer_name in deterministic_metrics
            and (
                str(result.get("status", "")).casefold() != "completed"
                or analyzer_score is None
            )
        )
        if result.get("requires_executed_testset") is True or analyzer_requires_execution:
            # An LLM proposal cannot replace missing execution evidence for a
            # metric whose configured analyzer requires assignment testcases.
            if llm_score_10 is not None:
                result["unvalidated_llm_proposal"] = llm_score_10
            llm_score = None
            llm_score_10 = None

        deterministic_is_authoritative = analyzer_name in deterministic_metrics
        selected = (
            analyzer_score if deterministic_is_authoritative and analyzer_score is not None
            else llm_score if llm_score is not None
            else analyzer_score
        )
        origin = (
            "deterministic_analyzer" if deterministic_is_authoritative and analyzer_score is not None
            else "llm" if llm_score is not None
            else "deterministic_analyzer" if analyzer_score is not None
            else "unmeasured"
        )
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
        evidence_class = (
            "measured" if origin in {"deterministic_test_results", "deterministic_analyzer"}
            else "estimated" if origin == "llm"
            else "unavailable"
        )
        available_evidence: list[str] = []
        missing_evidence: list[str] = []
        if isinstance(test_total, int) and test_total > 0:
            available_evidence.append(f"executed tests: {test_passed}/{test_total} passed")
        if isinstance(result.get("exit_code"), int):
            available_evidence.append(f"compiler exit code: {result['exit_code']}")
        if isinstance(result.get("median_runtime_ms"), (int, float)):
            available_evidence.append("runtime samples and per-test medians")
        runtime_tools = result.get("runtime_tools")
        if isinstance(runtime_tools, dict):
            for tool, key in (("Valgrind", "valgrind_status"), ("ASan/UBSan", "asan_ubsan_status")):
                tool_status = runtime_tools.get(key, "unknown")
                if tool_status == "completed":
                    available_evidence.append(f"{tool} execution")
                else:
                    missing_evidence.append(f"{tool} execution ({tool_status})")
        if isinstance(llm_item, dict) and llm_item.get("evidence_ids"):
            available_evidence.append("validated LLM evidence references")
        if selected is None and result.get("reason"):
            missing_evidence.append(str(result["reason"]))
        result.update({
            "analyzer_score": analyzer_score,
            "llm_review_score": proposed_score_10,
            "llm_rsi_score": proposed_score_10,
            "llm_score_10": proposed_score_10,
            "score": round(selected, 2) if selected is not None else None,
            "score_scale": "0-100",
            "score_points": (
                round(float(selected) * float(metric.get("max_points", result.get("max_points", 100.0))) / 100.0, 2)
                if selected is not None else None
            ),
            "max_points": float(metric.get("max_points", result.get("max_points", 100.0))),
            "grade_contribution": bool(metric.get("grade_contribution", True)),
            "score_origin": origin,
            "scoring_status": "measured" if selected is not None else "inconclusive",
            "score_evidence_class": evidence_class,
            "score_provenance": {
                "score": round(selected, 2) if selected is not None else None,
                "points_awarded": (
                    round(float(selected) * float(metric.get("max_points", result.get("max_points", 100.0))) / 100.0, 2)
                    if selected is not None else None
                ),
                "max_points": float(metric.get("max_points", result.get("max_points", 100.0))),
                "weight": metric.get("weight"),
                "source": origin,
                "evidence_class": evidence_class,
                "analyzer": analyzer_name or None,
                "analyzer_status": result.get("status"),
                "unvalidated_llm_proposal_10": proposed_score_10 if llm_score is None else None,
                "reason": result.get("reason") if selected is None else None,
                "available_evidence": available_evidence,
                "missing_evidence": missing_evidence,
            },
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
                + (f"The LLM proposed {proposed_score_10:.2f}/10 and its proposal is retained for audit. " if proposed_score_10 is not None else "")
                + (str(explanation) if explanation and not re.search(
                    r"could not be measured|no .*evidence|not measured", str(explanation), re.IGNORECASE
                ) else "The configured test suite executed and its pass rate is the score for this metric.")
            )
        elif origin == "deterministic_analyzer" and analyzer_name in deterministic_metrics:
            explanation = (
                f"The completed {analyzer_name} measurement is authoritative for this metric. "
                + (str(explanation) if explanation else "")
            )
        result["score_rationale"] = compact_rationale(explanation, max_words=60, max_chars=420)
        if isinstance(llm_item, dict) and llm_item:
            result["llm_review"] = {
                "proposed_score": proposed_score_10, "score_scale": "0-10",
                "effective_score_100": selected,
                "effective_points": result.get("score_points"),
                "max_points": result.get("max_points"),
                "used_for_score": origin == "llm",
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
    enabled = [
        item for item in metrics
        if item.get("enabled", True) and item.get("grade_contribution", True)
    ]
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
        contribution = round(weight * score, 2)
        weighted_total += contribution
        contributions.append({
            "metric": name,
            "analyzer": metric.get("analyzer"),
            "weight": weight,
            "score": round(score, 2),
            "contribution": round(contribution, 2),
            "points_awarded": round(float(score) * float(metric.get("max_points", weight * 100.0)) / 100.0, 2),
            "max_points": float(metric.get("max_points", weight * 100.0)),
            "score_origin": result.get("score_origin") or "deterministic_analyzer",
            "test_score_cap_applied": result.get("test_score_cap_applied", False),
            "test_score_override_applied": result.get("test_score_override_applied", False),
            "llm_review_score": _valid_score(result.get("llm_review_score")),
            "evidence_class": result.get("score_evidence_class", "unavailable"),
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
            "Executed test results determine correctness/logical metrics; completed compiler, runtime, and memory analyzers determine their metrics; validated LLM assessments are estimates for remaining rubric metrics."
        ),
    }
