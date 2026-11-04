"""Evidence-first metric selection and consolidated point calculation.

Executed correctness/logical tests determine their pass-rate scores. Validated
rubric point estimates assess other metrics where completed analyzer evidence
is unavailable.
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
        max_points = float(metric.get("max_points", result.get("max_points", 100.0)))
        analyzer_score = _valid_score(result.get("score"))
        if str(result.get("status", "")).casefold() != "completed":
            analyzer_score = None
        llm_item = llm_by_name.get(name, {})
        llm_points = None
        llm_score = None
        if isinstance(llm_item, dict) and llm_item.get("origin", "llm") == "llm":
            llm_rationale = str(llm_item.get("rationale", ""))
            llm_claims_unmeasured = bool(re.search(
                r"\b(?:could not|cannot|unable to)\s+be measured\b|\bnot measured\b|\binsufficient evidence\b",
                llm_rationale,
                re.IGNORECASE,
            ))
            candidate = llm_item.get("score_points")
            if not llm_claims_unmeasured and (
                not isinstance(candidate, bool)
                and isinstance(candidate, (int, float))
                and math.isfinite(float(candidate))
                and 0.0 <= float(candidate) <= max_points
            ):
                llm_points = float(candidate)
                llm_score = 100.0 * llm_points / max_points if max_points > 0 else None
        analyzer_requires_execution = (
            analyzer_name in deterministic_metrics
            and (
                str(result.get("status", "")).casefold() != "completed"
                or analyzer_score is None
            )
        )
        if result.get("requires_executed_testset") is True or analyzer_requires_execution:
            # Model points cannot replace missing execution evidence for a
            # metric whose configured analyzer requires assignment testcases.
            llm_score = None
            llm_points = None

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
            # score; model estimates never replace the observed pass rate.
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
            "score": round(selected, 2) if selected is not None else None,
            "score_scale": "0-100",
            "score_points": (
                round(float(selected) * max_points / 100.0, 2)
                if selected is not None else None
            ),
            "max_points": max_points,
            "grade_contribution": bool(metric.get("grade_contribution", True)),
            "score_origin": origin,
            "scoring_status": "measured" if selected is not None else "inconclusive",
            "score_evidence_class": evidence_class,
            "score_provenance": {
                "score": round(selected, 2) if selected is not None else None,
                "points_awarded": (
                    round(float(selected) * max_points / 100.0, 2)
                    if selected is not None else None
                ),
                "max_points": max_points,
                "weight": metric.get("weight"),
                "source": origin,
                "evidence_class": evidence_class,
                "analyzer": analyzer_name or None,
                "analyzer_status": result.get("status"),
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
        if not explanation and origin == "deterministic_analyzer" and analyzer_score is not None:
            if analyzer_name == "compiler_diagnostics":
                explanation = (
                    f"Compilation succeeded (exit code {result.get('exit_code')}); "
                    f"{result.get('warning_count', 0)} warning(s) and "
                    f"{result.get('error_count', 0)} error(s) were recorded."
                )
            elif analyzer_name == "memory_safety":
                total = result.get("tests_checked", result.get("tests_total", 0))
                clean = result.get("tests_clean")
                if isinstance(total, int) and total > 0 and isinstance(clean, int):
                    explanation = f"Memory-safety execution completed: {clean}/{total} test run(s) were classified as clean."
            elif analyzer_name == "runtime_profile":
                samples = result.get("samples", [])
                count = len(samples) if isinstance(samples, list) else 0
                median = result.get("median_runtime_ms")
                if count and isinstance(median, (int, float)):
                    explanation = f"Runtime profiling completed with {count} sample(s); median runtime was {float(median):.3f} ms."
            if not explanation:
                explanation = (
                    f"The completed {analyzer_name or 'deterministic analyzer'} returned "
                    f"a score of {analyzer_score:.2f}/100 from its recorded measurement."
                )
        if not explanation:
            explanation = f"{name.replace('_', ' ')} could not be measured from available evidence."
        elif origin == "deterministic_analyzer" and analyzer_score is not None and re.search(
            r"\b(?:could not|cannot|unable to)\s+be measured\b|\bnot measured\b|\binsufficient evidence\b",
            str(explanation),
            re.IGNORECASE,
        ):
            explanation = (
                f"The completed {analyzer_name or 'deterministic analyzer'} returned "
                f"a score of {analyzer_score:.2f}/100 from its recorded measurement."
            )
        if origin == "deterministic_test_results":
            explanation = (
                f"Configured {name} tests passed {test_passed}/{test_total}; the score is the observed pass rate "
                f"({observed_test_score:.2f}/100). "
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


def calculate_overall_score(
    metrics: list[dict[str, Any]],
    metric_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Sum configured metric points only when every graded metric is measured."""
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
            "formula": "sum(metric_points_awarded)",
        }

    total_weight = sum(float(item["weight"]) for item in enabled)
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Enabled metric weights must sum to 1.0, got {total_weight:.6f}."
        )
    total_points = sum(float(item.get("max_points", float(item["weight"]) * 100.0)) for item in enabled)
    if not math.isclose(total_points, 100.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Enabled metric point caps must sum to 100, got {total_points:.6f}."
        )
    for item in enabled:
        maximum = float(item.get("max_points", float(item["weight"]) * 100.0))
        expected_weight = maximum / 100.0
        if not math.isclose(float(item["weight"]), expected_weight, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"Metric '{item['name']}' weight must match its point cap ({expected_weight:.6f})."
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
            "contributions": [],
            "missing_metrics": missing,
            "warnings": [
                "No final grade was calculated because one or more enabled metrics "
                "could not be measured. Resolve the listed causes and rerun."
            ],
            "formula": "sum(metric_points_awarded)",
            "metric_fusion": (
                "Executed correctness/logical test pass rates are authoritative; validated rubric point estimates are used only when analyzer evidence is unavailable. "
                "No neutral or zero score is fabricated when evidence is unavailable."
            ),
        }

    contributions: list[dict[str, Any]] = []
    points_total = 0.0
    for metric in enabled:
        name = metric["name"]
        result = metric_results[name]
        score = _valid_score(result.get("score"))
        if score is None:  # Defensive: missing scores were rejected above.
            raise AssertionError(f"Validated metric '{name}' lost its score.")
        maximum = float(metric.get("max_points", float(metric["weight"]) * 100.0))
        points_awarded = round(float(score) * maximum / 100.0, 2)
        points_total += points_awarded
        contributions.append({
            "metric": name,
            "points_awarded": points_awarded,
            "max_points": maximum,
            "score_origin": result.get("score_origin") or "deterministic_analyzer",
            "evidence_class": result.get("score_evidence_class", "unavailable"),
        })

    return {
        "status": "completed",
        "score": round(points_total, 2),
        "scale": "0-100",
        "contributions": contributions,
        "warnings": [],
        "formula": "sum(metric_points_awarded)",
        "metric_fusion": (
            "Executed test results and completed analyzers are authoritative where available; validated rubric point estimates fill only metrics without deterministic scores."
        ),
    }
