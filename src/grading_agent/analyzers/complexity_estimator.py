"""Heuristic cyclomatic-complexity and function-length assessment."""
from __future__ import annotations
from typing import Any
from grading_agent.analyzers.source_metrics import analyze_source


def run_complexity_estimator(
    source_code: str,
    max_cyclomatic: int = 8,
    max_function_length: int = 40,
    complexity_penalty: float = 8.0,
    length_penalty_per_line: float = 0.5,
) -> dict[str, Any]:
    metrics = analyze_source(source_code)
    functions = metrics["functions"]
    ccn_violations = [item for item in functions if item["cyclomatic_complexity_estimate"] > max_cyclomatic]
    length_violations = [item for item in functions if item["length"] > max_function_length]
    penalty = sum((item["cyclomatic_complexity_estimate"] - max_cyclomatic) * complexity_penalty for item in ccn_violations)
    penalty += sum((item["length"] - max_function_length) * length_penalty_per_line for item in length_violations)
    return {
        "status": "completed",
        "functions": functions,
        "max_cyclomatic_complexity_estimate": metrics["max_cyclomatic_complexity_estimate"],
        "max_function_length": metrics["max_function_length"],
        "thresholds": {"max_cyclomatic": max_cyclomatic, "max_function_length": max_function_length},
        "complexity_violations": ccn_violations,
        "length_violations": length_violations,
        "score": round(max(0.0, 100.0 - penalty), 2),
        "score_scale": "0-100",
        "measurement_note": metrics["measurement_note"],
    }
