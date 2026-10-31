"""Heuristic source modularity analyzer."""
from __future__ import annotations
from typing import Any
from grading_agent.analyzers.source_metrics import analyze_source


def run_modularity(source_code: str, min_functions: int = 2, expect_helpers: bool = True) -> dict[str, Any]:
    metrics = analyze_source(source_code)
    functions = metrics["functions"]
    helpers = [item for item in functions if item["name"] != "main"]
    target = max(1, min_functions)
    if expect_helpers:
        score = min(100.0, 100.0 * len(helpers) / max(1, target - 1))
    else:
        score = min(100.0, 100.0 * len(functions) / target)
    return {
        "status": "completed",
        "function_count": len(functions),
        "helper_function_count": len(helpers),
        "functions": functions,
        "minimum_functions": min_functions,
        "helpers_expected": expect_helpers,
        "score": round(score, 2),
        "score_scale": "0-100",
        "measurement_note": metrics["measurement_note"],
    }
