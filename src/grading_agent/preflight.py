"""Cheap compile and smoke-execution checks before metric dispatch."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from grading_agent.config.rule_loader import load_test_cases


def run_preflight(
    sandbox: Any,
    *,
    source_code: str,
    source_filename: str,
    rules: dict[str, Any],
) -> dict[str, Any]:
    """Compile once and smoke-run the first configured execution test.

    This is an informational first pass. Individual analyzers still perform
    their own builds and enforce their own limits so that they can return
    metric-specific evidence even when the smoke test fails.
    """
    sandbox.write_file(source_filename, source_code)
    compile_result = sandbox.run(
        [
            "gcc", "-std=c11", "-Wall", "-Wextra", "-O2",
            source_filename, "-o", "preflight_program",
        ],
        timeout_s=10.0,
    )

    if compile_result["timed_out"]:
        return {
            "status": "compile_timeout",
            "compiled": False,
            "smoke_tested": False,
            "suspected_infinite_loop": False,
            "compiler_output": compile_result["output"],
        }

    if compile_result["exit_code"] != 0:
        return {
            "status": "compile_failed",
            "compiled": False,
            "smoke_tested": False,
            "suspected_infinite_loop": False,
            "compiler_output": compile_result["output"],
        }

    smoke_case: dict[str, str] | None = None
    for metric in rules.get("metrics", []):
        if not metric.get("enabled", True):
            continue
        if metric.get("analyzer") not in {
            "io_tests", "memory_safety", "edge_case_runner", "runtime_profile"
        }:
            continue
        resolved = metric.get("params", {}).get("resolved_testset_path")
        if resolved:
            cases = load_test_cases(Path(resolved))
            if cases:
                smoke_case = cases[0]
                break

    if smoke_case is None:
        return {
            "status": "compiled_smoke_skipped",
            "compiled": True,
            "smoke_tested": False,
            "suspected_infinite_loop": False,
            "compiler_output": compile_result["output"],
            "note": "No configured execution test input was available for a smoke run.",
        }

    execution = sandbox.run(
        ["./preflight_program"],
        timeout_s=float(rules["time_limit_s"]),
        stdin=smoke_case["input"],
    )
    timed_out = bool(execution["timed_out"])
    crashed = execution["exit_code"] != 0 and not timed_out

    return {
        "status": "timeout" if timed_out else "runtime_failure" if crashed else "passed",
        "compiled": True,
        "smoke_tested": True,
        "suspected_infinite_loop": timed_out,
        "crashed": crashed,
        "smoke_input": smoke_case["input"],
        "smoke_output": execution["output"],
        "exit_code": execution["exit_code"],
        "timed_out": timed_out,
        "output_may_be_truncated": execution["output_may_be_truncated"],
    }
