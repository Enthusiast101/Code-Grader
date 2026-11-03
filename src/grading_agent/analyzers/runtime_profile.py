"""Lightweight repeated wall-time profiling for configured inputs."""
from __future__ import annotations
import statistics
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from grading_agent.sandbox.docker_sandbox import DockerSandbox
from grading_agent.config import settings


def run_runtime_profile(
    sandbox: DockerSandbox,
    source_code: str,
    test_cases: list[dict[str, Any]],
    filename: str = settings.DEFAULT_SOURCE_FILENAME,
    compile_timeout_s: int = settings.ANALYZER_COMPILE_TIMEOUT_S,
    run_timeout_s: float = settings.ANALYZER_RUNTIME_RUN_TIMEOUT_S,
    target_runtime_ms: float = settings.ANALYZER_RUNTIME_TARGET_MS,
    repetitions: int = settings.ANALYZER_RUNTIME_REPETITIONS,
) -> dict[str, Any]:
    if repetitions < 1 or repetitions > settings.ANALYZER_RUNTIME_MAX_REPETITIONS:
        raise ValueError(f"repetitions must be between 1 and {settings.ANALYZER_RUNTIME_MAX_REPETITIONS}.")
    if target_runtime_ms <= 0:
        raise ValueError("target_runtime_ms must be positive.")
    if not test_cases:
        return {
            "status": "inconclusive",
            "score": None,
            "score_scale": "0-100",
            "samples": [],
            "target_runtime_ms": target_runtime_ms,
            "repetitions": repetitions,
            "measurement_status": "not_measured",
            "reason": "No assignment-specific runtime inputs were configured; runtime performance was not measured.",
        }
    sandbox.write_file(filename, source_code)
    compile_started = time.perf_counter()
    compile_result = sandbox.run(["gcc", "-std=c11", "-O2", filename, "-o", "student_program_profile"], timeout_s=compile_timeout_s)
    compile_ms = round((time.perf_counter() - compile_started) * 1000.0, 3)
    if compile_result["exit_code"] != 0 or compile_result["timed_out"]:
        return {
            "status": "inconclusive" if compile_result["timed_out"] else "failed",
            "score": None,
            "measurement_status": "not_measured",
            "compile_status": "timed_out" if compile_result["timed_out"] else "failed",
            "compile_elapsed_ms": compile_ms,
            "compile_exit_code": compile_result["exit_code"],
            "compiler_output": compile_result["output"],
            "samples": [],
            "target_runtime_ms": target_runtime_ms,
            "repetitions": repetitions,
            "reason": "The profiling binary could not be compiled; runtime was not measured.",
        }

    samples = []
    all_durations: list[float] = []
    failures: list[dict[str, Any]] = []
    for test_index, case in enumerate(test_cases, 1):
        durations = []
        failure = None
        for repetition in range(repetitions):
            started = time.perf_counter()
            execution = sandbox.run(["./student_program_profile"], timeout_s=run_timeout_s, stdin=case["input"])
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            measured_ms = round(elapsed_ms, 3)
            durations.append(measured_ms)
            all_durations.append(measured_ms)
            if execution.get("timed_out") or execution.get("exit_code") != 0:
                failure = {
                    "repetition": repetition + 1,
                    "exit_code": execution.get("exit_code"),
                    "timed_out": bool(execution.get("timed_out")),
                    "output_truncated": bool(execution.get("output_may_be_truncated")),
                    "output": str(execution.get("output", ""))[-2_000:],
                }
                failures.append({
                    "test_id": case.get("id", test_index),
                    "description": case.get("description", ""),
                    **failure,
                })
                break
        samples.append({
            "test_id": case.get("id", test_index),
            "description": case.get("description", ""),
            "tags": case.get("tags", []),
            "durations_ms": durations,
            "median_ms": round(statistics.median(durations), 3) if durations else None,
            "failure": failure,
        })

    medians = [item["median_ms"] for item in samples if item["median_ms"] is not None]
    if not medians:
        return {
            "status": "inconclusive", "score": None, "score_scale": "0-100",
            "measurement_status": "not_measured", "samples": samples, "failures": failures,
            "target_runtime_ms": target_runtime_ms, "repetitions": repetitions,
            "reason": "No valid runtime samples were collected.",
        }
    p95 = sorted(medians)[min(len(medians) - 1, int(0.95 * len(medians)))]
    target_met = p95 <= target_runtime_ms
    score = 0.0 if failures else min(100.0, 100.0 * target_runtime_ms / max(target_runtime_ms, p95))
    return {
        "status": "completed",
        "measurement_status": "measured_with_execution_failures" if failures else "measured",
        "compile_status": "completed",
        "compile_elapsed_ms": compile_ms,
        "compile_exit_code": compile_result["exit_code"],
        "samples": samples,
        "failures": failures,
        "mean_runtime_ms": round(statistics.mean(all_durations), 3),
        "median_runtime_ms": round(statistics.median(all_durations), 3),
        "median_per_test_runtime_ms": round(statistics.median(medians), 3),
        "max_runtime_ms": round(max(all_durations), 3),
        "p95_per_test_median_ms": round(p95, 3),
        "target_runtime_ms": target_runtime_ms,
        "target_runtime_met": target_met and not failures,
        "target_runtime_basis": "95th percentile of per-test median wall times",
        "repetitions": repetitions,
        "score": round(score, 2),
        "score_scale": "0-100",
        "measurement_note": "Wall time includes Docker exec overhead; use for relative grading in the same environment, not as isolated CPU time. Any timeout or non-zero exit scores zero and is listed as an execution failure.",
    }
