"""Lightweight repeated wall-time profiling for configured inputs."""
from __future__ import annotations
import statistics
import time
from typing import Any
from grading_agent.sandbox.docker_sandbox import DockerSandbox


def run_runtime_profile(
    sandbox: DockerSandbox,
    source_code: str,
    test_cases: list[dict[str, str]],
    filename: str = "main.c",
    compile_timeout_s: int = 10,
    run_timeout_s: float = 2.0,
    target_runtime_ms: float = 50.0,
    repetitions: int = 3,
) -> dict[str, Any]:
    if repetitions < 1 or repetitions > 10:
        raise ValueError("repetitions must be between 1 and 10.")
    if target_runtime_ms <= 0:
        raise ValueError("target_runtime_ms must be positive.")
    sandbox.write_file(filename, source_code)
    compile_result = sandbox.run(["gcc", "-std=c11", "-O2", filename, "-o", "student_program_profile"], timeout_s=compile_timeout_s)
    if compile_result["exit_code"] != 0 or compile_result["timed_out"]:
        return {"status": "failed", "score": 0.0, "compiler_output": compile_result["output"], "samples": []}

    samples = []
    for test_index, case in enumerate(test_cases, 1):
        durations = []
        failure = None
        for repetition in range(repetitions):
            started = time.perf_counter()
            execution = sandbox.run(["./student_program_profile"], timeout_s=run_timeout_s, stdin=case["input"])
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            durations.append(round(elapsed_ms, 3))
            if execution["timed_out"] or execution["exit_code"] != 0:
                failure = {"exit_code": execution["exit_code"], "timed_out": execution["timed_out"], "output": execution["output"]}
                break
        samples.append({"test_id": test_index, "durations_ms": durations, "median_ms": round(statistics.median(durations), 3) if durations else None, "failure": failure})

    medians = [item["median_ms"] for item in samples if item["median_ms"] is not None]
    if len(medians) != len(test_cases) or not medians:
        return {"status": "completed", "score": 0.0, "score_scale": "0-100", "samples": samples, "error": "One or more profiling executions failed or timed out; runtime metric receives zero."}
    p95 = sorted(medians)[min(len(medians) - 1, int(0.95 * len(medians)))]
    score = min(100.0, 100.0 * target_runtime_ms / max(target_runtime_ms, p95))
    return {
        "status": "completed",
        "samples": samples,
        "median_runtime_ms": round(statistics.median(medians), 3),
        "p95_per_test_median_ms": round(p95, 3),
        "target_runtime_ms": target_runtime_ms,
        "repetitions": repetitions,
        "score": round(score, 2),
        "score_scale": "0-100",
        "measurement_note": "Wall time includes Docker exec overhead; use for relative grading in the same environment, not as isolated CPU time.",
    }
