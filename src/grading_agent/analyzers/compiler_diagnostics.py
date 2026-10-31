"""GCC syntax and warning diagnostics."""
from __future__ import annotations

import math
from typing import Any

from grading_agent.sandbox.docker_sandbox import DockerSandbox


def run_compiler_diagnostics(
    sandbox: DockerSandbox,
    source_code: str,
    filename: str = "main.c",
    timeout_s: float = 10,
    flags: list[str] | None = None,
    warning_penalty: float = 5.0,
) -> dict[str, Any]:
    if not math.isfinite(warning_penalty) or not 0 <= warning_penalty <= 100:
        raise ValueError("warning_penalty must be between 0 and 100.")
    flags = flags or ["-std=c11", "-Wall", "-Wextra", "-Wpedantic"]
    if not isinstance(flags, list) or not all(isinstance(item, str) for item in flags):
        raise ValueError("compiler flags must be a list of strings.")

    sandbox.write_file(filename, source_code)
    result = sandbox.run(["gcc", *flags, "-fsyntax-only", filename], timeout_s=timeout_s)
    output = result["output"]
    warnings = [line.strip() for line in output.splitlines() if "warning:" in line.lower()]
    errors = [line.strip() for line in output.splitlines() if "error:" in line.lower()]
    timed_out = bool(result["timed_out"])
    compiled = result["exit_code"] == 0 and not timed_out and not errors
    score = 0.0 if not compiled else max(0.0, 100.0 - len(warnings) * warning_penalty)
    return {
        "status": "completed" if not timed_out else "inconclusive",
        "compiled": compiled,
        "exit_code": result["exit_code"],
        "timed_out": timed_out,
        "warning_count": len(warnings),
        "warnings": warnings,
        "error_count": len(errors),
        "errors": errors,
        "compiler_output": output,
        "score": round(score, 2) if not timed_out else None,
        "score_scale": "0-100",
    }
