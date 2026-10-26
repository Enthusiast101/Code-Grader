#used for checking compiler diagnostics, to check for compiler errors
# Score = max(0, 1 - n * p), where n is warning count and p is the penalty per warning.
#default penalty is 0.1
import math
from typing import Any

from grading_agent.sandbox.docker_sandbox import DockerSandbox


def run_static_analysis(
    sandbox: DockerSandbox,
    source_code: str,
    filename: str = "main.c",
    timeout_s: int = 10,
    warning_penalty: float = 0.1,
) -> dict[str, Any]:
    """
    Check C source code for syntax errors and compiler warnings.

    The score starts at 1.0 and decreases for each warning.
    Compilation errors or a timeout result in a score of 0.0.
    """
    if (
        isinstance(warning_penalty, bool)
        or not isinstance(warning_penalty, (int, float))
        or not math.isfinite(warning_penalty)
        or not 0 <= warning_penalty <= 1
    ):
        raise ValueError(
            "warning_penalty must be a finite number between 0 and 1."
        )

    # Place the source file inside the isolated Docker workspace.
    sandbox.write_file(filename, source_code)

    # Ask GCC to check the code without producing an executable.
    analysis = sandbox.run(
        [
            "gcc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-fsyntax-only",
            filename,
        ],
        timeout_s=timeout_s,
    )

    compiler_output = analysis["output"]

    # Extract individual diagnostic lines containing warnings and errors.
    warnings = [
        line.strip()
        for line in compiler_output.splitlines()
        if "warning:" in line.lower()
    ]

    errors = [
        line.strip()
        for line in compiler_output.splitlines()
        if "error:" in line.lower()
    ]

    failed = (
        analysis["exit_code"] != 0
        or analysis["timed_out"]
        or bool(errors)
    )

    # Transparent scoring rule: each warning incurs a configurable penalty.
    if failed:
        score = 0.0
    else:
        score = max(
            0.0,
            1.0 - len(warnings) * warning_penalty,
        )

    return {
        "compiled": not failed,
        "compiler_output": compiler_output,
        "compiler_exit_code": analysis["exit_code"],
        "timed_out": analysis["timed_out"],
        "warning_count": len(warnings),
        "warnings": warnings,
        "score": round(score, 4),
    }
