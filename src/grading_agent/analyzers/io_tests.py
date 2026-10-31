
from typing import Any

from grading_agent.sandbox.docker_sandbox import DockerSandbox


def run_io_tests(
    sandbox: DockerSandbox,
    source_code: str,
    test_cases: list[dict[str, str]],
    filename: str = "main.c",
    compile_timeout_s: int = 10,
    run_timeout_s: int = 2,
) -> dict[str, Any]:
    """
    Compile a student's C program and run it against test cases.

    Each test case must contain:
        input: input passed to stdin
        expected: expected stdout

    Returns a structured result for the grading agent.
    """
    sandbox.write_file(filename, source_code)

    compilation = sandbox.run(
        [
            "gcc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-O2",
            filename,
            "-o",
            "student_program",
        ],
        timeout_s=compile_timeout_s,
    )

    if compilation["exit_code"] != 0 or compilation["timed_out"]:
        return {
            "status": "inconclusive" if compilation["timed_out"] else "completed",
            "compiled": False,
            "compiler_output": compilation["output"],
            "tests_passed": 0,
            "tests_total": len(test_cases),
            "score": None if compilation["timed_out"] else 0.0,
            "score_scale": "0-100",
            "results": [],
        }

    results = []

    for index, case in enumerate(test_cases, start=1):
        execution = sandbox.run(
            ["./student_program"],
            timeout_s=run_timeout_s,
            stdin=case["input"],
        )

        actual = execution["output"]
        expected = case["expected"]

        # Ignore leading/trailing whitespace, but preserve internal content.
        passed = (
            not execution["timed_out"]
            and execution["exit_code"] == 0
            and not execution["output_may_be_truncated"]
            and actual.strip() == expected.strip()
        )

        results.append({
            "test_id": index,
            "passed": passed,
            "input": case["input"],
            "expected": expected,
            "actual": actual,
            "exit_code": execution["exit_code"],
            "timed_out": execution["timed_out"],
            "output_may_be_truncated": (
                execution["output_may_be_truncated"]
            ),
        })

    passed_count = sum(result["passed"] for result in results)

    return {
        "compiled": True,
        "compiler_output": compilation["output"],
        "tests_passed": passed_count,
        "tests_total": len(test_cases),
        "score": (
            100.0 * passed_count / len(test_cases)
            if test_cases else 0.0
        ),
        "score_scale": "0-100",
        "results": results,
    }
