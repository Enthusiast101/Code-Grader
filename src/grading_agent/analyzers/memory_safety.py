"""Valgrind-based memory-safety analyzer for student C submissions.

The analyzer preserves raw Valgrind logs and also extracts structured
summaries for deterministic checks and downstream LLM review.
"""

from __future__ import annotations

import re
from typing import Any

from grading_agent.sandbox.docker_sandbox import DockerSandbox


LEAK_CATEGORIES = (
    "definitely_lost",
    "indirectly_lost",
    "possibly_lost",
    "still_reachable",
)

RELEVANT_DIAGNOSTICS = (
    "Invalid read",
    "Invalid write",
    "Invalid free",
    "Mismatched free",
    "Use of uninitialised value",
    "Use of uninitialized value",
    "Conditional jump or move depends on uninitialised value",
    "Conditional jump or move depends on uninitialized value",
    "Source and destination overlap",
    "Syscall param",
)


def _summary_value(log: str, label: str) -> dict[str, int | None]:
    """Extract a Valgrind memory category, preserving unknown as None."""
    pattern = (
        rf"{re.escape(label)}:\s*([\d,]+)\s+bytes\s+"
        rf"in\s+([\d,]+)\s+blocks"
    )
    match = re.search(pattern, log, re.IGNORECASE)

    if not match:
        return {"bytes": None, "blocks": None}

    return {
        "bytes": int(match.group(1).replace(",", "")),
        "blocks": int(match.group(2).replace(",", "")),
    }


def _parse_valgrind_log(log: str) -> dict[str, Any]:
    """Parse summaries while preserving the distinction between zero and unknown."""
    normalized_log = re.sub(r"(?m)^==\d+==\s*", "", log)

    error_match = re.search(
        r"ERROR SUMMARY:\s*(\d+)\s+errors?",
        normalized_log,
        re.IGNORECASE,
    )

    summaries = {
        "definitely_lost": _summary_value(normalized_log, "definitely lost"),
        "indirectly_lost": _summary_value(normalized_log, "indirectly lost"),
        "possibly_lost": _summary_value(normalized_log, "possibly lost"),
        "still_reachable": _summary_value(normalized_log, "still reachable"),
    }

    # Some Valgrind versions omit the category-by-category leak summary
    # when no heap blocks remain allocated. In that specific case, these
    # missing categories are known to be zero, not unknown.
    all_blocks_freed = bool(
        re.search(
            r"All heap blocks were freed\s*--\s*no leaks are possible",
            normalized_log,
            re.IGNORECASE,
        )
    )
    if all_blocks_freed:
        for category in LEAK_CATEGORIES:
            if summaries[category]["bytes"] is None:
                summaries[category] = {"bytes": 0, "blocks": 0}

    diagnostics: list[str] = []
    for line in normalized_log.splitlines():
        if any(term.casefold() in line.casefold() for term in RELEVANT_DIAGNOSTICS):
            cleaned = line.strip()
            if cleaned and cleaned not in diagnostics:
                diagnostics.append(cleaned)

    return {
        "error_count": int(error_match.group(1)) if error_match else None,
        **summaries,
        "all_heap_blocks_freed": all_blocks_freed,
        "diagnostics": diagnostics,
    }


def run_memory_safety(
    sandbox: DockerSandbox,
    source_code: str,
    test_cases: list[dict[str, str]],
    filename: str = "main.c",
    compile_timeout_s: int = 10,
    run_timeout_s: int = 5,
) -> dict[str, Any]:
    """Compile a C submission and run Valgrind Memcheck for each test input.

    Each test case must contain string fields ``input`` and ``expected``.
    This analyzer reports memory diagnostics; the I/O analyzer evaluates
    whether the output matches ``expected``.
    """
    if not test_cases:
        return {
            "analyzer": "valgrind_memcheck",
            "compiled": False,
            "compiler_output": "No test cases supplied; memory analysis not run.",
            "tests_total": 0,
            "tests_checked": 0,
            "tests_clean": 0,
            "memory_safe": False,
            "analysis_error": True,
            "results": [],
        }

    for index, case in enumerate(test_cases, start=1):
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("input"), str)
            or not isinstance(case.get("expected"), str)
        ):
            raise ValueError(
                f"Test case {index} must contain string fields 'input' and 'expected'."
            )

    sandbox.write_file(filename, source_code)

    compilation = sandbox.run(
        [
            "gcc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-g",
            "-O0",
            "-fno-omit-frame-pointer",
            filename,
            "-o",
            "student_program_memcheck",
        ],
        timeout_s=compile_timeout_s,
    )

    if compilation["timed_out"] or compilation["exit_code"] != 0:
        return {
            "analyzer": "valgrind_memcheck",
            "compiled": False,
            "compiler_output": compilation["output"],
            "tests_total": len(test_cases),
            "tests_checked": 0,
            "tests_clean": 0,
            "memory_safe": False,
            "analysis_error": False,
            "results": [],
        }

    results: list[dict[str, Any]] = []

    for index, case in enumerate(test_cases, start=1):
        log_filename = f"valgrind_test_{index}.log"

        execution = sandbox.run(
            [
                "valgrind",
                "--tool=memcheck",
                "--leak-check=full",
                "--show-leak-kinds=all",
                "--errors-for-leak-kinds=definite,indirect",
                "--track-origins=yes",
                "--error-limit=yes",
                "--num-callers=20",
                "--error-exitcode=99",
                f"--log-file={log_filename}",
                "./student_program_memcheck",
            ],
            timeout_s=run_timeout_s,
            stdin=case["input"],
        )

        # Read through the sandbox so the log itself, not just the program's
        # stdout, is retained. The sandbox enforces its configured output cap.
        log_read_error = None
        log_truncated = False
        try:
            log_result = sandbox.run(["cat", log_filename], timeout_s=2)
            if log_result["exit_code"] == 0:
                raw_log = log_result["output"]
                log_truncated = log_result.get("output_may_be_truncated", False)
            else:
                raw_log = ""
                log_read_error = log_result["output"] or (
                    f"Could not read Valgrind log (exit {log_result['exit_code']})."
                )
        except Exception as exc:  # retain the failure as evidence
            raw_log = ""
            log_read_error = f"{type(exc).__name__}: {exc}"

        parsed = _parse_valgrind_log(raw_log)
        timed_out = bool(execution.get("timed_out"))
        log_available = bool(raw_log) and log_read_error is None

        # Missing, truncated, timed-out, or summary-less runs are inconclusive.
        analysis_error = (
            not log_available
            or log_truncated
            or timed_out
            or parsed["error_count"] is None
        )

        leak_bytes = [
            parsed[category]["bytes"]
            for category in ("definitely_lost", "indirectly_lost", "possibly_lost")
        ]
        leak_summary_known = all(value is not None for value in leak_bytes)
        no_reported_leaks = leak_summary_known and all(value == 0 for value in leak_bytes)
        no_reported_errors = parsed["error_count"] == 0 and not parsed["diagnostics"]

        memory_safe = (
            not analysis_error
            and no_reported_errors
            and no_reported_leaks
        )

        results.append({
            "test_id": index,
            "input": case["input"],
            "program_output": execution["output"],
            "timed_out": timed_out,
            "valgrind_exit_code": execution["exit_code"],
            "analysis_error": analysis_error,
            "log_read_error": log_read_error,
            "log_truncated": log_truncated,
            "memory_safe": memory_safe,
            "parsed_summary": parsed,
            "raw_log": raw_log,
        })

    tests_clean = sum(result["memory_safe"] for result in results)
    complete_results = len(results) == len(test_cases)
    overall_memory_safe = (
        complete_results
        and all(result["memory_safe"] for result in results)
    )

    return {
        "analyzer": "valgrind_memcheck",
        "compiled": True,
        "compiler_output": compilation["output"],
        "tests_total": len(test_cases),
        "tests_checked": len(results),
        "tests_clean": tests_clean,
        "memory_safe": overall_memory_safe,
        "analysis_error": any(result["analysis_error"] for result in results),
        "results": results,
    }
