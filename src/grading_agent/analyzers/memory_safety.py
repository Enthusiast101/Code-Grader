"""Runtime memory-safety analysis using Valgrind and ASan/UBSan."""
from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from grading_agent.sandbox.docker_sandbox import DockerSandbox

LEAK_CATEGORIES = ("definitely_lost", "indirectly_lost", "possibly_lost", "still_reachable")
RELEVANT_DIAGNOSTICS = (
    "Invalid read", "Invalid write", "Invalid free", "Mismatched free",
    "Use of uninitialised value", "Use of uninitialized value",
    "Conditional jump or move depends on uninitialised value",
    "Source and destination overlap", "Syscall param",
)


def _summary_value(log: str, label: str) -> dict[str, int | None]:
    match = re.search(
        rf"{re.escape(label)}:\s*([\d,]+)\s+bytes\s+in\s+([\d,]+)\s+blocks",
        log, re.IGNORECASE,
    )
    if not match:
        return {"bytes": None, "blocks": None}
    return {"bytes": int(match.group(1).replace(",", "")), "blocks": int(match.group(2).replace(",", ""))}


def _parse_valgrind_log(log: str) -> dict[str, Any]:
    normalized = re.sub(r"(?m)^==\d+==\s*", "", log)
    error_match = re.search(r"ERROR SUMMARY:\s*(\d+)\s+errors?", normalized, re.I)
    summaries = {
        "definitely_lost": _summary_value(normalized, "definitely lost"),
        "indirectly_lost": _summary_value(normalized, "indirectly lost"),
        "possibly_lost": _summary_value(normalized, "possibly lost"),
        "still_reachable": _summary_value(normalized, "still reachable"),
    }
    all_freed = bool(re.search(r"All heap blocks were freed\s*--\s*no leaks are possible", normalized, re.I))
    if all_freed:
        for category in LEAK_CATEGORIES:
            if summaries[category]["bytes"] is None:
                summaries[category] = {"bytes": 0, "blocks": 0}
    diagnostics = [
        line.strip() for line in normalized.splitlines()
        if any(term.casefold() in line.casefold() for term in RELEVANT_DIAGNOSTICS)
    ]
    return {
        "error_count": int(error_match.group(1)) if error_match else None,
        **summaries,
        "all_heap_blocks_freed": all_freed,
        "diagnostics": list(dict.fromkeys(diagnostics)),
    }



def run_memory_safety(
    sandbox: DockerSandbox,
    source_code: str,
    test_cases: list[dict[str, str]],
    filename: str = "main.c",
    compile_timeout_s: int = 10,
    run_timeout_s: int = 5,
    run_sanitizers: bool = True,
) -> dict[str, Any]:
    """Run Valgrind and optional AddressSanitizer/UBSan on every test input."""
    if not test_cases:
        return {"analyzer": "memory_safety", "status": "inconclusive", "compiled": False, "analysis_error": True, "tests_total": 0, "tests_checked": 0, "results": [], "score": None}

    for index, case in enumerate(test_cases, start=1):
        if not isinstance(case, dict) or not isinstance(case.get("input"), str) or not isinstance(case.get("expected"), str):
            raise ValueError(f"Test case {index} requires string input and expected fields.")

    sandbox.write_file(filename, source_code)
    vg_compile = sandbox.run(
        ["gcc", "-std=c11", "-Wall", "-Wextra", "-g", "-O0", "-fno-omit-frame-pointer", filename, "-o", "student_program_memcheck"],
        timeout_s=compile_timeout_s,
    )
    if vg_compile["timed_out"] or vg_compile["exit_code"] != 0:
        return {"analyzer": "memory_safety", "status": "failed", "compiled": False, "compiler_output": vg_compile["output"], "analysis_error": False, "tests_total": len(test_cases), "tests_checked": 0, "results": [], "score": None}

    vg_results: list[dict[str, Any]] = []
    for index, case in enumerate(test_cases, 1):
        # Capture Valgrind's diagnostics through the Docker exec output stream.
        # The sandbox merges stderr into stdout, so Memcheck's default stderr
        # report is captured directly. This avoids relying on Docker archive
        # retrieval for files created on the container's /workspace tmpfs.
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
                "./student_program_memcheck",
            ],
            timeout_s=run_timeout_s,
            stdin=case["input"],
        )
        raw_log = execution.get("output", "") or ""
        read_error = None
        truncated = bool(execution.get("output_may_be_truncated", False))
        parsed = _parse_valgrind_log(raw_log)
        timed_out = bool(execution["timed_out"])
        inconclusive = (
            not raw_log
            or truncated
            or timed_out
            or parsed["error_count"] is None
        )
        leak_values = [parsed[name]["bytes"] for name in ("definitely_lost", "indirectly_lost", "possibly_lost")]
        leaks_known = all(value is not None for value in leak_values)
        clean = (
            not inconclusive and leaks_known and all(value == 0 for value in leak_values)
            and parsed["error_count"] == 0 and not parsed["diagnostics"]
        )
        vg_results.append({
            "test_id": index,
            "input": case["input"],
            # DockerSandbox combines stdout/stderr for bounded capture; this
            # field therefore includes program stdout and Valgrind diagnostics.
            "captured_output": raw_log,
            "timed_out": timed_out,
            "exit_code": execution["exit_code"],
            "analysis_error": inconclusive,
            "log_read_error": read_error,
            "log_truncated": truncated,
            "memory_safe": clean,
            "parsed_summary": parsed,
            "raw_log": raw_log,
        })

    vg_complete = len(vg_results) == len(test_cases) and all(not item["analysis_error"] for item in vg_results)
    vg_score = (100.0 * sum(item["memory_safe"] for item in vg_results) / len(test_cases)) if vg_complete else None

    sanitizer_results: list[dict[str, Any]] = []
    sanitizer_complete = True
    if run_sanitizers:
        asan_compile = sandbox.run(
            ["gcc", "-std=c11", "-g", "-O1", "-fno-omit-frame-pointer", "-fsanitize=address,undefined", "-no-pie", filename, "-o", "student_program_sanitized"],
            timeout_s=compile_timeout_s,
        )
        if asan_compile["timed_out"] or asan_compile["exit_code"] != 0:
            sanitizer_complete = False
            sanitizer_results.append({"status": "unavailable", "compiler_output": asan_compile["output"]})
        else:
            for index, case in enumerate(test_cases, 1):
                run = sandbox.run(
                    ["env", "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1", "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1", "./student_program_sanitized"],
                    timeout_s=run_timeout_s, stdin=case["input"],
                )
                output = run["output"]
                address_issue = "ERROR: AddressSanitizer" in output or "ERROR: LeakSanitizer" in output
                undefined_issue = "runtime error:" in output
                sanitizer_error = address_issue or undefined_issue
                timed_out = bool(run["timed_out"])
                inconclusive = timed_out or run.get("output_may_be_truncated", False) or (run["exit_code"] != 0 and not sanitizer_error)
                if inconclusive:
                    sanitizer_complete = False
                sanitizer_results.append({
                    "test_id": index, "input": case["input"], "output": output,
                    "exit_code": run["exit_code"], "timed_out": timed_out,
                    "address_sanitizer_finding": address_issue,
                    "undefined_behavior_finding": undefined_issue,
                    "sanitizer_error": sanitizer_error,
                    "analysis_error": inconclusive,
                    "clean": not inconclusive and not sanitizer_error,
                })

    sanitizer_score = None
    if run_sanitizers and sanitizer_complete and len(sanitizer_results) == len(test_cases):
        sanitizer_score = 100.0 * sum(item.get("clean", False) for item in sanitizer_results) / len(test_cases)

    # The instructor's configured runtime component weight applies to
    # Valgrind, preserving the rules.json contract. ASan/UBSan runs as
    # supplementary corroborating evidence and is exposed to the LLM/report,
    # but is not silently folded into the Valgrind subweight.
    runtime_tools_score = vg_score
    runtime_tools_complete = vg_score is not None

    return {
        "analyzer": "memory_safety", "status": "completed" if runtime_tools_complete else "inconclusive",
        "compiled": True, "tests_total": len(test_cases), "tests_checked": len(vg_results),
        "tests_clean": sum(item["memory_safe"] for item in vg_results),
        "memory_safe": runtime_tools_complete and runtime_tools_score == 100.0,
        "analysis_error": not runtime_tools_complete,
        "runtime_tools": {
            "status": "completed" if runtime_tools_complete else "inconclusive",
            "score": round(runtime_tools_score, 2) if runtime_tools_score is not None else None,
            "score_scale": "0-100",
            "valgrind_score": round(vg_score, 2) if vg_score is not None else None,
            "asan_ubsan_score": round(sanitizer_score, 2) if sanitizer_score is not None else None,
            "valgrind_results": vg_results,
            "asan_ubsan_results": sanitizer_results,
            "scoring_note": "The configured runtime-tools component score is the Valgrind score; ASan/UBSan is supplementary evidence and is not additionally weighted.",
        },
        # Backward-compatible field for existing memory scoring helper.
        "results": vg_results,
        "raw_compiler_output": vg_compile["output"],
    }
