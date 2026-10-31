"""Cppcheck-based static diagnostics."""
from __future__ import annotations

import re
from typing import Any

from grading_agent.sandbox.docker_sandbox import DockerSandbox
from grading_agent.config import settings


CPP_CHECK_LINE = re.compile(r"^.*?:\d+:\d+:\s*(error|warning|style|performance|portability|information):\s*(.*)$", re.I)


def run_static_analysis(
    sandbox: DockerSandbox,
    source_code: str,
    filename: str = settings.DEFAULT_SOURCE_FILENAME,
    timeout_s: float = settings.ANALYZER_STATIC_TIMEOUT_S,
    severity_penalties: dict[str, float] | None = None,
) -> dict[str, Any]:
    sandbox.write_file(filename, source_code)
    command = [
        "cppcheck", "--enable=warning,style,performance,portability",
        "--inline-suppr", "--language=c", "--std=c11",
        "--template=gcc", "--error-exitcode=0", filename,
    ]
    result = sandbox.run(command, timeout_s=timeout_s)
    if result["exit_code"] == 127:
        return {"status": "unavailable", "score": None, "error": "cppcheck is not installed in the sandbox image."}
    if result["timed_out"]:
        return {"status": "inconclusive", "score": None, "error": "cppcheck timed out.", "raw_output": result["output"]}

    penalties = {**settings.ANALYZER_STATIC_PENALTIES, **(severity_penalties or {})}
    findings: list[dict[str, str]] = []
    for line in result["output"].splitlines():
        match = CPP_CHECK_LINE.match(line.strip())
        if match:
            severity = match.group(1).lower()
            findings.append({"severity": severity, "message": match.group(2), "raw": line.strip()})
    total_penalty = min(100.0, sum(float(penalties.get(item["severity"], 0.0)) for item in findings))
    return {
        "status": "completed",
        "tool": "cppcheck",
        "finding_count": len(findings),
        "findings": findings,
        "severity_counts": {
            severity: sum(item["severity"] == severity for item in findings)
            for severity in penalties
        },
        "raw_output": result["output"],
        "score": round(max(0.0, 100.0 - total_penalty), 2),
        "score_scale": "0-100",
        "scoring_note": "Configured deterministic severity deductions; findings are not proof of exploitability.",
    }
