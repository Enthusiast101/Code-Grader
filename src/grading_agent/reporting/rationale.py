"""Preserve concise but useful 2–3 sentence grading explanations."""
from __future__ import annotations

import re
from typing import Any

def compact_rationale(value: Any, *, max_words: int = 60, max_chars: int = 420) -> str:
    """Normalize a rationale while retaining up to three explanatory sentences."""
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(sentences[:3]).strip()
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(" ,;:") + "…"
    if len(text) > max_chars:
        text = text[: max_chars - 1].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return text


def score_source_label(origin: Any) -> str:
    """Turn internal score provenance into a clear student-facing label."""
    labels = {
        "llm": "validated LLM rubric assessment",
        "llm_test_capped": "LLM rubric assessment capped by configured test results",
        "deterministic_test_results": "executed test pass rate",
        "deterministic_analyzer": "completed deterministic analyzer",
        "unmeasured": "unmeasured",
    }
    return labels.get(str(origin or ""), "score source unavailable")


def evidence_summary(report: dict[str, Any]) -> str:
    """Summarize only explicit test and validated-finding evidence."""
    statements: list[str] = []
    for metric_name in ("correctness", "logical"):
        result = report.get(metric_name, {})
        if not isinstance(result, dict):
            continue
        rows = result.get("results", [])
        valid_rows = [
            row for row in rows
            if isinstance(row, dict) and isinstance(row.get("passed"), bool)
        ] if isinstance(rows, list) else []
        if valid_rows:
            passed = sum(row["passed"] for row in valid_rows)
            statements.append(f"Configured {metric_name} tests passed {passed}/{len(valid_rows)}.")
        else:
            total = result.get("tests_total")
            passed = result.get("tests_passed")
            if (
                isinstance(total, int) and not isinstance(total, bool) and total > 0
                and isinstance(passed, int) and not isinstance(passed, bool)
                and 0 <= passed <= total
            ):
                statements.append(f"Configured {metric_name} tests passed {passed}/{total}.")

    findings = report.get("llm_rsi_findings", [])
    findings = [item for item in findings if isinstance(item, dict)] if isinstance(findings, list) else []
    if findings:
        first = findings[0]
        location = f" at line {first['line_start']}" if isinstance(first.get("line_start"), int) else ""
        statements.append(
            f"The validated RSI retained {len(findings)} finding(s); the first is "
            f"{str(first.get('title', 'a source issue'))[:140]}{location}."
        )
    else:
        rsi = report.get("llm_rsi_evaluation", {})
        status = rsi.get("status") if isinstance(rsi, dict) else None
        if status in {"unavailable", "completed_with_fallback", "completed_with_warnings"}:
            statements.append(f"RSI status is {status}; the review was incomplete or used a fallback.")
        else:
            statements.append("The completed RSI retained no source-grounded findings; this alone does not prove correctness.")
    return " ".join(statements)
