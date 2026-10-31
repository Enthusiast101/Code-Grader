"""Render a student-facing report from deterministic analyzer evidence."""
from __future__ import annotations

from typing import Any


def _as_text(item: Any) -> str:
    if isinstance(item, dict):
        text = str(item.get("text", "")).strip()
        refs = item.get("evidence_ids", [])
        if refs:
            text += " [" + ", ".join(str(ref) for ref in refs[:5]) + "]"
        if item.get("basis") == "optional":
            text = "Optional: " + text
        return text
    return str(item)


def _format_test_failure(metric_name: str, test: dict[str, Any]) -> list[str]:
    lines = [f"  {metric_name} test {test.get('test_id', '?')} failed:"]
    description = test.get("description")
    if description:
        lines.append(f"    Case:     {description}")
    lines.extend([
        f"    Input:    {test.get('input', '')!r}",
        f"    Expected: {test.get('expected', '')!r}",
        f"    Actual:   {test.get('actual', test.get('output', ''))!r}",
    ])
    if test.get("timed_out"):
        lines.append("    Cause:    execution timed out.")
    elif test.get("exit_code") not in (0, None):
        lines.append(f"    Cause:    process exited with code {test.get('exit_code')}.")
    elif test.get("output_may_be_truncated"):
        lines.append("    Cause:    captured output was truncated; result is not a confirmed pass.")
    else:
        lines.append("    Cause:    actual output differs from expected output.")
    tags = test.get("tags")
    if isinstance(tags, list) and tags:
        lines.append("    Tags:     " + ", ".join(str(tag) for tag in tags))
    return lines


def render_text_report(report: dict[str, Any]) -> str:
    lines = [
        f"Code Checking Assistant — {report.get('problem', 'C assignment')}",
        f"Submission: {report.get('submission', 'unknown')}",
        f"Rules SHA-256: {report.get('rules_hash', 'unknown')}",
        f"Source SHA-256: {report.get('source_sha256', 'unknown')}",
        "",
        "METRIC RESULTS",
        "-" * 72,
    ]

    preflight = report.get("preflight", {})
    if isinstance(preflight, dict) and preflight:
        lines.extend([
            "PREFLIGHT", "-" * 72,
            f"Status: {preflight.get('status', 'unknown')}",
            f"Compiled: {preflight.get('compiled', 'unknown')}",
            f"Smoke tested: {preflight.get('smoke_tested', 'unknown')}",
        ])
        if preflight.get("suspected_infinite_loop"):
            lines.append("Warning: smoke test timed out; suspected non-termination.")
        lines.append("")

    config = report.get("configuration", {})
    for metric in config.get("enabled_metrics", []):
        name = metric["name"]
        result = report.get(name, {})
        score = result.get("score")
        status = result.get("status", "completed" if score is not None else "inconclusive")
        score_text = f"{float(score):.2f}/100" if isinstance(score, (int, float)) else "N/A"
        lines.append(f"{name:22s} weight={metric['weight']:.3f}  score={score_text:12s} status={status}")
        if result.get("scoring_basis"):
            lines.append(
                f"  Analyzer={result.get('analyzer_score', 'N/A')}  "
                f"LLM RSI={result.get('llm_rsi_score', 'N/A')}  "
                f"Basis={result.get('scoring_basis', 'unknown')}"
            )
        rsi_detail = result.get("llm_rsi", {})
        if isinstance(rsi_detail, dict) and rsi_detail.get("rationale"):
            lines.append(f"  RSI: {rsi_detail['rationale']}")
        if name == "correctness":
            lines.append(f"  Tests: {result.get('tests_passed', 0)}/{result.get('tests_total', 0)} passed")
        elif name == "memory_safety":
            runtime = result.get("runtime_tools", {})
            lines.append(f"  Valgrind score: {runtime.get('valgrind_score', 'N/A')}")
            lines.append(f"  ASan/UBSan score (supplementary): {runtime.get('asan_ubsan_score', 'N/A')}")
        elif name == "originality":
            lines.append(f"  Similarity flagged for review: {result.get('flagged_for_human_review', 'N/A')}")

    failures: list[str] = []
    for metric_name in ("correctness", "logical"):
        metric = report.get(metric_name, {})
        for test in metric.get("results", []) if isinstance(metric, dict) else []:
            if isinstance(test, dict) and test.get("passed") is False:
                failures.extend(_format_test_failure(metric_name, test))
    if failures:
        lines.extend(["", "FAILING INPUTS / COUNTEREXAMPLES", "-" * 72, *failures])

    scoring = report.get("scoring", {})
    lines.extend(["", "OVERALL SCORE", "-" * 72])
    if scoring.get("status") == "completed":
        lines.append(f"Final grade: {scoring.get('score'):.2f}/100")
        for item in scoring.get("contributions", []):
            lines.append(
                f"  {item['metric']}: {item['score']:.2f} × {item['weight']:.3f} = {item['contribution']:.2f}"
            )
    else:
        lines.append(f"Final grade: {float(scoring.get('score', 0.0)):.2f}/100")
    for warning in scoring.get("warnings", []):
        lines.append(f"  Note: {warning}")

    rsi = report.get("llm_rsi_evaluation", {})
    if isinstance(rsi, dict) and rsi:
        lines.extend(["", "LLM EVIDENCE-BASED RSI", "-" * 72, f"Status: {rsi.get('status', 'unavailable')}"])
        if rsi.get("error"):
            lines.append(str(rsi["error"]))
        for item in rsi.get("metrics", []) if isinstance(rsi.get("metrics"), list) else []:
            if isinstance(item, dict):
                lines.append(f"  {item.get('name')}: {item.get('score')}/100 — {item.get('rationale', '')}")
        if rsi.get("missing_metrics"):
            lines.append("LLM scores missing: " + ", ".join(rsi["missing_metrics"]))

    lines.extend([
        "",
        "Interpretation note: automated findings are evidence for review, not proof of intent or universal correctness.",
    ])
    return "\n".join(lines) + "\n"
