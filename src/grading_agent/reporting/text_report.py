"""Render a compact human-readable grading report from the JSON result."""
from __future__ import annotations
from typing import Any


def render_text_report(report: dict[str, Any]) -> str:
    lines = [
        f"Code Checking Assistant — {report.get('problem', 'C assignment')}",
        f"Submission: {report.get('submission', 'unknown')}",
        f"Rules SHA-256: {report.get('rules_hash', 'unknown')}",
        "",
        "METRIC RESULTS",
        "-" * 72,
    ]
    preflight = report.get("preflight", {})
    if preflight:
        lines.extend([
            "PREFLIGHT",
            "-" * 72,
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
        if name == "correctness":
            lines.append(f"  Tests: {result.get('tests_passed', 0)}/{result.get('tests_total', 0)} passed")
        elif name == "memory_safety":
            composite = result.get("composite_score", {})
            lines.append(f"  Runtime tools: {composite.get('runtime_tools', {}).get('score', 'N/A')}")
            lines.append(f"  LLM source audit: {composite.get('llm_source_audit', {}).get('score', 'N/A')}")
        elif name == "originality":
            lines.append(f"  Similarity flagged for review: {result.get('flagged_for_human_review', 'N/A')}")

    scoring = report.get("scoring", {})
    lines.extend(["", "OVERALL SCORE", "-" * 72])
    if scoring.get("status") == "completed":
        lines.append(f"Final grade: {scoring.get('score'):.2f}/100")
        for item in scoring.get("contributions", []):
            lines.append(
                f"  {item['metric']}: {item['score']:.2f} × {item['weight']:.3f} = {item['contribution']:.2f}"
            )
    else:
        lines.append("Final grade: PENDING / INCOMPLETE")
        for item in scoring.get("incomplete_metrics", []):
            lines.append(f"  {item['metric']}: {item['reason']}")

    review = report.get("llm_review", {})
    if isinstance(review, dict) and review:
        body = review.get("review", {})
        body = body if isinstance(body, dict) else {}
        review_status = review.get("status", "unknown")
        lines.extend([
            "",
            "LLM REVIEW",
            "-" * 72,
            f"Status: {review_status}",
            str(body.get("summary") or review.get("error") or "No valid LLM summary was returned."),
        ])
        for section, heading in (
            ("findings", "Findings"),
            ("contradictions", "Contradictions"),
            ("recommendations", "Recommendations"),
            ("uncertainties", "Uncertainties"),
        ):
            items = body.get(section, [])
            if isinstance(items, list) and items:
                lines.append(f"  {heading}:")
                for item in items[:5]:
                    lines.append(f"    - {item}")
        if review.get("final_synthesis_status"):
            lines.append(f"  Final synthesis: {review['final_synthesis_status']}")
        for failure in review.get("failed_passes", []) if isinstance(review.get("failed_passes"), list) else []:
            if isinstance(failure, dict):
                lines.append(
                    f"  Review pass {failure.get('iteration', '?')} error: "
                    f"{failure.get('error', 'Invalid model response.')}"
                )

    lines.extend([
        "",
        "Interpretation note: automated findings are evidence for review, not proof of intent or universal correctness.",
    ])
    return "\n".join(lines) + "\n"
