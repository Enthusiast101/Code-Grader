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
        if name == "correctness":
            lines.append(f"  Tests: {result.get('tests_passed', 0)}/{result.get('tests_total', 0)} passed")
        elif name == "memory_safety":
            composite = result.get("composite_score", {})
            runtime_component = composite.get("runtime_tools", result.get("runtime_tools", {}))
            audit_component = composite.get("llm_source_audit", {})
            audit_record = result.get("llm_source_audit", {})
            runtime_score = runtime_component.get("score")
            audit_score = audit_component.get("score")
            lines.append(
                "  Runtime tools: "
                + (f"{float(runtime_score):.2f}/100" if isinstance(runtime_score, (int, float)) else "N/A")
            )
            audit_status = audit_record.get("status", audit_component.get("status", "unknown"))
            lines.append(
                "  LLM source audit: "
                + (f"{float(audit_score):.2f}/100" if isinstance(audit_score, (int, float)) else f"N/A ({audit_status})")
            )
            audit_error = audit_record.get("error") or audit_component.get("reason")
            if audit_error:
                lines.append(f"  Source-audit detail: {audit_error}")
            if result.get("reason"):
                lines.append(f"  Memory-safety detail: {result['reason']}")
            audit_payload = audit_record.get("audit", {})
            if isinstance(audit_payload, dict):
                verified_findings = audit_payload.get("findings", [])
                rejected_findings = audit_payload.get("rejected_findings", [])
                for finding in verified_findings[:5] if isinstance(verified_findings, list) else []:
                    if isinstance(finding, dict):
                        lines.append(
                            f"  Verified source finding: {finding.get('severity', 'unknown')} "
                            f"{finding.get('category', 'unknown')} at line {finding.get('line', '?')}: "
                            f"{finding.get('description', '')}"
                        )
                for finding in rejected_findings[:5] if isinstance(rejected_findings, list) else []:
                    if isinstance(finding, dict):
                        lines.append(
                            f"  Unscored source-audit claim: {finding.get('category', 'unknown')} "
                            f"at line {finding.get('line', '?')}: {finding.get('reason', 'Not corroborated.')}"
                        )
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
        lines.append("Final grade: PENDING / INCOMPLETE")
        for item in scoring.get("incomplete_metrics", []):
            lines.append(f"  {item['metric']}: {item['reason']}")

    review = report.get("llm_review", {})
    if isinstance(review, dict) and review:
        body = review.get("review", {})
        body = body if isinstance(body, dict) else {}
        lines.extend([
            "", "EVIDENCE REVIEW", "-" * 72,
            f"Status: {review.get('status', 'unknown')}",
            f"Review engine: {review.get('review_engine', 'unknown')}",
            f"Local LLM status: {review.get('llm_status', 'not_recorded')}",
            str(body.get("summary") or review.get("error") or "No review summary was returned."),
        ])
        if review.get("llm_error"):
            lines.append(f"  LLM diagnostic: {review['llm_error']}")
        for section, heading in (
            ("findings", "Findings"),
            ("contradictions", "Contradictions"),
            ("recommendations", "Recommendations"),
            ("uncertainties", "Uncertainties"),
        ):
            items = body.get(section, [])
            if isinstance(items, list) and items:
                lines.append(f"  {heading}:")
                lines.extend(f"    - {_as_text(item)}" for item in items[:12])
        validation = review.get("claim_validation", {})
        if isinstance(validation, dict) and validation:
            lines.append(
                "  Deterministic evidence validation: "
                f"{validation.get('verified_claim_count', 0)} verified, "
                f"{validation.get('rejected_claim_count', 0)} rejected."
            )
        advisory = review.get("llm_advisory")
        if isinstance(advisory, dict):
            lines.append("  LLM ADVISORY (non-scoring):")
            lines.append(f"    {advisory.get('summary', 'No advisory summary.')}")
            for section, heading in (("findings", "Findings"), ("contradictions", "Contradictions"), ("recommendations", "Recommendations"), ("uncertainties", "Uncertainties")):
                entries = advisory.get(section, [])
                if isinstance(entries, list) and entries:
                    lines.append(f"    {heading}:")
                    lines.extend(f"      - {_as_text(item)}" for item in entries[:5])
        else:
            lines.append("  LLM advisory: unavailable; deterministic evidence review is complete.")
        llm_validation = review.get("llm_claim_validation", {})
        if isinstance(advisory, dict) and isinstance(llm_validation, dict) and llm_validation:
            if "verified_claim_count" in llm_validation or "rejected_claim_count" in llm_validation:
                lines.append(
                    "  LLM advisory validation: "
                    f"{llm_validation.get('verified_claim_count', 0)} verified, "
                    f"{llm_validation.get('rejected_claim_count', 0)} rejected."
                )
            else:
                lines.append(
                    "  LLM advisory validation status: "
                    f"{llm_validation.get('status', 'not completed')}; "
                    f"{llm_validation.get('reason', 'No valid advisory was accepted.')}"
                )
        advisory_claims = advisory.get("unverified_claims", []) if isinstance(advisory, dict) else []
        rejected_claims = body.get("unverified_claims", [])
        if isinstance(advisory_claims, list) and advisory_claims:
            rejected_claims = list(rejected_claims) + advisory_claims
        if isinstance(rejected_claims, list) and rejected_claims:
            lines.append("  Rejected / unverified claims:")
            for claim in rejected_claims[:8]:
                if isinstance(claim, dict):
                    lines.append(f"    - {claim.get('claim', '(claim omitted)')}")
                    lines.append(f"      Reason: {claim.get('reason', 'Evidence validation failed.')}")
        if review.get("final_synthesis_status"):
            lines.append(f"  Final synthesis: {review['final_synthesis_status']}")
        for failure in review.get("failed_passes", []) if isinstance(review.get("failed_passes"), list) else []:
            if isinstance(failure, dict):
                lines.append(
                    f"  Review pass {failure.get('iteration', '?')} error: "
                    f"{failure.get('error', 'Invalid model response.')}"
                )
        recovered = review.get("recovered_passes", [])
        if isinstance(recovered, list) and recovered:
            iterations = ", ".join(str(item.get("iteration", "?")) for item in recovered if isinstance(item, dict))
            lines.append(
                f"  Recovered intermediate LLM pass(es): {iterations}; "
                "a later valid final advisory superseded these attempts."
            )

    lines.extend([
        "",
        "Interpretation note: automated findings are evidence for review, not proof of intent or universal correctness.",
    ])
    return "\n".join(lines) + "\n"
