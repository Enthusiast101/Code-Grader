"""Render a student-facing report from deterministic analyzer evidence."""
from __future__ import annotations

from typing import Any

from grading_agent.reporting.rationale import compact_rationale, evidence_summary, score_source_label


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


def _render_text_report_impl(report: dict[str, Any]) -> str:
    lines = [
        f"Code Checking Assistant — {report.get('problem', 'C assignment')}",
        f"Submission: {report.get('submission', 'unknown')}",
        f"Rules SHA-256: {report.get('rules_hash', 'unknown')}",
        f"Source SHA-256: {report.get('source_sha256', 'unknown')}",
    ]
    rsi_summary = report.get("llm_rsi_evaluation", {})
    if isinstance(rsi_summary, dict) and rsi_summary:
        lines.extend(["", "EVIDENCE SUMMARY", "-" * 72, evidence_summary(report)])
    generation = report.get("test_generation", {})
    if isinstance(generation, dict) and generation.get("status") not in (None, "not_required"):
        lines.extend(["", "ASSIGNMENT TEST GENERATION", "-" * 72,
                      f"Status: {generation.get('status', 'unknown')}"])
        if generation.get("source"):
            lines.append(f"Suite source: {generation['source']}")
        total = generation.get("total_test_count")
        if isinstance(total, int):
            instructor = generation.get("instructor_test_count", 0)
            edge = generation.get("edge_test_count", 0)
            generated_count = max(0, total - instructor) if isinstance(instructor, int) else total
            lines.append(f"Cases: {total} total ({generated_count} generated, {instructor} instructor); {edge} edge cases")
        generation_events = generation.get("request_events", [])
        if isinstance(generation_events, list) and generation_events:
            lines.append(f"Generation requests: {len(generation_events)}")
            for event in generation_events:
                if isinstance(event, dict) and isinstance(event.get("duration_seconds"), (int, float)):
                    lines.append(
                        f"  {event.get('stage', 'test generation')}: {event['duration_seconds']:.3f} s "
                        f"({event.get('status', 'unknown')}; monotonic {event.get('started_monotonic_seconds', 'n/a')} → {event.get('ended_monotonic_seconds', 'n/a')})"
                    )
        detail = generation.get("error") or generation.get("reason")
        if detail:
            lines.append(f"Reason: {detail}")
    lines.extend(["", "METRIC POINTS", "-" * 72])

    provenance = report.get("rules_provenance", {})
    if isinstance(provenance, dict) and provenance:
        selection = provenance.get("selection", {})
        selection = selection if isinstance(selection, dict) else {}
        lines.extend([
            f"Rules source: {selection.get('selected_from', provenance.get('rules_path', 'unknown'))}",
            f"Rules file: {provenance.get('rules_path', 'unknown')}",
        ])
        if selection.get("editor_was_stale"):
            lines.append("Rules note: stale editor content detected; the current uploaded rules file was used.")
        lines.append("")

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
        max_points = float(metric.get("max_points", result.get("max_points", float(metric["weight"]) * 100.0)))
        score_points = result.get("score_points")
        if not metric.get("grade_contribution", True):
            score_text = "diagnostic only"
        elif isinstance(score_points, (int, float)):
            score_text = f"{float(score_points):.2f}/{max_points:g}"
        elif isinstance(score, (int, float)):
            score_text = f"{float(score) * max_points / 100.0:.2f}/{max_points:g}"
        else:
            score_text = f"N/A (max {max_points:g})"
        lines.append(f"{name:22s} points={score_text:12s} status={status}")
        rationale = compact_rationale(result.get("score_rationale", ""), max_words=60, max_chars=420)
        if rationale:
            lines.append(f"  Assessment: {rationale}")
        lines.append(f"  Score source: {score_source_label(result.get('score_origin'))}")
        lines.append(f"  Evidence class: {result.get('score_evidence_class', 'unavailable')}")
        lines.append(f"  Analyzer status: {result.get('status', 'unknown')}")
        provenance = result.get("score_provenance", {})
        if isinstance(provenance, dict):
            available = provenance.get("available_evidence", [])
            missing = provenance.get("missing_evidence", [])
            if available:
                lines.append("  Evidence used: " + "; ".join(str(item) for item in available))
            if missing:
                lines.append("  Evidence missing: " + "; ".join(str(item) for item in missing))
        if name == "correctness":
            tests_total = result.get("tests_total", 0)
            if tests_total:
                lines.append(f"  Tests: {result.get('tests_passed', 0)}/{tests_total} passed")
            else:
                lines.append("  Tests: not configured; execution correctness was not measured by test cases")
        elif name == "logical":
            tests_total = result.get("tests_total", 0)
            if tests_total:
                lines.append(f"  Edge-case tests: {result.get('tests_passed', 0)}/{tests_total} passed; this score covers the configured logical suite.")
            else:
                lines.append("  Edge-case tests: not configured; logical behavior was not measured by execution")
        elif name == "memory_safety":
            runtime = result.get("runtime_tools", {})
            lines.append(f"  Valgrind status/score: {runtime.get('valgrind_status', runtime.get('status', 'unknown'))} / {runtime.get('valgrind_score', 'N/A')}")
            lines.append(f"  ASan/UBSan status/score: {runtime.get('asan_ubsan_status', 'unknown')} / {runtime.get('asan_ubsan_score', 'N/A')}")
            lines.append(f"  Dynamic verification: {'completed' if runtime.get('valgrind_status') == 'completed' else 'not completed'}")
            if runtime.get("scoring_note"):
                lines.append(f"  Runtime scoring: {runtime['scoring_note']}")
            gcc_audit = result.get("gcc_analyzer", {})
            if isinstance(gcc_audit, dict):
                lines.append(f"  GCC -fanalyzer status/findings: {gcc_audit.get('status', 'unknown')} / {len(gcc_audit.get('findings', []))}")
        elif name == "runtime_efficiency":
            input_source = result.get("test_input_source_metric")
            if input_source:
                lines.append(
                    f"  Profile inputs: reused the configured {input_source} test inputs; expected outputs were not used for timing."
                )
            if result.get("compile_elapsed_ms") is not None:
                lines.append(f"  Compile duration: {result['compile_elapsed_ms']:.3f} ms ({result.get('compile_status', 'unknown')})")
            for label, key in (("Average runtime", "mean_runtime_ms"), ("Median runtime", "median_runtime_ms"), ("Maximum runtime", "max_runtime_ms")):
                if isinstance(result.get(key), (int, float)):
                    lines.append(f"  {label}: {result[key]:.3f} ms")
            if isinstance(result.get("median_per_test_runtime_ms"), (int, float)):
                lines.append(f"  Median of per-test medians: {result['median_per_test_runtime_ms']:.3f} ms")
            if result.get("target_runtime_ms") is not None:
                compliance = "met" if result.get("target_runtime_met") is True else "not met" if result.get("target_runtime_met") is False else "not measured"
                lines.append(f"  Target runtime: {result['target_runtime_ms']:.3f} ms ({compliance}; {result.get('target_runtime_basis', 'configured target')})")
            if result.get("failures"):
                lines.append(f"  Runtime failures/timeouts: {len(result['failures'])}")
            if result.get("measurement_note"):
                lines.append(f"  Measurement note: {result['measurement_note']}")
        elif name == "syntax":
            if result.get("exit_code") is not None:
                lines.append(f"  Compiler exit code: {result['exit_code']}; warnings: {result.get('warning_count', 0)}; errors: {result.get('error_count', 0)}")
            if result.get("compile_elapsed_ms") is not None:
                lines.append(f"  Compiler duration: {result['compile_elapsed_ms']:.3f} ms")
            lines.append(f"  Compiler streams separated: {result.get('compiler_streams_separated', False)}")
            for label, key in (("stdout", "compiler_stdout"), ("stderr", "compiler_stderr")):
                stream = result.get(key)
                if isinstance(stream, str) and stream:
                    lines.append(f"  Compiler {label}: {stream[:1_000]}")
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
                f"  {item['metric']}: {item.get('points_awarded', 0.0):.2f}/{item.get('max_points', 100):g} points"
            )
    else:
        final_score = scoring.get("score")
        lines.append(f"Final grade: {float(final_score):.2f}/100" if isinstance(final_score, (int, float)) else "Final grade: N/A — required grading evidence is incomplete.")
    for warning in scoring.get("warnings", []):
        lines.append(f"  Note: {warning}")

    rsi = report.get("llm_rsi_evaluation", {})
    if isinstance(rsi, dict) and rsi:
        lines.extend(["", "PROGRAM UNDERSTANDING + WHOLE-PROGRAM RSI", "-" * 72, f"Status: {rsi.get('status', 'unavailable')}"])
        if rsi.get("error"):
            lines.append(str(rsi["error"]))
        for iteration in rsi.get("iterations", []) if isinstance(rsi.get("iterations"), list) else []:
            if not isinstance(iteration, dict):
                continue
            if iteration.get("status") in {"completed", "completed_with_fallback"}:
                stage_label = "completed with deterministic fallback" if iteration.get("fallback_used") else "completed"
                lines.append(
                    f"  RSI pass {iteration.get('iteration')}: "
                    f"{stage_label}; "
                    f"{iteration.get('findings_count', 0)} retained finding(s); "
                    f"{iteration.get('overall_assessment', '')}"
                )
                for finding in iteration.get("findings", [])[:12]:
                    if isinstance(finding, dict):
                        lines.append(
                            f"    - {finding.get('metric')}: {finding.get('title')} "
                            f"(line {finding.get('line_start') or 'n/a'})"
                        )
            else:
                lines.append(
                    f"  RSI pass {iteration.get('iteration')}: failed; "
                    f"{iteration.get('error', 'no valid response')}"
                )

    understanding_meta = report.get("program_understanding_metadata", {})
    if not understanding_meta and isinstance(rsi_summary, dict):
        understanding_meta = rsi_summary.get("program_understanding_metadata", {})
    if isinstance(understanding_meta, dict) and understanding_meta:
        lines.extend(["", "PROGRAM UNDERSTANDING VALIDATION", "-" * 72])
        lines.append(f"Status: {understanding_meta.get('status', 'unknown')}; fallback used: {understanding_meta.get('fallback_used', False)}")
        if understanding_meta.get("error"):
            lines.append("Validation error: " + str(understanding_meta["error"]))
        diagnostics = understanding_meta.get("diagnostics", [])
        if isinstance(diagnostics, list):
            lines.extend("Diagnostic: " + str(item)[:500] for item in diagnostics[:4])

    understanding = report.get("program_understanding", {})
    if isinstance(understanding, dict) and understanding.get("program_purpose"):
        lines.extend(["", "PROGRAM UNDERSTANDING", "-" * 72])
        lines.append("Actual source behavior: " + str(understanding.get("actual_behavior", understanding.get("program_purpose", ""))))
        if understanding.get("intended_behavior"):
            lines.append("Expected assignment behavior: " + str(understanding["intended_behavior"]))
        if understanding.get("contract_alignment"):
            lines.append("Contract comparison: " + str(understanding["contract_alignment"]))
        if understanding.get("algorithm_overview"):
            lines.append("Actual algorithm: " + str(understanding["algorithm_overview"]))
        functions = understanding.get("functions", [])
        if functions:
            lines.append("Functions:")
            for function in functions[:20]:
                if isinstance(function, dict):
                    location = ""
                    if function.get("line_start") is not None:
                        location = f" (lines {function.get('line_start')}-{function.get('line_end') or function.get('line_start')})"
                    unresolved = " [source line unresolved]" if function.get("source_line_status", "valid" if location else "unresolved") != "valid" else ""
                    lines.append(f"  - {function.get('name')}{location}{unresolved}: {function.get('role', '')}")
        for label, key in (("Execution flow", "execution_flow"), ("Assumptions", "assumptions"), ("Ambiguities", "ambiguities")):
            values = understanding.get(key, [])
            if values:
                lines.append(label + ":")
                lines.extend(f"  - {value}" for value in values[:8])
    elif isinstance(understanding, dict) and understanding.get("error"):
        lines.extend(["", "PROGRAM UNDERSTANDING", "-" * 72, "Unavailable: " + str(understanding["error"])])

    findings = report.get("llm_rsi_findings", [])
    if isinstance(findings, list) and findings:
        lines.extend(["", "WHOLE-PROGRAM RSI FINDINGS", "-" * 72])
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            location = (
                f"lines {finding.get('line_start')}-{finding.get('line_end') or finding.get('line_start')}"
                if finding.get("line_start") is not None else "source/evidence location not resolved"
            )
            refs = ", ".join(finding.get("evidence_ids", []))
            lines.append(
                f"[{finding.get('severity', 'unknown').upper()}] "
                f"{finding.get('metric', 'unmapped')} — {finding.get('title', 'Issue')} ({location})"
            )
            if finding.get("source_line_status", "valid" if finding.get("line_start") is not None else "unresolved") != "valid":
                lines.append(f"  Source-line anchor: {finding.get('source_line_status')}; review location is unresolved.")
            lines.append("  " + str(finding.get("description", "")))
            if refs:
                lines.append("  Evidence: " + refs)
            if finding.get("code_excerpt"):
                lines.append("  Source:\n" + str(finding["code_excerpt"]))
    else:
        rsi = report.get("llm_rsi_evaluation", {})
        rsi = rsi if isinstance(rsi, dict) else {}
        rsi_status = rsi.get("status", "unavailable")
        if rsi_status == "completed":
            finding_note = "No concrete, supported defects were retained by the successfully completed RSI review sequence."
        elif rsi_status == "completed_with_warnings":
            finding_status = rsi.get("findings_status", "")
            if finding_status == "completed_with_fallback":
                finding_note = (
                    "The LLM did not return a fully validated review for every requested pass. "
                    "The findings below, if any, were retained from deterministic evidence; "
                    "an empty list is not evidence that the source is defect-free."
                )
            else:
                finding_note = (
                    "The RSI sequence completed with warnings. Review the per-pass diagnostics; "
                    "an empty findings list must not be interpreted as proof that no defects exist."
                )
        else:
            finding_note = "RSI findings unavailable: no whole-program review pass returned a valid response. This does not mean the source is defect-free."
        lines.extend(["", "WHOLE-PROGRAM RSI FINDINGS", "-" * 72, finding_note])

    rejected_findings = report.get("llm_rsi_rejected_findings", [])
    if isinstance(rejected_findings, list) and rejected_findings:
        lines.extend(["", "EXCLUDED RSI CANDIDATES (NOT USED FOR SCORING)", "-" * 72])
        for item in rejected_findings[:20]:
            if not isinstance(item, dict):
                continue
            location = f"line {item.get('line_start')}" if item.get("line_start") is not None else "location unresolved"
            lines.append(
                f"[{item.get('metric', 'unmapped')}] {item.get('title', 'Candidate')} ({location}): "
                f"{str(item.get('reason', 'Not retained after validation.'))[:500]}"
            )

    timings = report.get("timings", {})
    if isinstance(timings, dict) and timings:
        lines.extend(["", "EXECUTION TIMINGS", "-" * 72])
        if timings.get("analyzer_pipeline_seconds") is not None:
            lines.append(f"Deterministic analyzer pipeline: {timings['analyzer_pipeline_seconds']:.3f} s")
        if timings.get("llm_server_setup_seconds") is not None:
            lines.append(f"LLM server readiness/startup: {timings['llm_server_setup_seconds']:.3f} s")
        if timings.get("llm_request_seconds") is not None:
            lines.append(f"Reviewer LLM API requests: {timings['llm_request_seconds']:.3f} s")
        if timings.get("test_generation_request_seconds") is not None:
            lines.append(f"Test-generation LLM requests: {timings['test_generation_request_seconds']:.3f} s")
        if timings.get("total_llm_request_seconds") is not None:
            lines.append(f"Total LLM API request time (test generation + review): {timings['total_llm_request_seconds']:.3f} s")
        if timings.get("total_through_scoring_seconds") is not None:
            lines.append(f"Total through scoring: {timings['total_through_scoring_seconds']:.3f} s")
        for metric_name, elapsed in timings.get("analyzer_metric_seconds", {}).items():
            if isinstance(elapsed, (int, float)):
                lines.append(f"  {metric_name}: {elapsed:.3f} s")

    rsi = report.get("llm_rsi_evaluation", {})
    if isinstance(rsi, dict):
        lines.extend(["", "LLM PROGRAM UNDERSTANDING + RSI", "-" * 72])
        lines.append(f"Status: {rsi.get('status', 'unavailable')}")
        llm_metric_items = rsi.get("metrics", [])
        llm_metric_items = [item for item in llm_metric_items if isinstance(item, dict)] if isinstance(llm_metric_items, list) else []
        if llm_metric_items:
            lines.extend(["", "Metric assessments are consolidated into the point totals above."])
        if rsi.get("llm_calls") is not None:
            lines.append(
                f"LLM calls: {rsi.get('llm_calls')} "
                "(program understanding + sequential RSI + post-RSI finding validation + separate metric scoring; "
                "structured requests may use bounded repair retries)"
            )
        latency = rsi.get("latency_seconds", {})
        if isinstance(latency, dict):
            if isinstance(latency.get("program_understanding"), (int, float)):
                lines.append(f"Program-understanding stage elapsed: {latency['program_understanding']:.3f} s")
            request_events = rsi.get("llm_request_events", [])
            if isinstance(request_events, list):
                request_total = 0.0
                lines.append(f"Recorded LLM API requests: {len(request_events)}")
                for event in request_events:
                    if not isinstance(event, dict):
                        continue
                    duration = event.get("duration_seconds")
                    if not isinstance(duration, (int, float)):
                        continue
                    request_total += float(duration)
                    attempt = f" attempt {event['attempt']}" if event.get("attempt") is not None else ""
                    status = f" ({event.get('status')})" if event.get("status") else ""
                    lines.append(
                        f"  {event.get('stage', 'LLM stage')}{attempt}: {duration:.3f} s{status}; "
                        f"monotonic {event.get('started_monotonic_seconds', 'n/a')} → {event.get('ended_monotonic_seconds', 'n/a')}"
                    )
                lines.append(f"Reviewer-stage LLM API request total: {request_total:.3f} s")
            elif isinstance(latency.get("total_llm"), (int, float)):
                lines.append(f"Reviewer-stage LLM API request total: {latency['total_llm']:.3f} s")
            if isinstance(latency.get("context_compression"), (int, float)):
                lines.append(f"Context-compression stage elapsed: {latency['context_compression']:.3f} s (included request time is listed above)")
        if rsi.get("error"):
            lines.append("LLM error: " + str(rsi.get("error"))[:1_000])
        for diagnostic in rsi.get("diagnostics", [])[:4]:
            lines.append("LLM diagnostic: " + str(diagnostic)[:500])
        compression = rsi.get("context_compression", {})
        if isinstance(compression, dict) and compression:
            compression_status = str(compression.get("status", "not_needed"))
            same_model_calls = int(compression.get("same_model_calls", 0) or 0)
            fallbacks = int(compression.get("deterministic_fallbacks", 0) or 0)
            if compression_status == "same_model_used":
                lines.append(f"Context compression: same grading model used ({same_model_calls} request(s)).")
            elif compression_status == "mixed":
                lines.append(f"Context compression: same-model used ({same_model_calls} request(s)); deterministic fallback used {fallbacks} time(s).")
            elif compression_status == "deterministic_fallback":
                lines.append(f"Context compression: deterministic fallback used ({fallbacks} event(s)); model={compression.get('model', 'configured grading model')}.")
            else:
                lines.append("Context compression: enabled; no model compression was needed for this run." if compression.get("enabled") else "Context compression: disabled.")
            events = compression.get("events", [])
            if isinstance(events, list):
                for event in events:
                    if isinstance(event, dict) and event.get("error"):
                        stage = str(event.get("stage", "compression"))
                        reason = str(event.get("error", ""))[:220]
                        lines.append(f"  Compression detail ({stage}): {reason}")
                        break

    lines.extend([
        "",
        "Evaluation method: the selected model first maps the program, then performs sequential whole-program rechecks, and finally assigns rubric scores. The weighted overall grade is calculated from those metric scores.",
        "Interpretation note: automated findings are evidence for review, not proof of intent or universal correctness.",
    ])
    return "\n".join(lines) + "\n"


def render_text_report(report: dict[str, Any]) -> str:
    """Render a readable report; malformed optional fields never break grading."""
    try:
        if not isinstance(report, dict):
            report = {"report_value": str(report)}
        return _render_text_report_impl(report)
    except Exception as exc:  # Last-resort presentation fallback.
        import json
        problem = str(report.get("problem", "C assignment")) if isinstance(report, dict) else "C assignment"
        submission = str(report.get("submission", "unknown")) if isinstance(report, dict) else "unknown"
        try:
            snapshot = json.dumps(report, ensure_ascii=False, indent=2, default=str)[:12000]
        except Exception:
            snapshot = repr(report)[:12000]
        return (
            f"Code Checking Assistant — {problem}\nSubmission: {submission}\n\n"
            "REPORT FORMATTING FALLBACK\n"
            f"The structured report was saved, but one optional field could not be formatted: {type(exc).__name__}: {exc}\n\n"
            "Raw report snapshot (truncated):\n" + snapshot + "\n"
        )
