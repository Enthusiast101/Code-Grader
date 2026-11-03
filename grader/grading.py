from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .analyzers import analyze_source, originality_similarity, score_structure
from .llm import LLMClient, LLMError
from .rules import load_rules
from .sandbox import DockerSandbox, SandboxError
from .suite import SuiteError, _compare, prepare_suite


class GradingError(RuntimeError):
    pass


def _metric_for(rules: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((item for item in rules["metrics"] if item["name"] == name and item["enabled"] and item["weight"] > 0), None)


def _case_groups(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edge_words = {"edge", "boundary", "adversarial", "corner", "overflow", "negative", "zero", "minimum", "maximum", "large", "limit", "extreme"}
    core, edge = [], []
    for case in cases:
        tags = {str(tag).casefold() for tag in case.get("tags", [])}
        if tags & edge_words:
            edge.append(case)
        else:
            core.append(case)
    return core, edge


def _compile_student(sandbox: DockerSandbox, source: Path) -> dict[str, Any]:
    sandbox.put_text("/work/submission.c", source.read_text(encoding="utf-8"))
    result = sandbox.compile("submission.c", "student", flags=[
        "-std=c11", "-O2", "-pipe", "-fno-ident", "-fdiagnostics-color=never", "-Wall", "-Wextra", "-Wpedantic",
    ], timeout=15)
    if result.get("exit_code") == 0 and not result.get("timed_out"):
        sandbox.lock_binary("student", "student")
    return result


def _run_cases(sandbox: DockerSandbox, program: str, cases: list[dict[str, Any]], rules: dict[str, Any]) -> dict[str, Any]:
    results = []
    passed = 0
    for case in cases:
        run = sandbox.run_program(program, case["input"], timeout_s=rules["time_limit_s"])
        is_pass = (
            not run.get("timed_out")
            and not run.get("output_truncated")
            and run.get("exit_code") == 0
            and _compare(run.get("stdout", ""), case["expected"], rules["output_comparison"])
        )
        passed += int(is_pass)
        reason = None
        if run.get("timed_out"):
            reason = f"Program exceeded the {rules['time_limit_s']:g}s time limit."
        elif run.get("output_truncated"):
            reason = "Program output exceeded the configured capture limit."
        elif run.get("exit_code") != 0:
            reason = f"Program exited with code {run.get('exit_code')}."
        elif not is_pass:
            reason = "Program output differs from the reference output."
        results.append({
            "id": case["id"], "tags": case.get("tags", []), "description": case.get("description", ""),
            "passed": is_pass, "input": case["input"], "expected": case["expected"],
            "actual": run.get("stdout", ""), "stderr": run.get("stderr", ""),
            "exit_code": run.get("exit_code"), "timed_out": run.get("timed_out", False),
            "output_truncated": run.get("output_truncated", False), "failure_reason": reason,
        })
    score = 100.0 * passed / len(cases) if cases else None
    return {"status": "completed" if cases else "unavailable", "score": round(score, 2) if score is not None else None,
            "tests_passed": passed, "tests_total": len(cases), "results": results,
            "rubric": "100 × passing assigned tests / total assigned tests. Each test is counted once.",
            "rationale": f"Passed {passed} of {len(cases)} assigned test(s)." if cases else "No tests were assigned to this metric."}


def _compile_failure_result(cases: list[dict[str, Any]], *, metric_label: str) -> dict[str, Any]:
    """Record concrete counterexamples even when compilation prevents execution."""
    results = [{
        "id": case["id"],
        "tags": case.get("tags", []),
        "description": case.get("description", ""),
        "passed": False,
        "input": case["input"],
        "expected": case["expected"],
        "actual": "<not executed: compilation failed>",
        "stderr": "",
        "exit_code": None,
        "timed_out": False,
        "output_truncated": False,
        "failure_reason": "Not executed because the C source did not compile; see preflight compiler diagnostics.",
    } for case in cases]
    return {
        "status": "completed" if cases else "unavailable",
        "score": 0.0 if cases else None,
        "tests_passed": 0,
        "tests_total": len(cases),
        "results": results,
        "rubric": f"100 × passing {metric_label} tests / total {metric_label} tests; compilation failure prevents every execution test from passing.",
        "rationale": f"Compilation failed; 0/{len(cases)} {metric_label} test(s) could execute." if cases else f"No {metric_label} tests were assigned.",
    }


def _syntax_result(compile_result: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    diagnostics = (compile_result.get("stderr", "") + "\n" + compile_result.get("stdout", "")).strip()
    warning_lines = [line.strip() for line in diagnostics.splitlines() if "warning:" in line]
    error_lines = [line.strip() for line in diagnostics.splitlines() if "error:" in line]
    if compile_result.get("timed_out") or compile_result.get("exit_code") != 0:
        score = 0.0
        rationale = "The source did not compile under the configured C11 compiler flags."
    else:
        penalty = float(params.get("warning_penalty", 5))
        score = max(0.0, 100.0 - penalty * len(warning_lines))
        rationale = "Compiled without warnings." if not warning_lines else f"Compiled with {len(warning_lines)} warning(s); {penalty:g} points deducted per warning."
    return {"status": "completed", "score": round(score, 2), "warning_count": len(warning_lines),
            "warnings": warning_lines, "errors": error_lines, "compiler_output": diagnostics[:8000],
            "rubric": f"Compile failure = 0; otherwise 100 minus {float(params.get('warning_penalty', 5)):g} per compiler warning, floor 0.",
            "rationale": rationale}


def _memory_safety(sandbox: DockerSandbox, cases: list[dict[str, Any]], rules: dict[str, Any], enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"status": "disabled", "score": None, "rationale": "Sanitizer execution disabled in rules."}
    flags = ["-std=c11", "-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
    compiled = sandbox.compile("submission.c", "student_san", flags=flags, timeout=20)
    if compiled.get("timed_out") or compiled.get("exit_code") != 0:
        return {"status": "unavailable", "score": None, "compile_output": (compiled.get("stderr", "") + compiled.get("stdout", ""))[-3000:],
                "rationale": "The compiler could not build the sanitizer binary; memory safety was not measured."}
    sandbox.lock_binary("student_san", "student_san")
    results = []
    clean = 0
    diagnostic_pattern = re.compile(r"AddressSanitizer|UndefinedBehaviorSanitizer|runtime error:|LeakSanitizer", re.IGNORECASE)
    for case in cases:
        run = sandbox.run_program("student_san", case["input"], timeout_s=rules["time_limit_s"])
        diagnostic = diagnostic_pattern.search((run.get("stderr", "") + "\n" + run.get("stdout", "")))
        exit_code = run.get("exit_code")
        signalled = isinstance(exit_code, int) and exit_code >= 128
        passed = not run.get("timed_out") and not run.get("output_truncated") and not signalled and diagnostic is None
        clean += int(passed)
        results.append({"id": case["id"], "clean": passed, "exit_code": run.get("exit_code"),
                        "timed_out": run.get("timed_out"), "diagnostic": diagnostic.group(0) if diagnostic else None,
                        "stderr": run.get("stderr", "")[:2000]})
    score = 100.0 * clean / len(cases) if cases else None
    return {"status": "completed" if cases else "unavailable", "score": round(score, 2) if score is not None else None,
            "tests_clean": clean, "tests_total": len(cases), "results": results,
            "rubric": "100 × sanitizer-clean test runs / total sanitizer test runs; crashes, timeouts and sanitizer reports fail a run.",
            "rationale": f"ASan/UBSan clean on {clean}/{len(cases)} test run(s)." if cases else "No tests available for sanitizer runs."}


def _modularity_result(metrics: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    minimum = int(params.get("min_functions", 1))
    helpers_required = bool(params.get("expect_helpers", False))
    functions = metrics["function_count"]
    helpers = max(0, functions - (1 if metrics["has_main"] else 0))
    if not helpers_required:
        score = 100.0
        rationale = "The assignment does not require helper functions; a single main() is accepted."
    else:
        minimum_helpers = max(1, int(params.get("min_helpers", minimum - 1)))
        score = min(100.0, 100.0 * helpers / minimum_helpers)
        rationale = f"Found {helpers} helper function(s); {minimum_helpers} required by the instructor rubric."
    return {"status": "completed", "score": round(score, 2), "function_count": functions, "helper_count": helpers,
            "helpers_required": helpers_required,
            "rubric": "100 when no helper decomposition is required; otherwise 100 × helper functions / required helper functions, capped at 100.",
            "rationale": rationale}


def _normalize_finding(item: Any, source_lines: list[str]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    title, description, excerpt = item.get("title"), item.get("description"), item.get("evidence_excerpt", item.get("code_excerpt", ""))
    line = item.get("line_start", item.get("line"))
    if not isinstance(title, str) or not title.strip() or not isinstance(description, str) or not description.strip():
        return None
    if isinstance(line, bool) or not isinstance(line, int) or not 1 <= line <= len(source_lines):
        return None
    if not isinstance(excerpt, str) or not excerpt.strip():
        return None
    # Findings without an exact source anchor are not retained. This does not
    # prove the finding is right; it prevents invented line references.
    normalized_excerpt = " ".join(excerpt.split())
    local = " ".join(" ".join(source_lines[max(0, line - 2):min(len(source_lines), line + 1)]).split())
    if normalized_excerpt not in local:
        return None
    severity = str(item.get("severity", "medium")).lower()
    if severity not in {"critical", "high", "medium", "low", "info"}:
        severity = "medium"
    metric = str(item.get("metric", "logic"))
    return {"title": title.strip()[:160], "description": description.strip()[:700], "severity": severity,
            "metric": metric, "line_start": line, "evidence_excerpt": normalized_excerpt[:400],
            "verification": "LLM advisory; not included in numeric score unless a deterministic test/analyzer independently confirms it."}


def _compact_rsi_evidence(analyzer_results: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for name, result in analyzer_results.items():
        if not isinstance(result, dict):
            continue
        item = {key: result[key] for key in ("status", "score", "weight", "rubric", "rationale", "tests_passed", "tests_total", "warning_count", "warnings", "findings") if key in result}
        failures = [test for test in result.get("results", []) if isinstance(test, dict) and not test.get("passed", test.get("clean", True))]
        if failures:
            item["failures"] = [{
                "id": test.get("id"),
                "input": str(test.get("input", ""))[:1200],
                "expected": str(test.get("expected", ""))[:500],
                "actual": str(test.get("actual", ""))[:500],
                "failure_reason": test.get("failure_reason", test.get("diagnostic", "runtime/sanitizer failure")),
                "stderr": str(test.get("stderr", ""))[:500],
            } for test in failures[:6]]
        compact[name] = item
    return compact


def _numbered_source_excerpt(source: str, max_chars: int = 12000) -> tuple[str, bool]:
    """Preserve original source line numbers even when a long file is clipped."""
    lines = source.splitlines()
    total = sum(len(line) + 1 for line in lines)
    if total <= max_chars:
        return "\n".join(f"{index:05d}: {line}" for index, line in enumerate(lines, 1)), False

    head_budget = max_chars // 2 - 500
    tail_budget = max_chars - head_budget - 500
    head: list[tuple[int, str]] = []
    used = 0
    for index, line in enumerate(lines, 1):
        snippet = line if len(line) <= 3000 else line[:1500] + " ... [line clipped] ... " + line[-1500:]
        if used + len(snippet) + 8 > head_budget:
            break
        head.append((index, snippet))
        used += len(snippet) + 8

    tail: list[tuple[int, str]] = []
    used = 0
    for index in range(len(lines), 0, -1):
        line = lines[index - 1]
        snippet = line if len(line) <= 3000 else line[:1500] + " ... [line clipped] ... " + line[-1500:]
        if used + len(snippet) + 8 > tail_budget or (head and index <= head[-1][0]):
            break
        tail.append((index, snippet))
        used += len(snippet) + 8
    tail.reverse()

    output = [f"{index:05d}: {line}" for index, line in head]
    if head and tail and head[-1][0] + 1 < tail[0][0]:
        output.append(f"... [original source lines {head[-1][0] + 1}-{tail[0][0] - 1} omitted] ...")
    output.extend(f"{index:05d}: {line}" for index, line in tail)
    return "\n".join(output), True



def _semantic_compress_source(llm: LLMClient, rules: dict[str, Any], source: str,
                              *, chunk_chars: int = 5200) -> dict[str, Any]:
    """Compress every source chunk into a validated, line-anchored semantic map.

    The compressor never replaces the source of truth: exact line anchors are
    checked against the original file, and RSI also receives deterministic
    source excerpts plus analyzer evidence. If compression fails, the caller
    can fall back to bounded source context without losing the grade.
    """
    lines = source.splitlines()
    chunks: list[tuple[int, int, str]] = []
    if not lines:
        chunks.append((1, 0, ""))
    else:
        start = 1
        while start <= len(lines):
            end = start
            used = 0
            while end <= len(lines):
                line = lines[end - 1]
                cost = len(line) + 1
                if end > start and used + cost > chunk_chars:
                    break
                used += cost
                end += 1
            chunks.append((start, end - 1, "\\n".join(lines[start - 1:end - 1])))
            start = end

    system = (
        "You are a semantic code summarizer for a C program grader. Summarize the supplied source chunk "
        "without losing behaviorally important information. Return JSON only with keys: functions "
        "(array of {name, lines, purpose, inputs, outputs, calls}), control_flow (array of {lines, construct, "
        "behavior, bounds}), data_flow (array of {lines, variable, role, type_or_inferred_type}), "
        "io_behavior (array of {lines, operation, format_or_semantics}), memory_behavior (array of {lines, "
        "operation, risk}), invariants (array of strings), and unresolved (array of strings). "
        "Every lines value must be an array of original integer line numbers within this chunk. Do not "
        "invent source lines, types, constraints, or behavior. Use empty arrays when absent. Be concise."
    )
    maps = []
    failures = []
    for index, (first, last, text_chunk) in enumerate(chunks, 1):
        user = json.dumps({
            "assignment": rules["problem"],
            "original_line_range": [first, last],
            "source_chunk_with_original_line_numbers": "\n".join(
                f"{n:05d}: {line}" for n, line in enumerate(text_chunk.splitlines(), first)
            ),
            "instruction": "Create a semantic map of this chunk. Preserve all functions, branches, loops, I/O, arithmetic, data structures, and memory operations."
        }, ensure_ascii=False, sort_keys=True)
        try:
            value = llm.json(system, user, max_tokens=min(1800, llm.max_tokens))
            if not isinstance(value, dict):
                raise LLMError("semantic compressor response is not an object")
            clean = {}
            for key in ("functions", "control_flow", "data_flow", "io_behavior", "memory_behavior"):
                arr = value.get(key, [])
                if not isinstance(arr, list):
                    raise LLMError(f"semantic compressor field {key} is not an array")
                clean_items = []
                for item in arr[:80]:
                    if not isinstance(item, dict):
                        continue
                    refs = item.get("lines", [])
                    if not isinstance(refs, list) or not refs or any(
                        isinstance(n, bool) or not isinstance(n, int) or not first <= n <= last for n in refs
                    ):
                        continue
                    clean_items.append({**item, "lines": sorted(set(refs))})
                clean[key] = clean_items
            for key in ("invariants", "unresolved"):
                arr = value.get(key, [])
                clean[key] = [str(x)[:400] for x in arr[:30] if isinstance(x, str)] if isinstance(arr, list) else []
            maps.append({"chunk": index, "line_range": [first, last], **clean})
        except (LLMError, TypeError, ValueError) as exc:
            failures.append({"chunk": index, "line_range": [first, last], "error": str(exc)[:500]})

    # Hierarchical semantic reduction keeps very large programs within the
    # RSI context budget. Each merge may cite only line anchors retained by
    # its child maps; the original source remains the final verification base.
    merge_levels = 0
    merge_system = (
        "Merge semantic maps of adjacent C source chunks into one concise semantic map. "
        "Preserve all behaviorally important functions, branches, loops, I/O, data flow, memory behavior, "
        "invariants and unresolved questions. Return the same JSON schema as the child maps. "
        "Every lines entry must be selected from the supplied child line references; never invent line numbers. "
        "Do not add claims unsupported by the child maps."
    )
    categories = ("functions", "control_flow", "data_flow", "io_behavior", "memory_behavior")
    while len(maps) > 4:
        next_level = []
        for group_start in range(0, len(maps), 4):
            group = maps[group_start:group_start + 4]
            allowed_lines = set()
            covered_ranges = [
                source_range
                for child in group
                for source_range in child.get("covered_source_ranges", [child["line_range"]])
            ]
            for child in group:
                for category in categories:
                    for item in child.get(category, []):
                        allowed_lines.update(item.get("lines", []))
            if not allowed_lines:
                next_level.append({
                    "chunk": f"merge_{merge_levels+1}_{group_start//4+1}",
                    "line_range": [group[0]["line_range"][0], group[-1]["line_range"][1]],
                    "covered_source_ranges": covered_ranges,
                    **{category: [] for category in categories},
                    "invariants": [x for child in group for x in child.get("invariants", [])][:20],
                    "unresolved": [x for child in group for x in child.get("unresolved", [])][:20],
                })
                continue
            payload = {
                "child_semantic_maps": group,
                "allowed_original_line_references": sorted(allowed_lines),
                "instruction": "Merge these adjacent chunk summaries; retain exact original source line references."
            }
            try:
                value = llm.json(merge_system, json.dumps(payload, ensure_ascii=False, sort_keys=True),
                                 max_tokens=min(1800, llm.max_tokens))
                if not isinstance(value, dict):
                    raise LLMError("semantic merge response is not an object")
                merged = {}
                for category in categories:
                    raw_items = value.get(category, [])
                    if not isinstance(raw_items, list):
                        raise LLMError(f"semantic merge field {category} is not an array")
                    clean_items = []
                    for item in raw_items[:24]:
                        if not isinstance(item, dict):
                            continue
                        refs = item.get("lines", [])
                        if not isinstance(refs, list) or not refs or any(
                            isinstance(n, bool) or not isinstance(n, int) or n not in allowed_lines for n in refs
                        ):
                            continue
                        clean_item = {"lines": sorted(set(refs[:16]))}
                        for key, val in item.items():
                            if key == "lines":
                                continue
                            if isinstance(val, str):
                                clean_item[key] = val[:280]
                            elif isinstance(val, (int, float, bool)) or val is None:
                                clean_item[key] = val
                        clean_items.append(clean_item)
                    merged[category] = clean_items
                for key in ("invariants", "unresolved"):
                    raw_values = value.get(key, [])
                    merged[key] = [str(x)[:280] for x in raw_values[:16] if isinstance(x, str)] if isinstance(raw_values, list) else []
                next_level.append({
                    "chunk": f"merge_{merge_levels+1}_{group_start//4+1}",
                    "line_range": [group[0]["line_range"][0], group[-1]["line_range"][1]],
                    "covered_source_ranges": covered_ranges,
                    **merged,
                })
            except (LLMError, TypeError, ValueError) as exc:
                failures.append({
                    "chunk": f"merge_{merge_levels+1}_{group_start//4+1}",
                    "line_range": [group[0]["line_range"][0], group[-1]["line_range"][1]],
                    "error": f"hierarchical semantic merge failed: {str(exc)[:400]}",
                })
                # Keep child maps rather than silently dropping source meaning.
                next_level.extend(group)
        if len(next_level) >= len(maps):
            break
        maps = next_level
        merge_levels += 1

    # Exact coverage accounting: source chunks cover every original line;
    # merge failures retain child maps and are explicitly reported.
    return {
        "method": "hierarchical_llm_semantic_compression_with_line_anchor_validation",
        "source_lines": len(lines),
        "chunk_count": len(chunks),
        "merge_levels": merge_levels,
        "compressed_chunks": maps,
        "failed_chunks": failures,
        "coverage_complete": not failures and (
            sorted([r for item in maps for r in item.get("covered_source_ranges", [item.get("line_range")])]) ==
            sorted([[first, last] for first, last, _ in chunks])
        ),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "validation": "Every retained semantic claim points to original line numbers inside its source chunk.",
    }

def _normalise_complexity(value: Any, source_lines: list[str]) -> dict[str, Any] | None:
    """Validate an LLM complexity explanation; never treat it as a measured score."""
    if not isinstance(value, dict):
        return None
    time_c = value.get("time_complexity")
    space_c = value.get("space_complexity")
    algorithm = value.get("algorithm")
    explanation = value.get("explanation")
    evidence = value.get("evidence", [])
    if not all(isinstance(x, str) and x.strip() for x in (time_c, space_c, algorithm, explanation)):
        return None
    if not re.fullmatch(r"O\s*\([^\n]{1,80}\)", time_c.strip(), re.I):
        return None
    if not re.fullmatch(r"O\s*\([^\n]{1,80}\)", space_c.strip(), re.I):
        return None
    if not isinstance(evidence, list):
        return None
    checked = []
    for item in evidence[:12]:
        if not isinstance(item, dict):
            continue
        line = item.get("line")
        claim = item.get("claim")
        if isinstance(line, int) and not isinstance(line, bool) and 1 <= line <= len(source_lines) and isinstance(claim, str) and claim.strip():
            checked.append({"line": line, "claim": claim.strip()[:300], "source": source_lines[line - 1].strip()[:300]})
    if not checked:
        return None
    return {
        "algorithm": algorithm.strip()[:500],
        "time_complexity": time_c.strip(),
        "space_complexity": space_c.strip(),
        "explanation": explanation.strip()[:1800],
        "evidence": checked,
        "confidence": str(value.get("confidence", "medium")).lower() if str(value.get("confidence", "medium")).lower() in {"low", "medium", "high"} else "medium",
        "status": "llm_advisory_validated_source_lines",
        "numeric_grade_effect": "none; complexity is explanatory unless a separate deterministic rubric is configured",
    }


def run_rsi_review(llm: LLMClient, rules: dict[str, Any], source: str, analyzer_results: dict[str, Any]) -> dict[str, Any]:
    source_lines = source.splitlines()
    numbered_source, source_truncated = _numbered_source_excerpt(source)
    compact_evidence = _compact_rsi_evidence(analyzer_results)
    semantic_context = _semantic_compress_source(llm, rules, source)
    compression = {
        **{key: value for key, value in semantic_context.items() if key != "compressed_chunks"},
        "source_characters": len(source),
        "review_context_characters": len(numbered_source),
        "source_excerpt_truncated": source_truncated,
        "analyzer_evidence_metrics": len(compact_evidence),
        "failure_examples_cap_per_metric": 6,
        "semantic_map_characters": len(json.dumps(semantic_context.get("compressed_chunks", []), ensure_ascii=False)),
        "note": "Semantic summaries are line-anchored and validated; exact source excerpts and deterministic analyzer evidence remain available to RSI.",
    }
    if not semantic_context.get("coverage_complete"):
        warning = "Semantic compression did not cover the complete source; whole-program RSI was withheld rather than reviewing an incomplete or oversized context."
        return {
            "status": "unavailable",
            "iterations": [],
            "assessment": "",
            "findings": [],
            "complexity_analysis": None,
            "context_compression": compression,
            "semantic_context": semantic_context,
            "warnings": [warning],
            "source_truncated_for_review": source_truncated,
            "numeric_score_authority": "deterministic analyzers and frozen I/O tests only",
        }
    system = (
        "You are reviewing a C program against an assignment. Perform sequential iterative review (RSI). "
        "Use only assignment, source, and deterministic analyzer evidence. Re-evaluate previous conclusions, "
        "identify missed defects, and explicitly retract unsupported claims. Cite exact original source lines. "
        "Also analyze algorithmic complexity: identify the dominant algorithm, time Big-O, auxiliary-space Big-O, "
        "and explain how loops/recursion/data structures imply those bounds. Do not confuse input storage with "
        "auxiliary space. If source evidence is insufficient, say so and use low confidence. "
        "Return JSON with assessment, findings, changes_from_previous, and complexity object containing "
        "algorithm, time_complexity, space_complexity, explanation, confidence, and evidence [{line,claim}]. "
        "Do not provide numeric grades or private chain-of-thought; provide concise auditable reasoning only."
    )
    previous: list[dict[str, Any]] = []
    previous_assessment = ""
    previous_complexity: dict[str, Any] | None = None
    iterations: list[dict[str, Any]] = []
    warnings: list[str] = []
    for iteration in range(1, 4):
        user = {
            "pass": iteration,
            "assignment": rules["problem"],
            "instructor_contract": rules["problem_contract"],
            "source_with_original_line_numbers": numbered_source,
            "source_truncated_for_review": source_truncated,
            "semantic_source_map": semantic_context.get("compressed_chunks", []),
            "semantic_compression_coverage_complete": semantic_context.get("coverage_complete", False),
            "semantic_compression_failures": semantic_context.get("failed_chunks", []),
            "deterministic_analyzer_evidence": compact_evidence,
            "previous_assessment": previous_assessment,
            "previous_findings": previous,
            "previous_complexity_analysis": previous_complexity,
            "instruction": (
                "Pass 1: build a complete program understanding and complexity analysis. "
                "Passes 2 and 3: critically recheck the prior assessment, each finding, and each Big-O claim; "
                "trace all relevant loops and data structures again, correct errors, add missed issues, and return "
                "the complete current assessment/findings/complexity. Do not merely repeat prior text."
            ),
        }
        try:
            answer = llm.json(system, json.dumps(user, ensure_ascii=False, sort_keys=True), max_tokens=min(3500, llm.max_tokens))
            raw_findings = answer.get("findings", [])
            if not isinstance(raw_findings, list):
                raise LLMError("RSI findings field is not an array.")
            accepted = []
            for item in raw_findings[:40]:
                finding = _normalize_finding(item, source_lines)
                if finding:
                    accepted.append(finding)
            dedup: dict[tuple[int, str], dict[str, Any]] = {}
            for item in accepted:
                dedup[(item["line_start"], item["title"].casefold())] = item
            previous = [dedup[key] for key in sorted(dedup)]
            assessment = answer.get("assessment", "")
            if not isinstance(assessment, str) or not assessment.strip():
                raise LLMError("RSI assessment is missing or empty.")
            previous_assessment = assessment.strip()[:1600]
            complexity = _normalise_complexity(answer.get("complexity"), source_lines)
            if complexity is not None:
                previous_complexity = complexity
            else:
                warnings.append(f"RSI pass {iteration} did not provide a valid source-anchored complexity analysis; prior valid analysis retained.")
            iterations.append({
                "pass": iteration, "status": "completed", "assessment": previous_assessment,
                "retained_findings": len(previous), "complexity_valid": complexity is not None,
                "complexity": complexity,
                "changes_from_previous": answer.get("changes_from_previous", [])[:12] if isinstance(answer.get("changes_from_previous"), list) else [],
            })
        except (LLMError, TypeError, ValueError) as exc:
            warnings.append(f"Source RSI pass {iteration} did not return valid structured review: {exc}")
            iterations.append({"pass": iteration, "status": "failed_retained_previous", "error": str(exc), "retained_findings": len(previous), "complexity_valid": False})
    status = "completed" if all(item["status"] == "completed" and item.get("complexity_valid") for item in iterations) else "completed_with_warnings" if previous_assessment or previous_complexity else "unavailable"
    return {
        "status": status, "iterations": iterations, "assessment": previous_assessment,
        "findings": previous, "complexity_analysis": previous_complexity,
        "context_compression": compression, "warnings": warnings,
        "source_truncated_for_review": source_truncated,
        "semantic_context": semantic_context,
        "numeric_score_authority": "deterministic analyzers and frozen I/O tests only",
    }


SCORE_RUBRIC = [
    {"range": "0-39", "label": "Fail", "criteria": "Core behavior is absent, compilation fails for a compile-dependent criterion, or most measured tests/checks fail."},
    {"range": "40-59", "label": "Weak", "criteria": "Some basic behavior works, but major requirements or common cases fail; evidence shows substantial defects."},
    {"range": "60-74", "label": "Partial", "criteria": "Most basic behavior is present, but important edge cases, constraints, warnings, or quality requirements remain unmet."},
    {"range": "75-89", "label": "Good", "criteria": "The main requirements are demonstrated and most tests/checks pass; remaining defects are limited and identifiable."},
    {"range": "90-100", "label": "Strong", "criteria": "Nearly all applicable requirements are demonstrated, tests/checks pass, and no material defect is evidenced."},
]


def _score_band(score: float) -> str:
    if score < 40:
        return "0-39: fail — core behavior absent or most evidence fails"
    if score < 60:
        return "40-59: weak — basic behavior partly works; major gaps remain"
    if score < 75:
        return "60-74: partial — core behavior present; important defects remain"
    if score < 90:
        return "75-89: good — most requirements demonstrated; limited defects remain"
    return "90-100: strong — nearly all applicable requirements demonstrated"


def _calculate_grade(metrics: list[dict[str, Any]], results: dict[str, Any]) -> dict[str, Any]:
    contributions = []
    missing = []
    exact_total = 0.0
    for metric in metrics:
        if not metric["enabled"] or metric["weight"] <= 0:
            continue
        result = results.get(metric["name"], {})
        if not isinstance(result, dict):
            result = {}
        score = result.get("score")
        if result.get("status") != "completed" or isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 100.0:
            missing.append({"metric": metric["name"], "status": result.get("status", "missing"), "reason": result.get("rationale", "No complete deterministic measurement."), "weight": metric["weight"]})
            continue
        contribution = float(metric["weight"]) * float(score)
        exact_total += contribution
        contributions.append({"metric": metric["name"], "weight": metric["weight"], "score": round(float(score), 2), "contribution": round(contribution, 2), "score_band": _score_band(float(score))})
    if missing:
        return {"status": "incomplete", "score": None, "scale": "0-100", "contributions": contributions, "missing_metrics": missing,
                "formula": "sum(weight × score); no substitution or redistribution for missing measurements.",
                "score_rubric": SCORE_RUBRIC,
                "reason": "At least one enabled weighted metric was not measured; a numeric overall grade is withheld."}
    return {"status": "completed", "score": round(exact_total, 2), "scale": "0-100",
            "contributions": contributions, "missing_metrics": [],
            "formula": "sum(metric_weight × deterministic_metric_score)",
            "score_rubric": SCORE_RUBRIC,
            "reason": "All enabled weighted metrics were measured; LLM advice did not change any numeric score."}


def grade(rules_path: str | Path, *, source_override: str | Path | None = None, refresh_suite: bool = False) -> tuple[dict[str, Any], Path, Path]:
    rules = load_rules(rules_path, source_override=source_override)
    source_path = rules["source_path"]
    if not source_path.is_file() or source_path.suffix.lower() != ".c":
        raise GradingError(f"C source file not found: {source_path}")
    source = source_path.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    llm = LLMClient(rules["llm"])
    report: dict[str, Any] = {
        "score_rubric": SCORE_RUBRIC,
        "submission": source_path.name,
        "source_sha256": source_hash,
        "rules_sha256": rules["sha256"],
        "problem": rules["problem"],
        "grading_policy": "Numeric grade is computed only from deterministic analyzer evidence and frozen I/O tests. LLM RSI findings are advisory.",
        "metrics": {},
    }
    project_root = rules["root"]
    try:
        with DockerSandbox(rules["docker_image"], rules["memory_limit_mb"], rules["cpu_limit"]) as sandbox:
            suite = prepare_suite(rules, sandbox, llm, refresh=refresh_suite)
            report["test_suite"] = {key: value for key, value in suite.items() if key not in {"reference_source", "test_cases"}}
            report["test_suite"]["test_cases"] = [{k: c[k] for k in ("id", "input", "expected", "tags", "description", "origin")} for c in suite["test_cases"]]
            cases = suite["test_cases"]
            core_cases, edge_cases = _case_groups(cases)
            logical_enabled = _metric_for(rules, "logical") is not None
            correctness_cases = core_cases if logical_enabled else cases

            compile_result = _compile_student(sandbox, source_path)
            compiled = compile_result.get("exit_code") == 0 and not compile_result.get("timed_out")
            report["preflight"] = {
                "compiled": compiled,
                "compiler_version": sandbox.compiler_version(),
                "compile_exit_code": compile_result.get("exit_code"),
                "compile_timed_out": compile_result.get("timed_out"),
                "compiler_output": (compile_result.get("stderr", "") + compile_result.get("stdout", ""))[:8000],
                "sandbox_image": rules["docker_image"],
                "sandbox_image_id": sandbox.image_id,
            }
            source_metrics = analyze_source(source)
            originality_cfg = rules["originality"]
            if originality_cfg["enabled"]:
                originality_result = originality_similarity(
                    source, source_path, originality_cfg["corpus_dir"],
                    threshold=originality_cfg["threshold"], min_tokens=originality_cfg["min_tokens"],
                    max_files=originality_cfg["max_files"], max_file_bytes=originality_cfg["max_file_bytes"],
                )
            else:
                originality_result = {"status": "disabled", "flagged": False, "reason": "Optional corpus similarity check is disabled."}
            report["originality"] = originality_result
            report["flags"] = ([{"type": "possible_code_similarity", "severity": "review",
                                 "matched_file": originality_result.get("matched_file"),
                                 "similarity": originality_result.get("max_similarity"),
                                 "message": originality_result.get("reason", "Review similarity match.")}]
                               if originality_result.get("flagged") else [])

            for metric in rules["metrics"]:
                if not metric["enabled"] or metric["weight"] <= 0:
                    continue
                name, params = metric["name"], metric["params"]
                if name == "syntax":
                    result = _syntax_result(compile_result, params)
                elif name == "correctness":
                    if compiled or not correctness_cases:
                        result = _run_cases(sandbox, "student", correctness_cases, rules) if compiled else {"status": "unavailable", "score": None, "rationale": "No core correctness tests were available for this metric."}
                    else:
                        result = _compile_failure_result(correctness_cases, metric_label="correctness")
                elif name == "logical":
                    if not edge_cases:
                        result = {"status": "unavailable", "score": None, "rationale": "The frozen suite contains no tests tagged edge/boundary/adversarial; this metric was not measured."}
                    elif compiled:
                        result = _run_cases(sandbox, "student", edge_cases, rules)
                    else:
                        result = _compile_failure_result(edge_cases, metric_label="edge")
                elif name == "memory_safety":
                    if compiled:
                        result = _memory_safety(sandbox, cases, rules, bool(params.get("run_sanitizers", True)))
                    else:
                        result = {"status": "unavailable", "score": None, "rationale": "Compilation failed; sanitizer execution was not possible."}
                elif name == "structure":
                    result = score_structure(source_metrics, params)
                elif name == "readability":
                    max_line = int(params.get("max_line_length", 100))
                    long_lines = [i for i, line in enumerate(source.splitlines(), 1) if len(line) > max_line]
                    penalty = float(params.get("long_line_penalty", 2))
                    score = max(0.0, 100.0 - penalty * len(long_lines))
                    result = {"status": "completed", "score": round(score, 2), "long_lines": long_lines, "max_line_length": max_line,
                              "rubric": f"100 points minus {penalty:g} per line longer than {max_line} characters; floor 0.",
                              "rationale": "No lines exceed the configured length." if not long_lines else f"{len(long_lines)} line(s) exceed {max_line} characters: {long_lines[:12]}."}
                elif name == "modularity":
                    result = _modularity_result(source_metrics, params)
                else:
                    result = {"status": "unavailable", "score": None, "rationale": f"No deterministic implementation is available for analyzer '{name}'."}
                result["analyzer"] = metric["analyzer"]
                result["weight"] = metric["weight"]
                result["score_band"] = _score_band(result["score"]) if isinstance(result.get("score"), (int, float)) else "N/A"
                report["metrics"][name] = result

            report["test_summary"] = {
                "total": len(cases), "core": len(core_cases), "edge": len(edge_cases),
                "core_and_edge_partition": "disjoint; each test contributes to at most one of correctness/logical",
            }
            report["program_source_metrics"] = source_metrics
            report["llm_rsi"] = run_rsi_review(llm, rules, source, report["metrics"])
            report["scoring"] = _calculate_grade(rules["metrics"], report["metrics"])
            report["reproducibility"] = {
                "grade_inputs": {"source_sha256": source_hash, "rules_sha256": rules["sha256"], "suite_sha256": suite["suite_sha256"], "sandbox_image_id": sandbox.image_id},
                "deterministic_numeric_scoring": True,
                "cross_machine_note": "Exact cross-machine reproduction requires the same frozen suite cache and the same Docker image ID. Fresh LLM generation can vary across model builds/hardware even at temperature 0.",
            }
            report["warnings"] = list(suite.get("generation_warnings", [])) + list(report["llm_rsi"].get("warnings", []))
    except (SandboxError, SuiteError, LLMError) as exc:
        report["scoring"] = {"status": "incomplete", "score": None, "scale": "0-100", "contributions": [],
                             "missing_metrics": [{"metric": "grading_pipeline", "status": "failed", "reason": str(exc)}],
                             "reason": "The grading pipeline could not create a validated test suite or restricted runtime."}
        report["pipeline_error"] = f"{type(exc).__name__}: {exc}"
        report["warnings"] = [str(exc)]
    finally:
        llm.close()

    score_status = report.get("scoring", {}).get("status")
    if score_status == "completed":
        report["status"] = "completed_with_warnings" if report.get("warnings") else "completed"
    else:
        report["status"] = "incomplete"

    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "grading.json"
    text_path = report_dir / "grading.txt"
    safe_report = _json_safe(report)
    json_path.write_text(json.dumps(safe_report, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    text_path.write_text(render_text(safe_report), encoding="utf-8")
    return safe_report, json_path, text_path


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def render_text(report: dict[str, Any]) -> str:
    lines = [f"CODE GRADER REPORT — {report.get('submission', 'submission.c')}", "=" * 72,
             f"Status: {report.get('status', 'unknown')}",
             f"Problem: {report.get('problem', 'N/A')}",
             f"Source SHA-256: {report.get('source_sha256', 'N/A')}",
             f"Rules SHA-256: {report.get('rules_sha256', 'N/A')}", ""]
    preflight = report.get("preflight")
    if isinstance(preflight, dict):
        lines.extend(["PREFLIGHT", "-" * 72, f"Compiled: {preflight.get('compiled', 'N/A')}",
                      f"Compiler: {preflight.get('compiler_version', 'N/A')}", ""])
    suite = report.get("test_suite", {})
    if suite:
        lines.extend(["TEST SUITE", "-" * 72,
                      f"Status: {suite.get('status', 'unknown')}; cases: {suite.get('test_count', 0)}; generated: {suite.get('generated_count', 0)}; rejected: {suite.get('rejected_generated_count', 0)}; instructor: {suite.get('instructor_count', 0)}",
                      f"Suite SHA-256: {suite.get('suite_sha256', 'N/A')}",
                      f"Reference: {suite.get('reference_origin', 'unknown')}; instructor cross-check: {suite.get('oracle_crosschecked_with_instructor_examples', False)}"])
        inferred = suite.get("inferred_contract", {})
        if inferred:
            lines.append("  LLM-inferred I/O contract:")
            for key in ("input_format", "output_format", "constraints", "semantics", "formatting", "ambiguities"):
                if key in inferred:
                    lines.append(f"    {key}: {json.dumps(inferred[key], ensure_ascii=False)}")
        for iteration in suite.get("rsi_history", []):
            lines.append(f"  Test RSI pass {iteration.get('pass')}: {iteration.get('status')}; total tests={iteration.get('total_tests')}; {iteration.get('review_summary', iteration.get('error', ''))}")
        for warning in suite.get("generation_warnings", []):
            lines.append("  Warning: " + str(warning))
        lines.append("")
    lines.extend(["METRIC SCORES", "-" * 72, "Metric             Weight    Score     Contribution    Evidence"])
    score = report.get("scoring", {})
    contributions = {item["metric"]: item for item in score.get("contributions", [])}
    for name, result in report.get("metrics", {}).items():
        contribution = contributions.get(name, {})
        value = result.get("score")
        score_text = f"{value:.2f}" if isinstance(value, (int, float)) else "N/A"
        weight = result.get("weight", 0)
        contribution_text = f"{contribution.get('contribution', 0):.2f}" if contribution else "N/A"
        lines.append(f"{name:18s} {weight:7.3f}  {score_text:>7s}  {contribution_text:>12s}    {result.get('rationale', '')}")
        if result.get("rubric"):
            lines.append("  Rubric: " + str(result["rubric"]))
        if name in {"correctness", "logical"}:
            for test in result.get("results", []):
                if not test.get("passed"):
                    lines.extend([f"  FAIL {test['id']}: {test.get('failure_reason', 'output mismatch')}",
                                  "    INPUT: " + repr(test.get("input", "")),
                                  "    EXPECTED: " + repr(test.get("expected", "")),
                                  "    ACTUAL: " + repr(test.get("actual", ""))])
        elif name == "memory_safety":
            for test in result.get("results", []):
                if not test.get("clean"):
                    lines.append(f"  SANITIZER {test.get('id')}: {test.get('diagnostic') or test.get('stderr') or 'runtime failure'}")
    lines.extend(["", "SCORING RUBRIC", "-" * 72])
    for band in SCORE_RUBRIC:
        lines.append(f"  {band['range']} {band['label']}: {band['criteria']}")
    lines.extend(["", "OVERALL GRADE", "-" * 72])
    if isinstance(score.get("score"), (int, float)):
        lines.append(f"{score['score']:.2f}/100 — {score.get('status')}")
        for item in score.get("contributions", []):
            lines.append(f"  {item['metric']}: {item['score']:.2f} × {item['weight']:.3f} = {item['contribution']:.2f} ({item['score_band']})")
    else:
        lines.append("N/A — incomplete; no numeric grade is issued until all enabled weighted metrics are measured.")
        for item in score.get("missing_metrics", []):
            lines.append(f"  Missing {item.get('metric')}: {item.get('reason')}")
    originality = report.get("originality", {})
    lines.append("\nORIGINALITY / CORPUS SIMILARITY TRIAGE")
    lines.append("-" * 72)
    lines.append(f"Status: {originality.get('status', 'not configured')}; flagged: {originality.get('flagged', False)}")
    if originality.get("reason"):
        lines.append(str(originality["reason"]))
    if originality.get("matched_file"):
        lines.append(f"Closest corpus file: {originality['matched_file']} ({originality.get('max_similarity', 0):.1%} containment similarity)")
    if originality.get("disclaimer"):
        lines.append(str(originality["disclaimer"]))
    complexity = report.get("llm_rsi", {}).get("complexity_analysis")
    if complexity:
        lines.extend(["\nALGORITHM & COMPLEXITY ANALYSIS (LLM ADVISORY)", "-" * 72,
                      f"Algorithm: {complexity.get('algorithm')}",
                      f"Time: {complexity.get('time_complexity')}; auxiliary space: {complexity.get('space_complexity')}",
                      f"Explanation: {complexity.get('explanation')}"])
        for item in complexity.get("evidence", []):
            lines.append(f"  Line {item.get('line')}: {item.get('claim')} | {item.get('source')}")
    compression = report.get("llm_rsi", {}).get("context_compression", {})
    if compression:
        lines.extend(["\nLLM CONTEXT COMPRESSION", "-" * 72,
                      f"Method: {compression.get('method')}; source chars={compression.get('source_characters')}; review chars={compression.get('review_context_characters')}; truncated={compression.get('source_truncated')}"])
    lines.append("\nWHOLE-PROGRAM RSI (ADVISORY)")
    lines.append("-" * 72)
    rsi = report.get("llm_rsi", {})
    lines.append(f"Status: {rsi.get('status', 'unavailable')}")
    for item in rsi.get("iterations", []):
        lines.append(f"  Pass {item.get('pass')}: {item.get('status')}; retained findings={item.get('retained_findings')}; {item.get('assessment', item.get('error', ''))}")
    for finding in rsi.get("findings", []):
        lines.append(f"  [{finding['severity'].upper()}] {finding['title']} — line {finding['line_start']}: {finding['description']}")
        lines.append(f"    Evidence: {finding['evidence_excerpt']}")
    lines.append("  RSI findings are advisory only; no LLM-generated score changes the final grade.")
    lines.append("\nREPRODUCIBILITY")
    lines.append("-" * 72)
    for key, value in report.get("reproducibility", {}).get("grade_inputs", {}).items():
        lines.append(f"{key}: {value}")
    if report.get("reproducibility", {}).get("cross_machine_note"):
        lines.append(report["reproducibility"]["cross_machine_note"])
    for warning in report.get("warnings", []):
        lines.append("WARNING: " + str(warning))
    if report.get("pipeline_error"):
        lines.extend(["", "PIPELINE ERROR", report["pipeline_error"]])
    return "\n".join(lines) + "\n"
