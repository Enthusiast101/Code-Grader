"""Build and validate a deterministic evidence ledger for LLM reviews.

The ledger is generated exclusively from instructor rules, analyzer output,
and the submitted source. LLM text is never treated as grading evidence.
"""
from __future__ import annotations

import re
from typing import Any

from grading_agent.config import settings


INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def _add(
    ledger: list[dict[str, Any]],
    evidence_id: str,
    kind: str,
    statement: str,
    **facts: Any,
) -> None:
    record: dict[str, Any] = {
        "id": evidence_id,
        "kind": kind,
        "statement": statement[:360],
    }
    if facts:
        record["facts"] = facts
    ledger.append(record)


def _is_signed_int32_overflow_test(case: dict[str, Any], contract: dict[str, Any]) -> bool:
    """Classify only explicitly contracted signed-32-bit addition cases."""
    if contract.get("operand_type") != "int32" or contract.get("sum_type") != "int64":
        return False
    if "signed_int32_overflow" in case.get("tags", []):
        return True
    tokens = re.findall(r"[-+]?\d+", str(case.get("input", "")))
    if len(tokens) < 2:
        return False
    left, right = int(tokens[0]), int(tokens[1])
    if not (INT32_MIN <= left <= INT32_MAX and INT32_MIN <= right <= INT32_MAX):
        return False
    return not (INT32_MIN <= left + right <= INT32_MAX)


def build_evidence_ledger(
    report: dict[str, Any],
    rules: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create stable IDs for every material fact supplied to the reviewer."""
    rules = rules or {}
    ledger: list[dict[str, Any]] = []

    preflight = report.get("preflight")
    if isinstance(preflight, dict):
        _add(
            ledger, "E-PREFLIGHT-001", "preflight",
            "Preflight: " + ", ".join(
                f"{key}={preflight[key]}" for key in
                ("status", "compiled", "smoke_tested", "suspected_infinite_loop", "crashed", "exit_code", "timed_out")
                if key in preflight
            ),
            status=preflight.get("status"), compiled=preflight.get("compiled"),
            smoke_tested=preflight.get("smoke_tested"),
            suspected_infinite_loop=preflight.get("suspected_infinite_loop"),
        )

    contract = rules.get("problem_contract", {})
    if isinstance(contract, dict) and contract:
        _add(ledger, "E-RULE-CONTRACT-001", "rubric", "Instructor-defined problem contract.", **contract)

    rule_by_name = {
        item.get("name"): item
        for item in rules.get("metrics", [])
        if isinstance(item, dict) and item.get("enabled", True)
    }
    configuration = report.get("configuration", {})
    metric_names = [
        item.get("name") for item in configuration.get("enabled_metrics", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]

    for name in metric_names:
        metric = report.get(name)
        if not isinstance(metric, dict):
            continue
        score = metric.get("score")
        status = metric.get("status", "unknown")
        weight = metric.get("weight")
        metric_id = f"E-METRIC-{name.upper()}-001"
        _add(
            ledger, metric_id, "metric_summary",
            f"Metric {name}: status={status}, score={score}/100, weight={weight}.",
            metric=name, status=status, score=score, weight=weight,
        )

        rule = rule_by_name.get(name, {})
        params = rule.get("params", {}) if isinstance(rule, dict) else {}
        if name == "modularity":
            minimum = metric.get("minimum_functions", params.get("min_functions"))
            helpers = metric.get("helpers_expected", params.get("expect_helpers"))
            _add(
                ledger, "E-RULE-MODULARITY-001", "rubric",
                f"Modularity rubric: minimum_functions={minimum}; helpers_expected={helpers}.",
                minimum_functions=minimum, helpers_expected=helpers,
            )
            _add(
                ledger, "E-MODULARITY-001", "source_metric",
                f"Detected {metric.get('function_count')} functions and {metric.get('helper_function_count')} helper functions.",
                function_count=metric.get("function_count"),
                helper_function_count=metric.get("helper_function_count"),
                functions=metric.get("functions", [])[:8],
            )

        elif name in {"correctness", "logical"}:
            passed = metric.get("tests_passed", 0)
            total = metric.get("tests_total", 0)
            _add(
                ledger, f"E-{name.upper()}-SUMMARY-001", "test_summary",
                f"{name} test summary: {passed}/{total} passed.",
                metric=name, tests_passed=passed, tests_total=total,
            )
            overflow_cases: list[dict[str, Any]] = []
            for index, case in enumerate(metric.get("results", [])[:30], 1):
                if not isinstance(case, dict):
                    continue
                test_id = case.get("test_id", index)
                overflow_case = _is_signed_int32_overflow_test(case, contract)
                case_tags = case.get("tags", [])
                tags = list(case_tags) if isinstance(case_tags, list) else []
                if overflow_case and "signed_int32_overflow" not in tags:
                    tags.append("signed_int32_overflow")
                if overflow_case:
                    overflow_cases.append(case)
                _add(
                    ledger,
                    f"E-{name.upper()}-TEST-{index:03d}",
                    "test_result",
                    f"{name} test {test_id}: passed={case.get('passed')}; input={case.get('input')!r}; expected={case.get('expected')!r}; actual={case.get('actual')!r}; tags={tags}.",
                    metric=name, test_id=test_id, passed=case.get("passed"),
                    input=case.get("input"), expected=case.get("expected"),
                    actual=case.get("actual"), tags=tags,
                    exit_code=case.get("exit_code"), timed_out=case.get("timed_out"),
                )
            if name == "logical" and isinstance(contract, dict) and contract.get("operand_type") == "int32":
                overflow_passed = sum(case.get("passed") is True for case in overflow_cases)
                _add(
                    ledger, "E-LOGICAL-OVERFLOW-COVERAGE-001", "coverage_summary",
                    f"Signed int32 overflow coverage: {len(overflow_cases)} configured cases; {overflow_passed} passed; {len(overflow_cases) - overflow_passed} failed.",
                    overflow_tests_total=len(overflow_cases),
                    overflow_tests_passed=overflow_passed,
                    overflow_tests_failed=len(overflow_cases) - overflow_passed,
                    contract_operand_type=contract.get("operand_type"),
                    contract_sum_type=contract.get("sum_type"),
                )

        elif name == "syntax":
            _add(
                ledger, "E-SYNTAX-001", "compiler_diagnostics",
                f"Compiler: compiled={metric.get('compiled')}; warnings={metric.get('warning_count')}; errors={metric.get('error_count')}.",
                compiled=metric.get("compiled"), warning_count=metric.get("warning_count"),
                error_count=metric.get("error_count"), warnings=metric.get("warnings", [])[:5],
                errors=metric.get("errors", [])[:5],
            )

        elif name == "structure":
            findings = metric.get("findings", [])
            _add(
                ledger, "E-STRUCTURE-SUMMARY-001", "static_analysis",
                f"Static analysis: tool={metric.get('tool')}; finding_count={metric.get('finding_count')}; severity_counts={metric.get('severity_counts')}.",
                tool=metric.get("tool"), finding_count=metric.get("finding_count"),
                severity_counts=metric.get("severity_counts"),
            )
            for index, finding in enumerate(findings[:10], 1):
                if isinstance(finding, dict):
                    _add(
                        ledger, f"E-STRUCTURE-FINDING-{index:03d}", "static_finding",
                        str(finding.get("message") or finding.get("verbose") or finding),
                        **{key: finding.get(key) for key in ("severity", "id", "line", "column")},
                    )

        elif name == "complexity":
            _add(
                ledger, "E-COMPLEXITY-001", "source_metric",
                f"Static complexity estimates: max cyclomatic={metric.get('max_cyclomatic_complexity_estimate')}; max function length={metric.get('max_function_length')}; thresholds={metric.get('thresholds')}.",
                max_cyclomatic=metric.get("max_cyclomatic_complexity_estimate"),
                max_function_length=metric.get("max_function_length"),
                thresholds=metric.get("thresholds"),
                complexity_violations=metric.get("complexity_violations", []),
                length_violations=metric.get("length_violations", []),
                measurement_note=metric.get("measurement_note"),
            )

        elif name == "runtime_efficiency":
            _add(
                ledger, "E-RUNTIME-001", "runtime_measurement",
                f"Runtime profile: median={metric.get('median_runtime_ms')} ms; p95 per-test median={metric.get('p95_per_test_median_ms')} ms; target={metric.get('target_runtime_ms')} ms.",
                median_runtime_ms=metric.get("median_runtime_ms"),
                p95_per_test_median_ms=metric.get("p95_per_test_median_ms"),
                target_runtime_ms=metric.get("target_runtime_ms"),
                measurement_note=metric.get("measurement_note"),
            )

        elif name == "memory_safety":
            runtime = metric.get("runtime_tools", {})
            _add(
                ledger, "E-MEMORY-SUMMARY-001", "memory_safety_measurement",
                f"Memory-safety checks: status={metric.get('status')}; runtime status={runtime.get('status') if isinstance(runtime, dict) else 'unknown'}; tests_clean={metric.get('tests_clean')}/{metric.get('tests_total')}; memory_safe={metric.get('memory_safe')}.",
                tests_clean=metric.get("tests_clean"), tests_total=metric.get("tests_total"),
                memory_safe=metric.get("memory_safe"), runtime_status=runtime.get("status") if isinstance(runtime, dict) else None,
            )
            if isinstance(runtime, dict):
                for index, test in enumerate(runtime.get("asan_ubsan_results", [])[:12], 1):
                    if isinstance(test, dict):
                        _add(
                            ledger, f"E-SANITIZER-{index:03d}", "sanitizer_result",
                            f"ASan/UBSan test {test.get('test_id')}: clean={test.get('clean')}; address_error={test.get('address_sanitizer_finding')}; undefined_behavior={test.get('undefined_behavior_finding')}; sanitizer_error={test.get('sanitizer_error')}.",
                            test_id=test.get("test_id"), clean=test.get("clean"),
                            address_sanitizer_finding=test.get("address_sanitizer_finding"),
                            undefined_behavior_finding=test.get("undefined_behavior_finding"),
                            sanitizer_error=test.get("sanitizer_error"),
                        )
            audit = metric.get("llm_source_audit", {})
            if isinstance(audit, dict):
                audit_payload = audit.get("audit", {})
                _add(
                    ledger, "E-MEMORY-SOURCE-AUDIT-001", "model_audit",
                    f"Separate model source audit status={audit.get('status')}; findings={len(audit_payload.get('findings', [])) if isinstance(audit_payload, dict) and isinstance(audit_payload.get('findings'), list) else 'unknown'}. Model-derived analysis is not independent runtime proof.",
                    status=audit.get("status"),
                    finding_count=(len(audit_payload.get("findings", [])) if isinstance(audit_payload, dict) and isinstance(audit_payload.get("findings"), list) else None),
                )
                if isinstance(audit_payload, dict):
                    for index, finding in enumerate(audit_payload.get("findings", [])[:10], 1):
                        if isinstance(finding, dict):
                            _add(
                                ledger, f"E-MEMORY-AUDIT-FINDING-{index:03d}", "model_audit_finding",
                                str(finding.get("description") or finding.get("category") or "Source-audit finding."),
                                category=finding.get("category"), severity=finding.get("severity"),
                                line=finding.get("line"), code_excerpt=finding.get("code_excerpt"),
                                description=finding.get("description"),
                            )

        elif name == "readability":
            _add(
                ledger, "E-READABILITY-001", "readability_measurement",
                f"Readability heuristics: score={score}; single_character_identifiers={metric.get('single_character_identifier_count')}; identifiers={metric.get('identifier_count')}; long_lines={metric.get('long_lines')}; comment_lines={metric.get('comment_line_count')}.",
                score=score, single_character_identifier_count=metric.get("single_character_identifier_count"),
                identifier_count=metric.get("identifier_count"), long_lines=metric.get("long_lines"),
                comment_line_count=metric.get("comment_line_count"),
                measurement_note=metric.get("measurement_note"),
            )

    source = report.get("source_code")
    if isinstance(source, str):
        lines = source.splitlines()
        target_lines: set[int] = set()
        for metric_name in ("structure", "memory_safety"):
            metric = report.get(metric_name, {})
            if not isinstance(metric, dict):
                continue
            candidate_findings = metric.get("findings", []) if metric_name == "structure" else metric.get("llm_source_audit", {}).get("audit", {}).get("findings", [])
            if isinstance(candidate_findings, list):
                for finding in candidate_findings:
                    line_number = finding.get("line") if isinstance(finding, dict) else None
                    if isinstance(line_number, int) and not isinstance(line_number, bool) and 1 <= line_number <= len(lines):
                        target_lines.add(line_number)
        selected_lines = set(range(1, min(len(lines), 50) + 1)) | target_lines
        for index in sorted(selected_lines):
            line = lines[index - 1]
            _add(ledger, f"E-SOURCE-L{index:03d}", "source_line", f"Source line {index}: {line[:180]!r}.", line=index, code=line[:180])
        if len(lines) > 50:
            _add(ledger, "E-SOURCE-TRUNCATED-001", "scope_limitation", f"The first 50 lines and directly cited finding lines are represented individually; full source remains in report ({len(lines)} lines).", line_count=len(lines))

    return ledger


def validate_review_payload(
    payload: dict[str, Any],
    ledger: list[dict[str, Any]],
    rules: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Reject uncited, mis-scoped, or rubric-contradicting LLM claims.

    This validator enforces referential and rule-specific constraints. It does
    not claim to prove natural-language entailment; verified claims retain
    evidence IDs so instructors can inspect the underlying records.
    """
    rules = rules or {}
    by_id = {item["id"]: item for item in ledger}
    required_lists = ("findings", "contradictions", "recommendations", "uncertainties")
    summary = payload.get("summary")
    summary_ids = payload.get("summary_evidence_ids")
    if not isinstance(summary, str) or not summary.strip() or not isinstance(summary_ids, list) or not summary_ids:
        return None, {"status": "invalid", "reason": "A non-empty summary and summary_evidence_ids are required."}
    if any(not isinstance(item, str) or item not in by_id for item in summary_ids):
        return None, {"status": "invalid", "reason": "Summary cites unknown evidence IDs."}
    all_overflow_tests = [
        item for item in ledger
        if item.get("kind") == "test_result"
        and "signed_int32_overflow" in item.get("facts", {}).get("tags", [])
    ]
    def claims_overflow_is_handled(text: str) -> bool:
        """Detect affirmative claims, not phrases about failed overflow tests."""
        normalized = text.casefold()
        positive_patterns = (
            r"\boverflow\b.{0,28}\b(?:(?:is|was|has been)\s+)?(?:correctly\s+)?(?:handled|supported|safe|works|passed|passes|succeeded|successful|avoided|avoids|prevented|prevents|correct)\b",
            r"\b(?:handles?|handling|supports?|avoids?|prevents?|passes?|succeeded)\b.{0,28}\b(?:signed )?(?:integer )?overflow\b",
            r"\b(?:signed )?integer overflow\b.{0,28}\b(?:correctly handled|handled safely|supported|passed|passes|successful)\b",
            r"\boverflow\b.{0,24}\b(?:case|test|tests)\b.{0,24}\b(?:passed|pass|succeeded|successful)\b",
            r"\bno (?:signed )?(?:integer )?overflow\b.{0,20}\b(?:occurs|happens|possible|present)\b",
        )
        return any(re.search(pattern, normalized) for pattern in positive_patterns)

    positive_overflow_claim = claims_overflow_is_handled(summary)
    if positive_overflow_claim and (not all_overflow_tests or not all(item.get("facts", {}).get("passed") is True for item in all_overflow_tests)):
        return None, {"status": "invalid", "reason": "Summary claims overflow handling without passing overflow-specific tests."}

    def required_test_metrics(text: str) -> set[str]:
        normalized = text.casefold()
        if not re.search(r"\ball\s+(?:configured\s+)?(?:tests|test cases|edge cases)\b.{0,45}\b(?:pass(?:ed)?|succeed(?:ed)?|correct(?:ly)?|handle(?:d)?)\b", normalized):
            return set()
        if re.search(r"\b(?:edge cases|edge tests|corner cases)\b", normalized):
            return {"logical"}
        if re.search(r"\b(?:correctness|i/o) tests\b", normalized):
            return {"correctness"}
        return {
            item.get("facts", {}).get("metric")
            for item in ledger if item.get("kind") == "test_summary"
        }

    summary_test_scope = required_test_metrics(summary)
    if summary_test_scope:
        summaries = [
            item for item in ledger
            if item.get("kind") == "test_summary" and item.get("facts", {}).get("metric") in summary_test_scope
        ]
        cited_summaries = [
            by_id[ref] for ref in summary_ids
            if by_id[ref].get("kind") == "test_summary"
            and by_id[ref].get("facts", {}).get("metric") in summary_test_scope
        ]
        all_metrics_pass = bool(summaries) and all(
            item.get("facts", {}).get("tests_total", 0) > 0
            and item.get("facts", {}).get("tests_passed") == item.get("facts", {}).get("tests_total")
            for item in summaries
        )
        if not all_metrics_pass or len(cited_summaries) != len(summaries):
            return None, {"status": "invalid", "reason": "Summary claims all tests passed without citing complete passing summaries for the stated test category."}
    if any(not isinstance(payload.get(key), list) for key in required_lists):
        return None, {"status": "invalid", "reason": "Review sections must be JSON arrays."}

    metric_config = {
        item.get("name"): item
        for item in rules.get("metrics", [])
        if isinstance(item, dict) and item.get("enabled", True)
    }
    modularity_params = metric_config.get("modularity", {}).get("params", {})
    helpers_expected = modularity_params.get("expect_helpers", settings.ANALYZER_EXPECT_HELPERS)

    verified: dict[str, list[dict[str, Any]]] = {key: [] for key in required_lists}
    rejected: list[dict[str, str]] = []

    def validate_item(section: str, item: Any, index: int) -> None:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            rejected.append({"section": section, "reason": f"Item {index} must be an object with non-empty text."})
            return
        ids = item.get("evidence_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(ref, str) or ref not in by_id for ref in ids):
            rejected.append({"section": section, "claim": item["text"][:240], "reason": "Missing or unknown evidence IDs."})
            return

        text = item["text"]
        normalized = text.casefold()
        evidence_types = {by_id[ref]["kind"] for ref in ids}
        if section == "findings" and item.get("kind") not in {"observed", "potential", "limitation"}:
            rejected.append({"section": section, "claim": text[:240], "reason": "Finding kind must be observed, potential, or limitation."})
            return
        if section == "findings" and not evidence_types.intersection({
            "test_result", "compiler_diagnostics", "static_finding", "source_metric",
            "runtime_measurement", "memory_safety_measurement", "sanitizer_result",
            "readability_measurement", "source_line", "model_audit", "model_audit_finding",
        }):
            rejected.append({"section": section, "claim": text[:240], "reason": "Finding cites no relevant analyzer or source evidence."})
            return

        positive_overflow_item = claims_overflow_is_handled(text)
        if positive_overflow_item:
            if not all_overflow_tests or not all(entry.get("facts", {}).get("passed") is True for entry in all_overflow_tests):
                rejected.append({"section": section, "claim": text[:240], "reason": "Cannot claim overflow handling without all overflow-specific tests passing."})
                return

        item_test_scope = required_test_metrics(text)
        if item_test_scope:
            summaries = [
                by_id[ref] for ref in ids
                if by_id[ref].get("kind") == "test_summary"
                and by_id[ref].get("facts", {}).get("metric") in item_test_scope
            ]
            all_metrics = [
                entry for entry in ledger
                if entry.get("kind") == "test_summary"
                and entry.get("facts", {}).get("metric") in item_test_scope
            ]
            if len(summaries) != len(all_metrics) or not all(
                entry.get("facts", {}).get("tests_total", 0) > 0
                and entry.get("facts", {}).get("tests_passed") == entry.get("facts", {}).get("tests_total")
                for entry in summaries
            ):
                rejected.append({"section": section, "claim": text[:240], "reason": "Cannot claim all tests passed without citing complete passing summaries for the stated test category."})
                return

        if (
            any(phrase in normalized for phrase in ("completely memory safe", "proven memory safe", "guaranteed memory safe", "universally safe"))
            or re.search(r"\b(?:the )?(?:code|program|submission)\b.{0,24}\b(?:is|remains)\s+(?:completely\s+)?memory[- ]safe\b", normalized)
        ):
            rejected.append({"section": section, "claim": text[:240], "reason": "Finite tests cannot establish universal memory safety."})
            return

        if section == "contradictions" and len(set(ids)) < 2:
            rejected.append({"section": section, "claim": text[:240], "reason": "A contradiction must cite at least two evidence records."})
            return

        if section == "recommendations":
            basis = item.get("basis")
            if basis not in {"rubric", "measured_issue", "optional"}:
                rejected.append({"section": section, "claim": text[:240], "reason": "Recommendation basis must be rubric, measured_issue, or optional."})
                return
            if not helpers_expected and re.search(r"\b(helper function|add a helper|extract .*function|split .*function)\b", normalized):
                rejected.append({"section": section, "claim": text[:240], "reason": "Recommendation conflicts with the instructor's helpers_expected=false rule."})
                return
            if basis == "rubric" and "rubric" not in evidence_types:
                rejected.append({"section": section, "claim": text[:240], "reason": "Rubric-required recommendation must cite a rubric evidence record."})
                return

        verified[section].append({
            "text": text.strip()[:500],
            "evidence_ids": list(dict.fromkeys(ids))[:5],
            **({"kind": item.get("kind")} if section == "findings" and item.get("kind") else {}),
            **({"basis": item.get("basis")} if section == "recommendations" else {}),
        })

    for section in required_lists:
        for index, item in enumerate(payload.get(section, []), 1):
            validate_item(section, item, index)

    valid_reference_ids = payload.get("evidence_references", [])
    if not isinstance(valid_reference_ids, list):
        valid_reference_ids = []
    valid_reference_ids = [ref for ref in valid_reference_ids if isinstance(ref, str) and ref in by_id]
    referenced = set(valid_reference_ids)
    referenced.update(summary_ids)
    for section_items in verified.values():
        for item in section_items:
            referenced.update(item["evidence_ids"])

    cleaned = {
        "summary": summary.strip()[:800],
        "summary_evidence_ids": list(dict.fromkeys(summary_ids))[:5],
        **verified,
        "evidence_references": sorted(referenced),
        "unverified_claims": rejected,
    }
    status = "completed" if not rejected else "completed_with_warnings"
    validation = {
        "status": status,
        "verified_claim_count": sum(len(items) for items in verified.values()),
        "rejected_claim_count": len(rejected),
        "known_evidence_ids": len(by_id),
        "referenced_evidence_ids": len(referenced),
        "semantic_entailment_note": "Python validates citation IDs, evidence types, explicit assignment constraints, and known unsupported claim patterns; natural-language entailment still requires human review.",
    }
    return cleaned, validation
