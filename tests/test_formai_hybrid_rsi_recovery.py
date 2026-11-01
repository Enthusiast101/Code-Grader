from __future__ import annotations

import unittest
from unittest.mock import patch

from grading_agent.benchmark.formai_predict import (
    _build_rsi_review_report,
    _prediction_from_rsi_review,
    _select_hybrid_prediction,
)
from grading_agent.llm.evidence import build_evidence_ledger
from grading_agent.llm import reviewer


class FormAIHybridRSIRecoveryTests(unittest.TestCase):
    def test_positive_vulnerability_summary_triggers_recovery(self) -> None:
        self.assertTrue(
            reviewer._summary_suggests_findings(
                "The code contains several potential vulnerabilities, including an out-of-bounds write."
            )
        )

    def test_uploaded_false_negative_summaries_trigger_focused_extraction(self) -> None:
        # These are the exact summary patterns in the user's 16-record
        # hybrid_rsi output where the model described defects but returned [].
        summaries = [
            "Potential buffer overflow and other vulnerabilities in the FTP client code",
            "The code contains a buffer overflow vulnerability in the filename input handling and potential issues with string manipulation.",
            "The provided C code contains several potential vulnerabilities that could be exploited in a real-world scenario.",
            "The code contains uninitialized variable access in functions that use the completion_time field, leading to potential undefined behavior.",
            "The code contains two critical vulnerabilities: an out-of-bounds write in rateMovie and a division-by-zero in averageRating.",
            "The provided C source code contains several potential vulnerabilities that could lead to undefined behavior or security issues.",
        ]
        for summary in summaries:
            with self.subTest(summary=summary):
                self.assertTrue(reviewer._summary_suggests_findings(summary))

    def test_negative_summary_does_not_trigger_recovery(self) -> None:
        self.assertFalse(
            reviewer._summary_suggests_findings(
                "No concrete source-level vulnerabilities found in the provided code."
            )
        )

    def test_audit_validator_normalises_fields_and_uses_source_line_excerpt(self) -> None:
        source = "int main(void) {\n    int value;\n    return value;\n}"
        audit = {
            "summary": "Potential uninitialised read.",
            "issues": [
                {
                    "error_type": "Use of uninitialized variable",
                    "severity": "warning",
                    "line_number": "line 3",
                    "code_excerpt": "return value;",
                    "reason": "value is read before it is initialized.",
                }
            ],
        }
        cleaned, error = reviewer._validate_memory_audit(audit, source, 10)
        self.assertIsNone(error)
        self.assertIsNotNone(cleaned)
        finding = cleaned["findings"][0]
        self.assertEqual(finding["category"], "Use of uninitialized variable")
        self.assertEqual(finding["severity"], "medium")
        self.assertEqual(finding["line"], 3)
        self.assertEqual(finding["code_excerpt"], "return value;")
        self.assertTrue(finding["source_line_verified"])

    def test_source_audit_prompt_contains_explicit_one_based_line_numbers(self) -> None:
        source = "int main(void) {\n    int value = 1;\n    return value;\n}"
        captured = {}

        def fake_request(messages, max_tokens):
            captured["messages"] = messages
            return ({"summary": "No concrete vulnerabilities found.", "findings": [], "limitations": []}, "{}")

        with patch.object(reviewer, "_request_json", side_effect=fake_request):
            result = reviewer._review_c_source(
                source, max_findings=5, audit_domain="vulnerability",
                audit_instructions="Check security defects.", task="Audit source.",
            )
        self.assertEqual(result["status"], "completed")
        payload = __import__("json").loads(captured["messages"][1]["content"])
        numbered = payload["source_code_with_line_numbers"]
        self.assertIn("00001 | int main(void) {", numbered)
        self.assertIn("00003 |     return value;", numbered)

    def test_audit_validator_rejects_excerpt_from_neighbouring_or_unrelated_line(self) -> None:
        source = "#include <sys/socket.h>\nint main(void) {\n    int sock = socket(AF_INET, SOCK_STREAM, 0);\n    fgets(buffer, sizeof(buffer), stdin);\n    return 0;\n}"
        audit = {
            "summary": "Potential buffer overflow in input handling.",
            "findings": [{
                "category": "Buffer Overflow",
                "severity": "critical",
                "line": 3,
                "code_excerpt": "fgets(buffer, sizeof(buffer), stdin);",
                "description": "Input is not safely bounded.",
            }],
            "limitations": [],
        }
        cleaned, error = reviewer._validate_memory_audit(audit, source, 10)
        self.assertIsNone(error)
        self.assertEqual(cleaned["findings"], [])
        self.assertEqual(
            cleaned["rejected_findings"][0]["reason"],
            "The quoted code_excerpt does not occur on the exact cited source line.",
        )

    def test_audit_validator_accepts_exact_cited_line_and_not_neighbouring_quote(self) -> None:
        source = "int main(void) {\n    char dst[8];\n    strcpy(dst, input);\n    return 0;\n}"
        audit = {
            "summary": "Potential buffer overflow.",
            "findings": [{
                "category": "Buffer Overflow", "severity": "high", "line": 3,
                "code_excerpt": "strcpy(dst, input);",
                "description": "The copy has no destination bound.",
            }],
            "limitations": [],
        }
        cleaned, error = reviewer._validate_memory_audit(audit, source, 10)
        self.assertIsNone(error)
        self.assertEqual(len(cleaned["findings"]), 1)
        self.assertEqual(cleaned["findings"][0]["line"], 3)
        self.assertTrue(cleaned["findings"][0]["source_line_verified"])

    def test_positive_summary_with_empty_findings_runs_focused_extraction(self) -> None:
        source = "int main(void) {\n    int value;\n    return value;\n}"
        first = {
            "summary": "Potential uninitialized read when main returns value.",
            "findings": [],
            "limitations": [],
        }
        follow_up = {
            "summary": "An uninitialized value is read in main.",
            "findings": [
                {
                    "category": "Use of uninitialized variable",
                    "severity": "warning",
                    "line": 3,
                    "code_excerpt": "return value;",
                    "description": "value is read before initialization.",
                }
            ],
            "limitations": [],
        }
        with patch.object(reviewer, "_request_json", side_effect=[(first, "{}"), (follow_up, "{}")]) as request:
            result = reviewer._review_c_source(
                source,
                max_findings=10,
                audit_domain="vulnerability",
                audit_instructions="Find concrete security defects.",
                task="Audit this C source.",
            )
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result["status"], "completed_with_warnings")
        self.assertTrue(result["finding_recovery_attempted"])
        self.assertTrue(result["finding_recovery_succeeded"])
        self.assertEqual(result["audit"]["findings"][0]["line"], 3)

    def test_positive_summary_without_recoverable_claim_is_reported_as_warning(self) -> None:
        source = "int main(void) { return 0; }"
        first = {
            "summary": "Potential buffer overflow risk was considered.",
            "findings": [],
            "limitations": [],
        }
        no_finding = {
            "summary": "The source does not support a concrete buffer overflow claim.",
            "findings": [],
            "limitations": [],
        }
        with patch.object(reviewer, "_request_json", side_effect=[(first, "{}"), (no_finding, "{}")]):
            result = reviewer._review_c_source(
                source,
                max_findings=10,
                audit_domain="vulnerability",
                audit_instructions="Find concrete security defects.",
                task="Audit this C source.",
            )
        self.assertEqual(result["status"], "completed_with_warnings")
        self.assertEqual(result["audit"]["findings"], [])
        self.assertTrue(any("focused follow-up" in x for x in result["audit"]["limitations"]))


    def test_rejected_llm_candidate_gets_focused_recovery(self) -> None:
        source = "int main(void) {\n    int value;\n    return value;\n}"
        first = {
            "summary": "Audit complete.",
            "findings": [{
                "category": "Use of uninitialized variable",
                "severity": "medium",
                "line": 99,
                "description": "value may be read before initialization.",
            }],
            "limitations": [],
        }
        retry = {
            "summary": "No concrete source-level vulnerabilities found in the provided code.",
            "findings": [],
            "limitations": [],
        }
        with patch.object(reviewer, "_request_json", side_effect=[(first, "{}"), (retry, "{}")]) as request:
            result = reviewer._review_c_source(
                source, max_findings=10, audit_domain="vulnerability",
                audit_instructions="Find concrete source-level security defects.",
                task="Audit this source.",
            )
        self.assertEqual(request.call_count, 2)
        self.assertTrue(result["finding_recovery_attempted"])
        self.assertTrue(result["finding_recovery_completed"])
        self.assertFalse(result["unresolved_invalid_candidates"])
        self.assertEqual(result["audit"]["findings"], [])

    def test_rejected_llm_candidate_without_successful_recovery_is_not_safe(self) -> None:
        source = "int main(void) { return 0; }"
        first = {
            "summary": "Audit complete.",
            "findings": [{
                "category": "Buffer Overflow", "line": 0,
                "description": "Possible overflow.",
            }],
            "limitations": [],
        }
        with patch.object(reviewer, "_request_json", side_effect=[(first, "{}"), (None, "invalid JSON")]):
            audit = reviewer._review_c_source(
                source, max_findings=10, audit_domain="vulnerability",
                audit_instructions="Find concrete source-level security defects.",
                task="Audit this source.",
            )
        self.assertTrue(audit["unresolved_invalid_candidates"])
        predicted, _, _, status = _select_hybrid_prediction(
            mode="hybrid_rsi", tool_status="completed", tool_findings=[],
            llm_audit=audit,
            rsi_review={"status": "completed", "claim_validation": {"status": "completed"}},
            rsi_findings=[],
        )
        self.assertIsNone(predicted)
        self.assertEqual(status, "inconclusive")

    def test_source_evidence_includes_context_around_late_cppcheck_line(self) -> None:
        source = "\n".join(f"int line_{i} = {i};" for i in range(1, 71))
        item = {"file_name": "late.c"}
        report = _build_rsi_review_report(
            item,
            source,
            [{"line": 60, "id": "arrayIndexOutOfBounds", "description": "Out-of-bounds array access"}],
            "completed",
            {"status": "completed", "audit": {"summary": "No model findings.", "findings": []}},
        )
        ledger = build_evidence_ledger(report, {"metrics": []})
        ids = {entry["id"] for entry in ledger}
        self.assertIn("E-SOURCE-L058", ids)
        self.assertIn("E-SOURCE-L060", ids)
        self.assertIn("E-SOURCE-L062", ids)
        prepared_ids = {
            entry["id"]
            for entry in reviewer._prepare_evidence(report, {"metrics": []}, ledger)["evidence_ledger"]
        }
        self.assertIn("E-SOURCE-L058", prepared_ids)
        self.assertIn("E-SOURCE-L062", prepared_ids)


    def test_positive_llm_summary_is_visible_to_rsi_even_without_findings(self) -> None:
        report = _build_rsi_review_report(
            {"file_name": "summary-only.c"},
            "int main(void) { return 0; }",
            [], "completed",
            {"status": "completed_with_warnings", "audit": {
                "summary": "The source may contain an out-of-bounds write.",
                "findings": [],
                "limitations": ["Focused line extraction did not identify a concrete line."],
            }},
        )
        ledger = build_evidence_ledger(report, {"metrics": []})
        audit_entry = next(entry for entry in ledger if entry["id"] == "E-MEMORY-SOURCE-AUDIT-001")
        self.assertIn("out-of-bounds write", audit_entry["facts"]["summary"])
        prepared = reviewer._prepare_evidence(report, {"metrics": []}, ledger)
        compact_audit = next(
            entry for entry in prepared["evidence_ledger"]
            if entry["id"] == "E-MEMORY-SOURCE-AUDIT-001"
        )
        self.assertIn("out-of-bounds write", compact_audit["facts"]["summary"])

    def test_rsi_focus_explicitly_adjudicates_llm_candidates(self) -> None:
        source = "int main(void) {\n int x;\n return x;\n}"
        report = _build_rsi_review_report(
            {"file_name": "uninit.c"},
            source,
            [],
            "completed",
            {
                "status": "completed",
                "audit": {
                    "summary": "Uninitialized read.",
                    "findings": [{
                        "category": "Use of uninitialized variable",
                        "severity": "medium",
                        "line": 3,
                        "code_excerpt": " return x;",
                        "description": "x is read before initialization.",
                    }],
                },
            },
        )
        self.assertIn("EACH LLM source-audit finding", report["review_focus"])
        ledger = build_evidence_ledger(report, {"metrics": []})
        self.assertIn("E-MEMORY-AUDIT-FINDING-001", {entry["id"] for entry in ledger})
        self.assertIn("E-SOURCE-L003", {entry["id"] for entry in ledger})
        prepared = reviewer._prepare_evidence(report, {"metrics": []}, ledger)
        prepared_ids = {entry["id"] for entry in prepared["evidence_ledger"]}
        self.assertIn("E-MEMORY-AUDIT-FINDING-001", prepared_ids)
        self.assertIn("E-SOURCE-L003", prepared_ids)
        self.assertIn("review_focus", prepared)

    def test_rsi_finds_can_be_classified_from_cited_evidence(self) -> None:
        ledger = [
            {
                "id": "E-SOURCE-L003", "kind": "source_line",
                "statement": "Source line 3: return value;",
                "facts": {"line": 3, "code": "return value;"},
            },
            {
                "id": "E-MEMORY-AUDIT-FINDING-001", "kind": "model_audit_finding",
                "statement": "Use of uninitialized variable: value is read before initialization.",
                "facts": {"line": 3, "category": "Use of uninitialized variable", "severity": "medium", "code_excerpt": "return value;"},
            },
        ]
        review_result = {
            "status": "completed",
            "claim_validation": {"status": "completed"},
            "review": {
                "findings": [{
                    "text": "The read at the cited line uses a value before initialization.",
                    "kind": "observed",
                    "evidence_ids": ["E-MEMORY-AUDIT-FINDING-001", "E-SOURCE-L003"],
                }]
            },
        }
        predicted, line, error_type, findings = _prediction_from_rsi_review(review_result, ledger)
        self.assertTrue(predicted)
        self.assertEqual(line, 3)
        self.assertEqual(error_type, "Use of uninitialized variable")
        self.assertEqual(len(findings), 1)

    def test_unquoted_llm_candidate_requires_rsi_confirmation(self) -> None:
        candidate = {
            "category": "Use of uninitialized variable",
            "severity": "medium",
            "line": 3,
            "code_excerpt": "return value;",
            "excerpt_derived_from_source": True,
            "description": "value is read before initialization.",
        }
        predicted, _, _, status = _select_hybrid_prediction(
            mode="hybrid_rsi", tool_status="completed", tool_findings=[],
            llm_audit={"status": "completed", "audit": {"findings": [candidate]}},
            rsi_review={"status": "completed", "claim_validation": {"status": "completed"}}, rsi_findings=[],
        )
        self.assertIsNone(predicted)
        self.assertEqual(status, "inconclusive")

        predicted, line, category, _ = _select_hybrid_prediction(
            mode="hybrid_rsi", tool_status="completed", tool_findings=[],
            llm_audit={"status": "completed", "audit": {"findings": [candidate]}},
            rsi_review={"status": "completed", "claim_validation": {"status": "completed"}},
            rsi_findings=[{"line": 3, "error_type": "Use of uninitialized variable"}],
        )
        self.assertTrue(predicted)
        self.assertEqual(line, 3)
        self.assertEqual(category, "Use of uninitialized variable")

    def test_unresolved_positive_summary_is_not_labelled_safe(self) -> None:
        predicted, line, category, status = _select_hybrid_prediction(
            mode="hybrid_rsi", tool_status="completed", tool_findings=[],
            llm_audit={
                "status": "completed_with_warnings",
                "unresolved_positive_summary": True,
                "audit": {"findings": []},
            },
            rsi_review={"status": "completed_with_warnings"}, rsi_findings=[],
        )
        self.assertIsNone(predicted)
        self.assertIsNone(line)
        self.assertIsNone(category)
        self.assertEqual(status, "inconclusive")

    def test_hybrid_keeps_valid_llm_finding_when_tools_are_clean(self) -> None:
        predicted, line, category, status = _select_hybrid_prediction(
            mode="hybrid",
            tool_status="completed",
            tool_findings=[],
            llm_audit={
                "status": "completed_with_warnings",
                "audit": {
                    "findings": [{
                        "category": "Use of uninitialized variable",
                        "severity": "medium",
                        "line": 3,
                        "code_excerpt": "return value;",
                        "description": "value is read before initialization.",
                    }]
                },
            },
            rsi_review={"status": "completed_with_warnings"},
            rsi_findings=[],
        )
        self.assertTrue(predicted)
        self.assertEqual(line, 3)
        self.assertEqual(category, "Use of uninitialized variable")
        self.assertEqual(status, "completed_with_warnings")


    def test_hybrid_rsi_does_not_call_incomplete_primary_stages_safe(self) -> None:
        predicted, _, _, status = _select_hybrid_prediction(
            mode="hybrid_rsi",
            tool_status="unavailable",
            tool_findings=[],
            llm_audit={"status": "unavailable", "audit": {"findings": []}},
            rsi_review={"status": "skipped"},
            rsi_findings=[],
        )
        self.assertIsNone(predicted)
        self.assertEqual(status, "inconclusive")


    def test_hybrid_rsi_does_not_return_safe_when_rsi_is_unavailable(self) -> None:
        predicted, _, _, status = _select_hybrid_prediction(
            mode="hybrid_rsi",
            tool_status="completed",
            tool_findings=[],
            llm_audit={"status": "completed", "audit": {"summary": "No concrete vulnerabilities found.", "findings": []}},
            rsi_review={"status": "unavailable", "error": "simulated RSI outage"},
            rsi_findings=[],
        )
        self.assertIsNone(predicted)
        self.assertEqual(status, "inconclusive")

    def test_hybrid_rsi_retains_static_positive_when_llm_is_unavailable(self) -> None:
        predicted, line, error_type, status = _select_hybrid_prediction(
            mode="hybrid_rsi",
            tool_status="completed",
            tool_findings=[{"line": 14, "id": "arrayIndexOutOfBounds", "description": "out of bounds"}],
            llm_audit={"status": "unavailable", "audit": {"findings": []}},
            rsi_review={"status": "partial"},
            rsi_findings=[],
        )
        self.assertTrue(predicted)
        self.assertEqual(line, 14)
        self.assertEqual(error_type, "arrayIndexOutOfBounds")
        self.assertEqual(status, "completed_with_warnings")

    def test_hybrid_rsi_returns_safe_only_when_all_stages_completed_cleanly(self) -> None:
        predicted, line, error_type, status = _select_hybrid_prediction(
            mode="hybrid_rsi",
            tool_status="completed",
            tool_findings=[],
            llm_audit={"status": "completed", "audit": {"summary": "No concrete vulnerabilities found.", "findings": []}},
            rsi_review={"status": "completed", "claim_validation": {"status": "completed"}},
            rsi_findings=[],
        )
        self.assertFalse(predicted)
        self.assertIsNone(line)
        self.assertIsNone(error_type)
        self.assertEqual(status, "completed")

    def test_review_evidence_preserves_earlier_validated_findings(self) -> None:
        source = "\n".join(["int main(void) {", "    int value;", "    return value;", "}"])
        report = _build_rsi_review_report(
            {"file_name": "preserve.c"},
            source,
            [],
            "completed",
            {
                "status": "completed",
                "audit": {
                    "summary": "Potential uninitialized read.",
                    "findings": [{
                        "category": "Use of uninitialized variable",
                        "severity": "medium",
                        "line": 3,
                        "code_excerpt": "return value;",
                        "description": "value is read before initialization.",
                    }],
                },
            },
        )
        prior_review = {
            "summary": "An uninitialized read is supported by the cited source.",
            "summary_evidence_ids": ["E-MEMORY-AUDIT-FINDING-001"],
            "findings": [{
                "text": "Potential use of uninitialized variable at line 3.",
                "kind": "potential",
                "evidence_ids": ["E-MEMORY-AUDIT-FINDING-001", "E-SOURCE-L003"],
            }],
            "contradictions": [], "recommendations": [], "uncertainties": [],
            "evidence_references": [],
        }
        final_review = {
            "summary": "No other vulnerability claims were added.",
            "summary_evidence_ids": ["E-MEMORY-SOURCE-AUDIT-001"],
            "findings": [], "contradictions": [], "recommendations": [],
            "uncertainties": [], "evidence_references": [],
        }
        valid = {"status": "completed", "rejected": []}
        with patch.object(
            reviewer,
            "_review_pass",
            side_effect=[(prior_review, "{}", valid, 1), (final_review, "{}", valid, 1)],
        ):
            result = reviewer.review_evidence(report, rules={"metrics": []}, max_iterations=2)
        self.assertEqual(result["status"], "completed_with_warnings")
        self.assertEqual(len(result["review"]["findings"]), 1)
        self.assertEqual(result["preserved_prior_findings"], 1)



    def test_later_explicit_rsi_contradiction_can_reject_prior_candidate(self) -> None:
        source = "int main(void) {\n    int value;\n    return value;\n}"
        report = _build_rsi_review_report(
            {"file_name": "reject.c"}, source, [], "completed",
            {"status": "completed", "audit": {"summary": "Uninitialized read.", "findings": [{
                "category": "Use of uninitialized variable", "severity": "medium",
                "line": 3, "code_excerpt": "return value;",
                "description": "value is read before initialization.",
            }]}},
        )
        first = {
            "summary": "The candidate is supported.",
            "summary_evidence_ids": ["E-MEMORY-AUDIT-FINDING-001"],
            "findings": [{
                "text": "Potential uninitialized read at line 3.", "kind": "potential",
                "evidence_ids": ["E-MEMORY-AUDIT-FINDING-001", "E-SOURCE-L003"],
            }],
            "contradictions": [], "recommendations": [], "uncertainties": [],
            "evidence_references": [],
        }
        final = {
            "summary": "The source line does not establish an uninitialized read.",
            "summary_evidence_ids": ["E-SOURCE-L003"],
            "findings": [],
            "contradictions": [{
                "text": "The uninitialized-read candidate is not supported by the cited source line.",
                "evidence_ids": ["E-MEMORY-AUDIT-FINDING-001", "E-SOURCE-L003"],
            }],
            "recommendations": [], "uncertainties": [], "evidence_references": [],
        }
        valid = {"status": "completed", "rejected": []}
        with patch.object(reviewer, "_review_pass", side_effect=[(first, "{}", valid, 1), (final, "{}", valid, 1)]):
            result = reviewer.review_evidence(report, rules={"metrics": []}, max_iterations=2)
        self.assertEqual(result["review"]["findings"], [])
        self.assertEqual(result["preserved_prior_findings"], 0)

    def test_rsi_uses_earlier_validated_findings_when_final_pass_fails(self) -> None:
        source = "\n".join(["int main(void) {", "    int value;", "    return value;", "}"])
        report = _build_rsi_review_report(
            {"file_name": "partial.c"}, source, [], "completed",
            {"status": "completed", "audit": {"summary": "Uninitialized read.", "findings": [{
                "category": "Use of uninitialized variable", "severity": "medium",
                "line": 3, "code_excerpt": "return value;",
                "description": "value is read before initialization.",
            }]}},
        )
        prior_review = {
            "summary": "An uninitialized read is supported by the source.",
            "summary_evidence_ids": ["E-MEMORY-SOURCE-AUDIT-001"],
            "findings": [{
                "text": "Potential use of uninitialized variable at line 3.",
                "kind": "potential",
                "evidence_ids": ["E-MEMORY-AUDIT-FINDING-001", "E-SOURCE-L003"],
            }],
            "contradictions": [], "recommendations": [], "uncertainties": [],
            "evidence_references": [],
        }
        valid = {"status": "completed", "rejected": []}
        invalid = {"status": "invalid", "reason": "simulated final pass failure"}
        with patch.object(
            reviewer,
            "_review_pass",
            side_effect=[(prior_review, "{}", valid, 1), (None, "bad json", invalid, 1)],
        ):
            result = reviewer.review_evidence(report, rules={"metrics": []}, max_iterations=2)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["claim_validation"]["status"], "completed_with_warnings")
        predicted, line, error_type, findings = _prediction_from_rsi_review(
            result, build_evidence_ledger(report, {"metrics": []})
        )
        self.assertTrue(predicted)
        self.assertEqual(line, 3)
        self.assertEqual(error_type, "Use of uninitialized variable")
        self.assertEqual(len(findings), 1)



if __name__ == "__main__":
    unittest.main()
