import json
import unittest
from unittest.mock import patch

import requests

from grading_agent.llm.reviewer import (
    MAX_EVIDENCE_CHARS,
    REVIEW_MAX_TOKENS,
    _prepare_evidence,
    _request_json,
    _validate_memory_audit,
    review_evidence,
)


class ReviewerContextTests(unittest.TestCase):
    def _report(self):
        return {
            "submission": "main.c",
            "problem": "sample",
            "source_code": "int main(void) { return 0; }",
            "configuration": {"enabled_metrics": [{"name": "correctness"}]},
            "correctness": {
                "status": "completed", "score": 100, "weight": 1.0,
                "tests_passed": 1, "tests_total": 1,
                "results": [{"test_id": 1, "passed": True, "input": "3 4", "expected": "7", "actual": "7"}],
            },
        }

    def _valid_review(self, evidence_id="E-METRIC-CORRECTNESS-001"):
        return {
            "summary": "The configured correctness test passed.",
            "summary_evidence_ids": [evidence_id],
            "findings": [],
            "contradictions": [],
            "recommendations": [],
            "uncertainties": [],
            "evidence_references": [evidence_id],
        }

    def test_evidence_payload_is_bounded_and_does_not_mutate_raw_logs(self):
        long_log = "==123== Memcheck header\n" + ("irrelevant detail\n" * 500) + "==123== ERROR SUMMARY: 0 errors\n"
        report = self._report()
        report["memory_safety"] = {
            "status": "completed", "score": 100, "tests_total": 4, "tests_clean": 4,
            "runtime_tools": {"status": "completed", "score": 100, "valgrind_results": [
                {"test_id": 1, "memory_safe": True, "captured_output": long_log, "raw_log": long_log,
                 "parsed_summary": {"error_count": 0, "all_heap_blocks_freed": True}}
            ]},
        }
        prepared = _prepare_evidence(report)
        encoded = json.dumps(prepared, ensure_ascii=False)
        self.assertLessEqual(len(encoded), MAX_EVIDENCE_CHARS)
        self.assertEqual(report["memory_safety"]["runtime_tools"]["valgrind_results"][0]["raw_log"], long_log)
        self.assertNotIn("raw_log", encoded)
        self.assertTrue(prepared["llm_input_metadata"]["full_logs_preserved_in_report"])

    @patch("grading_agent.llm.reviewer._request_json")
    def test_intermediate_failed_pass_is_recovered_by_valid_final_pass(self, request_json):
        valid = self._valid_review()
        request_json.side_effect = [
            (valid, json.dumps(valid)),
            (None, ""), (None, ""),
            (valid, json.dumps(valid)),
        ]
        result = review_evidence(self._report(), max_iterations=3)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["llm_status"], "completed")
        self.assertEqual(result["final_synthesis_status"], "llm_advisory_completed")
        self.assertEqual(result["request_attempts"], 4)
        self.assertEqual(result["failed_passes"], [])
        self.assertIsNone(result["llm_error"])
        self.assertEqual([p["iteration"] for p in result["recovered_passes"]], [2])

    @patch("grading_agent.llm.reviewer._request_json")
    def test_failed_final_pass_falls_back_and_marks_partial(self, request_json):
        valid = self._valid_review()
        from grading_agent.llm.reviewer import build_deterministic_review_fallback
        report = self._report()
        expected_deterministic_review = build_deterministic_review_fallback(report)["review"]
        request_json.side_effect = [
            (valid, json.dumps(valid)),
            (None, ""), (None, ""),
        ]
        result = review_evidence(report, max_iterations=2)
        self.assertEqual(result["status"], "completed_with_warnings")
        self.assertEqual(result["final_synthesis_status"], "previous_valid_review_used")
        self.assertEqual(result["llm_status"], "partial")
        self.assertEqual(result["review_engine"], "python_deterministic_evidence_review")
        self.assertEqual(result["review"], expected_deterministic_review)
        self.assertIn("configured correctness test", result["llm_advisory"]["summary"])

    @patch("grading_agent.llm.reviewer._request_json")
    def test_all_empty_responses_are_not_marked_completed(self, request_json):
        request_json.side_effect = [(None, ""), (None, "")]
        result = review_evidence(self._report(), max_iterations=1)
        self.assertEqual(result["status"], "completed_with_warnings")
        self.assertEqual(result["final_synthesis_status"], "deterministic_fallback")
        self.assertEqual(result["review_engine"], "python_deterministic_evidence_review")
        self.assertEqual(result["llm_status"], "unavailable")
        self.assertTrue(result["review"]["summary"])
        self.assertEqual(result["claim_validation"]["status"], "completed")
        self.assertGreater(result["claim_validation"]["verified_claim_count"], 0)
        self.assertIsNone(result["llm_advisory"])

    @patch("grading_agent.llm.reviewer._request_json")
    def test_primary_review_is_identical_when_llm_is_unavailable(self, request_json):
        request_json.return_value = (None, "")
        report = self._report()
        first = review_evidence(report, max_iterations=1)
        second = review_evidence(report, max_iterations=1)
        self.assertEqual(first["review_engine"], "python_deterministic_evidence_review")
        self.assertEqual(second["review_engine"], "python_deterministic_evidence_review")
        self.assertEqual(first["review"], second["review"])
        self.assertEqual(first["claim_validation"]["status"], "completed")
        self.assertEqual(second["claim_validation"]["status"], "completed")
        self.assertEqual(first["llm_status"], "unavailable")
        self.assertEqual(second["llm_status"], "unavailable")

    def test_deterministic_fallback_rebuilds_ledger_and_reports_metric_status(self):
        from grading_agent.llm.reviewer import build_deterministic_review_fallback
        report = self._report()
        # The basic I/O analyzer may omit a status field but has complete test evidence.
        report["correctness"].pop("status", None)
        report["evidence_ledger"] = [{"id": "STALE", "kind": "metric_summary", "facts": {}}]
        result = build_deterministic_review_fallback(report, reason="mock model failure")
        ledger = {item["id"]: item for item in report["evidence_ledger"]}
        self.assertEqual(ledger["E-METRIC-CORRECTNESS-001"]["facts"]["status"], "completed")
        self.assertIn("E-CORRECTNESS-SUMMARY-001", result["review"]["summary_evidence_ids"])
        self.assertEqual(result["claim_validation"]["status"], "completed")
        self.assertEqual(result["review_engine"], "python_deterministic_evidence_review")

    def test_compacted_prompt_keeps_aggregate_test_summaries(self):
        from grading_agent.llm.evidence import build_evidence_ledger
        from grading_agent.llm.reviewer import _compact_ledger
        report = self._report()
        report["configuration"]["enabled_metrics"] = [
            {"name": "correctness"}, {"name": "logical"}
        ]
        report["logical"] = {
            "score": 100, "tests_passed": 2, "tests_total": 2, "compiled": True,
            "results": [
                {"test_id": "E1", "passed": True, "input": "2147483647 1", "expected": "2147483648", "actual": "2147483648", "tags": ["signed_int32_overflow"]},
                {"test_id": "E2", "passed": True, "input": "-2147483648 -1", "expected": "-2147483649", "actual": "-2147483649", "tags": ["signed_int32_overflow"]},
            ],
        }
        ledger = build_evidence_ledger(report, {"problem_contract": {"operand_type": "int32", "sum_type": "int64"}})
        ids = {item["id"] for item in _compact_ledger(ledger)}
        self.assertIn("E-CORRECTNESS-SUMMARY-001", ids)
        self.assertIn("E-LOGICAL-SUMMARY-001", ids)
        self.assertIn("E-LOGICAL-OVERFLOW-COVERAGE-001", ids)

    @patch("grading_agent.llm.reviewer.requests.post")
    def test_http_error_includes_server_response_body(self, post):
        response = requests.Response()
        response.status_code = 400
        response._content = b'{"error":{"message":"request exceeds available context"}}'
        response.url = "http://127.0.0.1:8080/v1/chat/completions"
        post.return_value = response
        with self.assertRaisesRegex(RuntimeError, "request exceeds available context"):
            _request_json([{"role": "user", "content": "test"}])

    @patch("grading_agent.llm.reviewer.requests.post")
    def test_requests_constrained_json_output(self, post):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"choices": [{"message": {"content": json.dumps({"summary": "ok"})}}]}).encode()
        response.url = "http://127.0.0.1:8080/v1/chat/completions"
        post.return_value = response
        parsed, _ = _request_json([{"role": "user", "content": "Return JSON."}], max_tokens=128)
        self.assertEqual(parsed, {"summary": "ok"})
        self.assertEqual(post.call_args.kwargs["json"]["response_format"], {"type": "json_object"})

    def test_memory_audit_rewrites_summary_when_all_model_findings_are_rejected(self):
        source = "int main(void) { return 0; }\n"
        audit = {
            "summary": "Memory-safety issues found in the C source code.",
            "limitations": [],
            "findings": [{
                "category": "use_after_free", "severity": "high", "line": 1,
                "function": "main", "code_excerpt": "int main(void)",
                "description": "Possible use after free.",
                "evidence": "No deallocation lifecycle is present.",
                "trigger_condition": "Any call to main.",
                "recommendation": "Inspect object lifetime.",
            }],
        }
        cleaned, error = _validate_memory_audit(audit, source, 10)
        self.assertIsNone(error)
        self.assertEqual(cleaned["findings"], [])
        self.assertEqual(len(cleaned["rejected_findings"]), 1)
        self.assertIn("No validated memory-safety defects", cleaned["summary"])

    def test_source_audit_requires_exact_source_excerpt(self):
        source = "int main(void) {\n    int value = 0;\n    return value;\n}\n"
        base = {"summary": "A source inspection was performed.", "limitations": [], "findings": [{
            "category": "uninitialized_read", "severity": "high", "line": 2,
            "function": "main", "code_excerpt": "int value = value;",
            "description": "Example finding", "evidence": "Exact line", "trigger_condition": "test", "recommendation": "review",
        }]}
        cleaned, error = _validate_memory_audit(base, source, 10)
        self.assertIsNone(cleaned)
        self.assertIn("not present", error)

        clean_audit = {"summary": "No concrete memory defect was identified in this source snippet.", "findings": [], "limitations": ["Only supplied source was examined."]}
        cleaned, error = _validate_memory_audit(clean_audit, source, 10)
        self.assertIsNone(error)
        self.assertEqual(cleaned["findings"], [])


if __name__ == "__main__":
    unittest.main()
