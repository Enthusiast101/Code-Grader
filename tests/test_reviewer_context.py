import json
import unittest
from unittest.mock import patch

import requests

from grading_agent.llm.reviewer import (
    MAX_EVIDENCE_CHARS,
    REVIEW_MAX_TOKENS,
    _prepare_evidence,
    _request_json,
    _source_chunks,
    _validate_memory_audit,
    review_c_vulnerabilities_source,
    review_evidence,
    evaluate_metrics_with_llm,
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
    def test_intermediate_failed_pass_is_not_hidden_by_successful_final_pass(self, request_json):
        valid = self._valid_review()
        request_json.side_effect = [
            (valid, json.dumps(valid)),
            (None, ""), (None, ""),
            (valid, json.dumps(valid)),
        ]
        result = review_evidence(self._report(), max_iterations=3)
        self.assertEqual(result["status"], "completed_with_warnings")
        self.assertEqual(result["final_synthesis_status"], "completed_with_warnings")
        self.assertEqual(result["request_attempts"], 4)
        self.assertEqual([p["iteration"] for p in result["failed_passes"]], [2])

    @patch("grading_agent.llm.reviewer._request_json")
    def test_failed_final_pass_falls_back_and_marks_partial(self, request_json):
        valid = self._valid_review()
        request_json.side_effect = [
            (valid, json.dumps(valid)),
            (None, ""), (None, ""),
        ]
        result = review_evidence(self._report(), max_iterations=2)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["final_synthesis_status"], "failed_using_previous_valid_review")
        self.assertIn("configured correctness test", result["review"]["summary"])

    @patch("grading_agent.llm.reviewer._request_json")
    def test_all_empty_responses_are_not_marked_completed(self, request_json):
        request_json.side_effect = [(None, ""), (None, "")]
        result = review_evidence(self._report(), max_iterations=1)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["final_synthesis_status"], "failed_no_valid_review")
        self.assertTrue(result["review"]["summary"])


    @patch("grading_agent.llm.reviewer._request_json")
    def test_metric_rsi_keeps_partial_scores_when_one_metric_is_missing(self, request_json):
        rules = {"problem": "sample", "metrics": [
            {"name": "correctness", "enabled": True, "weight": 0.5, "analyzer": "io_tests", "params": {}},
            {"name": "logical", "enabled": True, "weight": 0.5, "analyzer": "edge_case_runner", "params": {}},
        ]}
        report = self._report()
        report["logical"] = {"status": "unavailable", "score": None, "reason": "No runtime test."}
        payload = {"metrics": [{
            "name": "correctness", "score": 95, "rationale": "Configured case passed.",
            "basis": "analyzer_and_source", "evidence_ids": ["E-METRIC-CORRECTNESS-001"],
        }]}
        request_json.return_value = (payload, json.dumps(payload))
        result = evaluate_metrics_with_llm(report, rules)
        self.assertEqual(result["status"], "completed_with_warnings")
        self.assertEqual(result["metrics"][0]["score"], 95)
        self.assertEqual(result["missing_metrics"], ["logical"])

    @patch("grading_agent.llm.reviewer._request_json")
    def test_source_only_rsi_score_is_kept_when_metric_has_no_tool_evidence(self, request_json):
        rules = {"problem": "sample", "metrics": [
            {"name": "logical", "enabled": True, "weight": 1.0, "analyzer": "edge_case_runner", "params": {}}
        ]}
        report = self._report()
        report["logical"] = {"status": "unavailable", "score": None, "reason": "No dynamic test evidence."}
        payload = {"metrics": [{
            "name": "logical", "score": 84, "rationale": "Source inspection suggests the control flow is sound; runtime behavior was not tested.",
            "basis": "source_only", "evidence_ids": [],
        }]}
        request_json.return_value = (payload, json.dumps(payload))
        result = evaluate_metrics_with_llm(report, rules)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metrics"][0]["score"], 84)
        self.assertEqual(result["metrics"][0]["evidence_ids"], [])

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


    @patch("grading_agent.llm.reviewer._request_json")
    def test_rsi_uses_one_bounded_request_per_metric(self, request_json):
        rules = {"problem": "sample", "metrics": [
            {"name": "correctness", "enabled": True, "weight": 0.5, "analyzer": "io_tests", "params": {"cases": 2}},
            {"name": "logical", "enabled": True, "weight": 0.5, "analyzer": "edge_case_runner", "params": {}},
        ]}
        report = self._report()
        report["logical"] = {"status": "completed", "score": 90, "tests_passed": 2, "tests_total": 2}
        request_json.side_effect = [
            ({"name": "correctness", "score": 96, "rationale": "Tests passed.", "basis": "analyzer_and_source", "evidence_ids": ["E-CORRECTNESS-SUMMARY-001"]}, "{}"),
            ({"name": "logical", "score": 91, "rationale": "Control flow is sound.", "basis": "analyzer_and_source", "evidence_ids": []}, "{}"),
        ]

        result = evaluate_metrics_with_llm(report, rules)

        self.assertEqual(result["status"], "completed")
        self.assertEqual({item["name"] for item in result["metrics"]}, {"correctness", "logical"})
        self.assertEqual(request_json.call_count, 2)
        for call in request_json.call_args_list:
            messages = call.args[0]
            user_prompt = messages[1]["content"]
            self.assertLess(len(user_prompt) + len(messages[0]["content"]), 5_500)
            self.assertIn('"metric"', user_prompt)
            self.assertNotIn('"metrics_to_score"', user_prompt)
            self.assertEqual(call.kwargs["max_tokens"], 512)

    @patch("grading_agent.llm.reviewer._request_json")
    def test_context_overflow_retries_compact_metric_payload(self, request_json):
        rules = {"problem": "sample", "metrics": [
            {"name": "correctness", "enabled": True, "weight": 1.0, "analyzer": "io_tests", "params": {"many": list(range(30))}},
        ]}
        request_json.side_effect = [
            RuntimeError("request exceeds the available context size (4096 tokens)"),
            ({"name": "correctness", "score": 88, "rationale": "Compact retry succeeded.", "basis": "limited_evidence", "evidence_ids": []}, "{}"),
        ]

        result = evaluate_metrics_with_llm(self._report(), rules)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metrics"][0]["score"], 88)
        self.assertEqual(request_json.call_count, 2)

    def test_source_audit_requires_exact_source_excerpt(self):
        source = "int main(void) {\n    int value = 0;\n    return value;\n}\n"
        base = {"summary": "A source inspection was performed.", "limitations": [], "findings": [{
            "category": "uninitialized_read", "severity": "high", "line": 2,
            "function": "main", "code_excerpt": "int value = value;",
            "description": "Example finding", "evidence": "Exact line", "trigger_condition": "test", "recommendation": "review",
        }]}
        cleaned, error = _validate_memory_audit(base, source, 10)
        self.assertIsNone(error)
        self.assertEqual(cleaned["findings"], [])
        self.assertEqual(len(cleaned["rejected_findings"]), 1)
        self.assertIn("not present", cleaned["rejected_findings"][0]["reason"])

        clean_audit = {"summary": "No concrete memory defect was identified in this source snippet.", "findings": [], "limitations": ["Only supplied source was examined."]}
        cleaned, error = _validate_memory_audit(clean_audit, source, 10)
        self.assertIsNone(error)
        self.assertEqual(cleaned["findings"], [])

    @patch("grading_agent.llm.reviewer.MAX_SOURCE_CHARS", 26)
    @patch("grading_agent.llm.reviewer._request_json")
    def test_large_source_is_chunked_and_finding_lines_are_remapped(self, request_json):
        source = "\n".join(f"int value{index};" for index in range(8)) + "\n"
        chunks = _source_chunks(source, max_chars=26)
        responses = []
        for _, chunk in chunks:
            first_line = chunk.splitlines()[0]
            payload = {
                "summary": "A source chunk was inspected.",
                "limitations": [],
                "findings": [{
                    "category": "buffer_overflow", "severity": "low", "line": 1,
                    "code_excerpt": first_line, "description": "Synthetic test finding for line remapping.",
                }],
            }
            responses.append((payload, json.dumps(payload)))
        request_json.side_effect = responses
        result = review_c_vulnerabilities_source(source, max_findings=20)
        self.assertEqual(result["status"], "completed")
        self.assertGreater(result["chunks_total"], 1)
        self.assertEqual(result["chunks_completed"], result["chunks_total"])
        global_lines = {finding["line"] for finding in result["audit"]["findings"]}
        self.assertTrue(global_lines.issubset(set(range(1, 9))))
        self.assertGreater(len(global_lines), 1)

    def test_one_invalid_finding_does_not_discard_other_valid_findings(self):
        source = "int main(void) {\n    char buffer[4];\n    buffer[7] = 1;\n    return 0;\n}\n"
        audit = {
            "summary": "The source contains a possible out-of-bounds write.",
            "limitations": [],
            "findings": [
                {
                    "category": "out_of_bounds_write", "severity": "high", "line": 3,
                    "function": "main", "code_excerpt": "buffer[7] = 1;",
                    "description": "Write exceeds the declared array bounds.",
                },
                {
                    "category": "buffer_overflow", "severity": "high", "line": 3,
                    "function": "main", "code_excerpt": "buffer[99] = 1;",
                    "description": "This excerpt does not exist.",
                },
            ],
        }
        cleaned, error = _validate_memory_audit(audit, source, 10)
        self.assertIsNone(error)
        self.assertEqual(len(cleaned["findings"]), 1)
        self.assertEqual(len(cleaned["rejected_findings"]), 1)
        self.assertIn("omitted", cleaned["limitations"][-1])


if __name__ == "__main__":
    unittest.main()
