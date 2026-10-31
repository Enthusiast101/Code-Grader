import json
import unittest
from unittest.mock import patch

import requests

from grading_agent.llm.reviewer import (
    MAX_EVIDENCE_CHARS,
    REVIEW_MAX_TOKENS,
    _prepare_evidence,
    _request_json,
    review_evidence,
)


class ReviewerContextTests(unittest.TestCase):
    def test_evidence_payload_has_a_real_hard_limit(self):
        long_log = "==123== Memcheck header\n" + ("irrelevant detail\n" * 500) + "==123== ERROR SUMMARY: 0 errors\n"
        report = {
            "submission": "main.c",
            "problem": "large_report",
            "source_code": "int main(void) { return 0; }",
            "correctness": {
                "status": "completed",
                "score": 100,
                "tests_passed": 4,
                "tests_total": 4,
                "results": [{"test_id": i, "passed": True, "actual": "ok"} for i in range(4)],
            },
            "memory_safety": {
                "status": "completed",
                "score": 100,
                "tests_total": 4,
                "tests_clean": 4,
                "runtime_tools": {
                    "status": "completed",
                    "score": 100,
                    "valgrind_results": [
                        {
                            "test_id": i,
                            "memory_safe": True,
                            "captured_output": long_log,
                            "raw_log": long_log,
                            "parsed_summary": {"error_count": 0, "all_heap_blocks_freed": True},
                        }
                        for i in range(4)
                    ],
                },
            },
            "scoring": {
                "status": "completed",
                "score": 100,
                "weights_sum": 1,
                "contributions": [
                    {"metric": f"metric_{i}", "weight": 0.1, "score": 100, "contribution": 10}
                    for i in range(9)
                ],
            },
        }

        prepared = _prepare_evidence(report)
        encoded = json.dumps(prepared, ensure_ascii=False)

        self.assertLessEqual(len(encoded), MAX_EVIDENCE_CHARS)
        self.assertEqual(report["memory_safety"]["runtime_tools"]["valgrind_results"][0]["raw_log"], long_log)
        self.assertNotIn("raw_log", encoded)
        self.assertTrue(prepared["llm_input_metadata"]["full_logs_preserved_in_report"])

    @patch("grading_agent.llm.reviewer._request_json")
    def test_three_pass_review_uses_bounded_requests(self, request_json):
        request_json.side_effect = [
            ({"summary": "initial", "findings": []}, "{}"),
            ({"summary": "audited", "findings": []}, "{}"),
            ({"summary": "final", "findings": []}, "{}"),
        ]
        report = {
            "source_code": "int main(void) { return 0; }",
            "correctness": {"score": 100, "tests_passed": 2, "tests_total": 2},
            "memory_safety": {"status": "completed", "score": 100},
        }
        result = review_evidence(report, max_iterations=3)
        self.assertEqual(result["review_iterations"], 3)
        self.assertEqual(request_json.call_count, 3)
        for call in request_json.call_args_list:
            self.assertLessEqual(call.kwargs["max_tokens"], REVIEW_MAX_TOKENS)
            user_message = call.args[0][1]["content"]
            self.assertLess(len(user_message), 6_000)

    @patch("grading_agent.llm.reviewer.requests.post")
    def test_http_error_includes_server_response_body(self, post):
        response = requests.Response()
        response.status_code = 400
        response._content = b'{"error":{"message":"request exceeds available context"}}'
        response.url = "http://127.0.0.1:8080/v1/chat/completions"
        post.return_value = response

        with self.assertRaisesRegex(RuntimeError, "request exceeds available context"):
            _request_json([{"role": "user", "content": "test"}])


if __name__ == "__main__":
    unittest.main()
