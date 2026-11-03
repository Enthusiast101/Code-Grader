"""Focused tests for the program-understanding + global RSI grading path."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from grading_agent.llm import reviewer  # noqa: E402
from grading_agent.llm import server as llm_server  # noqa: E402
from grading_agent.config import settings  # noqa: E402
from grading_agent.scoring.overall import apply_metric_scores, calculate_overall_score  # noqa: E402
from grading_agent.reporting.text_report import render_text_report  # noqa: E402


SOURCE = """#include <stdio.h>
int add(int a, int b) { return a + b; }
int main(void) {
    int a, b;
    if (scanf("%d %d", &a, &b) != 2) return 1;
    printf("%d\\n", add(a, b));
    return 0;
}
"""

RULES = {
    "problem": "Read two integers and print their sum.",
    "problem_contract": {"description": "Read two integers and print their sum."},
    "metrics": [
        {"name": "correctness", "analyzer": "io_tests", "weight": 0.6, "enabled": True, "params": {}},
        {"name": "syntax", "analyzer": "compiler_diagnostics", "weight": 0.4, "enabled": True, "params": {}},
    ],
}


def make_report() -> dict:
    return {
        "submission": "main.c",
        "problem": RULES["problem"],
        "source_code": SOURCE,
        "configuration": {
            "problem_contract": RULES["problem_contract"],
            "enabled_metrics": [
                {"name": "correctness", "analyzer": "io_tests", "weight": 0.6},
                {"name": "syntax", "analyzer": "compiler_diagnostics", "weight": 0.4},
            ],
        },
        "preflight": {"status": "completed", "compiled": True, "smoke_tested": True},
        "correctness": {
            "status": "completed", "score": 100, "tests_passed": 1, "tests_total": 1,
            "results": [{"test_id": 1, "passed": True, "input": "1 2", "expected": "3", "actual": "3"}],
        },
        "syntax": {"status": "completed", "score": 100, "compiled": True, "warning_count": 0, "error_count": 0},
    }


def understanding_response() -> dict:
    return {
        "actual_behavior": "Reads two integers and prints their sum.",
        "intended_behavior": "Read two integers and print their sum.",
        "contract_alignment": "The implementation matches the requested behavior.",
        "algorithm_overview": "Read two integers, add them, and print the result.",
        "functions": [
            {"name": "add", "role": "Returns the sum of its arguments.", "line_start": 2, "line_end": 2},
            {"name": "main", "role": "Reads input and prints the sum.", "line_start": 3, "line_end": 8},
        ],
        "data_structures": [],
        "execution_flow": ["Read input", "Call add", "Print result"],
        "assumptions": [],
        "ambiguities": [],
    }


def review_response(iteration: int) -> dict:
    response = {
        "overall_assessment": f"Whole-program assessment at pass {iteration}.",
        "findings": [
            {
                "metric": "correctness",
                "severity": "low",
                "title": f"Review note {iteration}",
                "description": f"Potential input-handling issue reviewed during pass {iteration}.",
                "line_start": 5,
                "line_end": 5,
                "evidence_ids": ["E-CORRECTNESS-SUMMARY-001"],
                "confidence": 0.6,
            }
        ],
        "unresolved_questions": [],
    }
    return response


def metric_score_response(correctness: int = 95, syntax: int = 94) -> dict:
    # The production scoring format is deliberately compact to avoid token-cap truncation.
    return {"scores": {"correctness": correctness, "syntax": syntax}}


class GlobalRSITests(unittest.TestCase):
    def test_understanding_three_chained_rsi_passes_then_separate_scoring(self) -> None:
        responses = [
            understanding_response(),
            review_response(1),
            review_response(2),
            review_response(3),
            metric_score_response(95, 94),
        ]
        captured: list[list[dict[str, str]]] = []

        def fake_request(messages, *, max_tokens=None, enable_thinking=None, response_schema=None, schema_name=None, **kwargs):
            captured.append(messages)
            self.assertIsNone(enable_thinking)  # Respect the globally configured reasoning mode.
            parsed = responses.pop(0)
            return parsed, json.dumps(parsed)

        report = make_report()
        with patch.object(reviewer, "_request_json", side_effect=fake_request):
            result = reviewer.evaluate_metrics_with_llm(report, RULES, max_iterations=3)

        self.assertEqual(result["status"], "completed_with_warnings")
        self.assertEqual(result["llm_calls"], 5)
        self.assertEqual(result["iterations_completed"], 3)
        self.assertEqual(len(captured), 5)
        self.assertIn("Whole-program assessment at pass 1.", captured[2][1]["content"])
        self.assertIn("Whole-program assessment at pass 2.", captured[3][1]["content"])
        self.assertIn("Do not provide metric scores", captured[1][1]["content"])
        self.assertIn("Do not provide metric scores", captured[3][1]["content"])
        self.assertIn("final_validated_findings", captured[4][1]["content"])
        self.assertEqual(result["metrics"][0]["score"], 95)
        self.assertEqual(result["metrics"][1]["score"], 94)
        self.assertEqual(report["program_understanding"]["functions"][0]["name"], "add")
        self.assertEqual(report["program_understanding"]["actual_behavior"], "Reads two integers and prints their sum.")
        self.assertEqual(report["program_understanding"]["intended_behavior"], "Read two integers and print their sum.")
        self.assertEqual(report["llm_rsi_findings"], [])
        self.assertTrue(report["llm_rsi_rejected_findings"])
        self.assertEqual(result["latency_seconds"]["metric_scoring_requests"], [0.0])

        apply_metric_scores(RULES["metrics"], report, result)
        overall = calculate_overall_score(RULES["metrics"], report)
        self.assertEqual(report["correctness"]["score"], 95)
        self.assertEqual(report["syntax"]["score"], 94)
        self.assertEqual(overall["score"], 94.6)

    def test_grading_request_can_explicitly_disable_reasoning(self) -> None:
        class FakeResponse:
            ok = True

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": '{"ok": true}'}}]}

        with patch.object(reviewer.requests, "post", return_value=FakeResponse()) as post:
            parsed, _ = reviewer._request_json(
                [{"role": "user", "content": "Return JSON."}],
                max_tokens=128,
                enable_thinking=False,
            )

        self.assertEqual(parsed, {"ok": True})
        request_body = post.call_args.kwargs["json"]
        self.assertEqual(request_body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(request_body["response_format"], {"type": "json_object"})
        self.assertEqual(request_body["reasoning_effort"], "none")
        self.assertEqual(request_body["reasoning_format"], "none")
        self.assertEqual(request_body["max_tokens"], 128)

    def test_grading_request_uses_prompt_schema_without_llama_grammar_by_default(self) -> None:
        class FakeResponse:
            ok = True

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": '{"findings": []}'}}]}

        schema = {
            "type": "object",
            "properties": {"findings": {"type": "array", "items": {"type": "string"}}},
            "required": ["findings"],
        }
        with patch.object(reviewer.requests, "post", return_value=FakeResponse()) as post:
            parsed, _ = reviewer._request_json(
                [{"role": "user", "content": "Return JSON."}],
                max_tokens=128,
                enable_thinking=False,
                response_schema=schema,
                schema_name="unit_test_schema",
            )

        self.assertEqual(parsed, {"findings": []})
        request_body = post.call_args.kwargs["json"]
        self.assertEqual(request_body["response_format"]["type"], "json_schema")
        self.assertEqual(request_body["response_format"]["schema"], schema)
        self.assertEqual(request_body["response_format"]["schema"], schema)
        self.assertNotIn("json_schema", request_body["response_format"])
        self.assertNotIn("grammar", request_body)
        self.assertIn("REQUIRED JSON OUTPUT CONTRACT", request_body["messages"][0]["content"])
        self.assertIn(json.dumps(schema, ensure_ascii=False, separators=(",", ":")), request_body["messages"][0]["content"])
        self.assertFalse(request_body["stream"])
        self.assertEqual(request_body["chat_template_kwargs"]["enable_thinking"], False)
        self.assertEqual(request_body["reasoning_effort"], "none")

    def test_structured_grading_respects_configured_reasoning_mode(self) -> None:
        responses = [understanding_response()]
        captured = []

        def fake_request(messages, *, max_tokens=None, enable_thinking=None, response_schema=None, schema_name=None, **kwargs):
            captured.append(enable_thinking)
            parsed = responses.pop(0)
            return parsed, json.dumps(parsed)

        with settings.runtime_overrides({"LLAMA_REASONING_ENABLED": True}), \
             patch.object(reviewer, "_request_json", side_effect=fake_request):
            understanding, metadata = reviewer.understand_program_with_llm(make_report(), RULES)

        self.assertIsNotNone(understanding)
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(captured, [None])

    def test_global_review_normalizer_accepts_common_aliases_and_empty_no_issue_review(self) -> None:
        empty = reviewer._normalise_global_review(
            {"overall_summary": "No defects were found in the submitted source."},
            RULES["metrics"], SOURCE, [], set(range(1, len(SOURCE.splitlines()) + 1)),
        )
        self.assertIsNotNone(empty)
        self.assertEqual(empty["findings"], [])

        aliased = reviewer._normalise_global_review(
            {
                "review": {
                    "summary": "A concrete issue was identified.",
                    "issues": [{
                        "category": "correctness",
                        "title": "Input validation issue",
                        "details": "The expected input is not fully checked.",
                        "line_number": 5,
                        "evidence_refs": ["E-CORRECTNESS-SUMMARY-001"],
                    }],
                }
            },
            RULES["metrics"], SOURCE,
            [{"id": "E-CORRECTNESS-SUMMARY-001", "kind": "metric_summary", "statement": "Correctness evidence."}],
            set(range(1, len(SOURCE.splitlines()) + 1)),
        )
        self.assertIsNotNone(aliased)
        self.assertEqual(aliased["findings"][0]["metric"], "correctness")
        self.assertEqual(aliased["findings"][0]["line_start"], 5)

    def test_schema_request_uses_response_format_once_without_grammar_retry(self) -> None:
        class FakeResponse:
            ok = True

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": '{"findings": []}'}}]}

        schema = {"type": "object", "properties": {"findings": {"type": "array"}}, "required": ["findings"]}
        with settings.runtime_overrides({"LLAMA_JSON_GRAMMAR_ENABLED": True, "LLAMA_REASONING_ENABLED": True}), \
             patch.object(reviewer.requests, "post", return_value=FakeResponse()) as post:
            parsed, _ = reviewer._request_json(
                [{"role": "user", "content": "Return JSON."}],
                max_tokens=128,
                enable_thinking=True,
                response_schema=schema,
                schema_name="compatibility_test",
            )
        self.assertEqual(parsed, {"findings": []})
        self.assertEqual(post.call_count, 1)
        request_body = post.call_args.kwargs["json"]
        self.assertEqual(request_body["response_format"]["type"], "json_schema")
        self.assertEqual(request_body["response_format"]["schema"], schema)
        self.assertTrue(request_body["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("grammar", request_body)

    def test_empty_final_content_explains_external_server_reasoning_mismatch(self) -> None:
        class FakeResponse:
            ok = True

            @staticmethod
            def json():
                return {
                    "choices": [{
                        "finish_reason": "length",
                        "message": {"content": "", "reasoning_content": "<think>analysis</think>"},
                    }],
                    "usage": {"completion_tokens": 128, "prompt_tokens": 64},
                }

        with patch.object(reviewer.requests, "post", return_value=FakeResponse()):
            parsed, diagnostic = reviewer._request_json(
                [{"role": "user", "content": "Return a JSON object."}],
                max_tokens=128,
                enable_thinking=False,
            )

        self.assertIsNone(parsed)
        self.assertIn("response_format.schema", diagnostic)
        self.assertIn("output budget", diagnostic)

    def test_managed_llama_server_starts_with_configured_reasoning(self) -> None:
        class FakeProcess:
            returncode = None

            @staticmethod
            def poll():
                return None

        class FakeHealthResponse:
            status_code = 200

        process = FakeProcess()
        version_result = type("VersionResult", (), {"returncode": 0, "stdout": "version 0.0.0 (build 12005)", "stderr": ""})()
        with settings.runtime_overrides({
            "LLAMA_REASONING_ENABLED": True,
            "LLAMA_REASONING_FORMAT": "deepseek",
            "LLAMA_REASONING_BUDGET": 768,
        }), patch.object(llm_server.shutil, "which", return_value="/usr/bin/llama-server"), \
             patch.object(llm_server.subprocess, "run", return_value=version_result), \
             patch.object(llm_server.subprocess, "Popen", return_value=process) as popen, \
             patch.object(llm_server.requests, "get", return_value=FakeHealthResponse()):
            returned = llm_server.initiate_server()

        self.assertIs(returned, process)
        command = popen.call_args.args[0]
        self.assertIn("--reasoning", command)
        self.assertEqual(command[command.index("--reasoning") + 1], "on")
        self.assertEqual(command[command.index("--reasoning-format") + 1], "deepseek")
        self.assertEqual(command[command.index("--reasoning-budget") + 1], "768")

    def test_failed_later_iteration_retains_findings_and_marks_partial(self) -> None:
        calls = 0

        def fake_request(messages, *, max_tokens=None, enable_thinking=None, response_schema=None, schema_name=None, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                parsed = understanding_response()
                return parsed, json.dumps(parsed)
            if calls == 2:
                parsed = review_response(1)
                return parsed, json.dumps(parsed)
            raise RuntimeError("simulated local server failure")

        report = make_report()
        with patch.object(reviewer.settings, "LLM_GRADING_RETRY_ATTEMPTS", 1), patch.object(reviewer, "_request_json", side_effect=fake_request):
            result = reviewer.evaluate_metrics_with_llm(report, RULES, max_iterations=3)

        self.assertEqual(result["status"], "completed_with_warnings")
        self.assertEqual(result["iterations_completed"], 3)
        self.assertEqual(result["iterations_with_fallback"], 2)
        self.assertEqual(len(result["metrics"]), 2)
        self.assertEqual(result["findings_status"], "completed_with_rejections")
        self.assertTrue(result["rejected_findings"])
        self.assertEqual(result["metric_scoring_status"], "completed_with_fallback")
        self.assertEqual(result["llm_calls"], 5)

    def test_metric_scoring_missing_metrics_falls_back_without_retry(self) -> None:
        partial_scores = {
            "scores": {
                "correctness": {
                    "score": 88,
                    "rationale": "Functional behavior matches the supplied contract.",
                    "zero_score_reason": "",
                    "confidence": 0.9,
                }
            }
        }
        responses = [understanding_response(), review_response(1), partial_scores]

        captured: list[list[dict[str, str]]] = []
        def fake_request(messages, *, max_tokens=None, enable_thinking=None, response_schema=None, schema_name=None, **kwargs):
            captured.append(messages)
            parsed = responses.pop(0)
            return parsed, json.dumps(parsed)

        report = make_report()
        with patch.object(reviewer.settings, "LLM_GRADING_RETRY_ATTEMPTS", 1), patch.object(reviewer, "_request_json", side_effect=fake_request):
            result = reviewer.evaluate_metrics_with_llm(report, RULES, max_iterations=1)

        self.assertEqual(result["status"], "completed_with_warnings")
        self.assertEqual(result["llm_calls"], 3)
        self.assertEqual(result["iterations_completed"], 1)
        self.assertEqual(result["metrics"][0]["score"], 88)
        self.assertEqual(len(captured), 3)
        self.assertTrue(any("Metric scoring response omitted" in item for item in result["diagnostics"]))

    def test_early_pass_can_be_valid_without_metric_scores(self) -> None:
        parsed = review_response(1)
        accepted = reviewer._normalise_global_review(
            parsed,
            RULES["metrics"],
            SOURCE,
            [{"id": "E-CORRECTNESS-SUMMARY-001"}, {"id": "E-SYNTAX-001"}],
            set(range(1, len(SOURCE.splitlines()) + 1)),
        )
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted["findings"][0]["metric"], "correctness")
        self.assertEqual(len(accepted["findings"]), 1)

    def test_structured_stage_retry_override_is_respected(self) -> None:
        with patch.object(reviewer.settings, "LLM_GRADING_RETRY_ATTEMPTS", 3), patch.object(
            reviewer, "_request_json", return_value=(None, "invalid JSON")
        ) as request:
            parsed, diagnostics, latencies, calls = reviewer._call_grading_json(
                [{"role": "user", "content": "Return JSON."}],
                max_tokens=100, schema_name="fast_validator", max_attempts=1,
            )
        self.assertIsNone(parsed)
        self.assertEqual(calls, 1)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(len(latencies), 1)
        self.assertTrue(diagnostics)

    def test_program_understanding_accepts_minimal_alias_response_and_optional_fields(self) -> None:
        schema = reviewer._program_understanding_response_schema()
        self.assertNotIn("actual_behavior", schema.get("required", []))
        normalized = reviewer._normalise_program_understanding(
            {"summary": "Reads N integers and prints their arithmetic mean."}, 1, {1}
        )
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["actual_behavior"], "Reads N integers and prints their arithmetic mean.")

    def test_integer_division_failure_is_recovered_from_source_and_exact_test_oracle(self) -> None:
        source = """#include <stdio.h>
int main(void) {
    int count;
    if (scanf("%d", &count) != 1 || count <= 0) return 1;
    long long sum = 0;
    for (int i = 0; i < count; ++i) {
        int value;
        if (scanf("%d", &value) != 1) return 1;
        sum += value;
    }
    double average = sum / count;
    printf("%.2f\n", average);
    return 0;
}
"""
        report = make_report()
        report["problem"] = "03_computation_average"
        report["source_code"] = source
        report["configuration"]["enabled_metrics"] = [
            {"name": "correctness", "analyzer": "io_tests", "weight": 1.0}
        ]
        report["correctness"] = {
            "status": "completed", "score": 0.0, "tests_passed": 0, "tests_total": 1,
            "results": [{
                "test_id": "fractional-average", "passed": False,
                "input": "4\n1 2 2 5\n", "expected": "2.50\n", "actual": "2.00\n",
            }],
        }
        rules = {"problem": "03_computation_average", "problem_contract": {}, "metrics": [
            {"name": "correctness", "enabled": True, "weight": 1.0, "params": {}}
        ]}
        ledger = reviewer.build_evidence_ledger(report, rules)
        recovered = reviewer._recover_integer_division_finding(source, ledger, rules["metrics"])
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["line_start"], 11)
        self.assertIn("E-CORRECTNESS-TEST-001", recovered["evidence_ids"])
        self.assertIn("E-SOURCE-L011", recovered["evidence_ids"])
        with patch.object(reviewer, "_request_json", side_effect=AssertionError("unneeded model adjudication")):
            accepted, rejected, metadata = reviewer._adjudicate_rsi_findings(
                [recovered], report=report, rules=rules,
                program_understanding=understanding_response(), ledger=ledger,
            )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(metadata["validation_mode"], "deterministic_source_test_corroboration")
        self.assertEqual(metadata["calls"], 0)

    def test_test_oracle_examples_are_available_when_assignment_is_only_a_slug(self) -> None:
        report = make_report()
        report["problem"] = "03_computation_average"
        report["correctness"]["results"] = [{
            "test_id": "fractional-average", "input": "4\n1 2 2 5\n",
            "expected": "2.50\n", "actual": "2.00\n", "passed": False,
        }]
        examples = reviewer._test_oracle_examples(report)
        self.assertEqual(examples[0]["expected_output"], "2.50\n")
        payload = reviewer._program_understanding_payload(report, RULES)
        self.assertTrue(payload["instructor_test_oracle_examples"])


    def test_program_understanding_accepts_small_model_behavior_aliases(self) -> None:
        schema = reviewer._program_understanding_response_schema()
        self.assertEqual(schema.get("type"), "object")
        self.assertTrue(schema.get("additionalProperties"))
        normalized = reviewer._normalise_program_understanding(
            {"program_understanding": {"actual_source_behavior": "Reads N values and prints their mean."}},
            1, {1},
        )
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["actual_behavior"], "Reads N values and prints their mean.")

    def test_integer_division_detector_handles_cast_after_quotient_in_printf(self) -> None:
        source = '''#include <stdio.h>
int main(void) {
    int count;
    long long sum = 0;
    double average;
    if (scanf("%d", &count) != 1) return 1;
    average = (double)(sum / count);
    printf("Average: %.2f\\n", (double)(sum / count));
    return 0;
}
'''
        self.assertIsNotNone(reviewer._integer_division_at_line(source, 7))
        self.assertIsNotNone(reviewer._integer_division_at_line(source, 8))
        # Casting an operand before division is correct and must not be flagged.
        correct_source = source.replace("(double)(sum / count)", "(double)sum / count")
        self.assertIsNone(reviewer._integer_division_at_line(correct_source, 7))
        self.assertIsNone(reviewer._integer_division_at_line(correct_source, 8))

    def test_integer_division_recovery_accepts_labeled_numeric_output(self) -> None:
        source = '''#include <stdio.h>
int main(void) {
    int count;
    long long sum = 0;
    double average;
    if (scanf("%d", &count) != 1) return 1;
    average = (double)(sum / count);
    printf("Average: %.2f\\n", average);
    return 0;
}
'''
        report = make_report()
        report["problem"] = "03_computation_average"
        report["source_code"] = source
        report["configuration"]["enabled_metrics"] = [{"name": "correctness", "analyzer": "io_tests", "weight": 1.0}]
        report["correctness"] = {
            "status": "completed", "score": 0.0, "tests_passed": 0, "tests_total": 1,
            "results": [{
                "test_id": "fractional-average", "passed": False,
                "input": "2\\n2 3\\n", "expected": "Average: 2.50\\n", "actual": "Average: 2.00\\n",
            }],
        }
        rules = {"problem": report["problem"], "problem_contract": {}, "metrics": [
            {"name": "correctness", "enabled": True, "weight": 1.0, "params": {}}
        ]}
        ledger = reviewer.build_evidence_ledger(report, rules)
        recovered = reviewer._recover_integer_division_finding(source, ledger, rules["metrics"])
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["line_start"], 7)
        self.assertIn("E-CORRECTNESS-TEST-001", recovered["evidence_ids"])

    def test_fallback_program_understanding_reports_actual_average_flow_and_partial_oracle(self) -> None:
        source = '''#include <stdio.h>
int main(void) {
    int count;
    if (scanf("%d", &count) != 1 || count <= 0) return 1;
    long long sum = 0;
    for (int i = 0; i < count; ++i) {
        int value;
        if (scanf("%d", &value) != 1) return 1;
        sum += value;
    }
    printf("Average: %.2f\\n", (double)(sum / count));
    return 0;
}
'''
        report = {
            "source_code": source, "problem": "03_computation_average",
            "correctness": {"results": [{"input": "2\\n2 3\\n", "expected": "2.50\\n", "actual": "2.00\\n", "passed": False}]},
        }
        fallback = reviewer._fallback_program_understanding(report, {"problem_contract": {}})
        self.assertIn("integer operands", fallback["actual_behavior"])
        self.assertIn("complete assignment statement is unavailable", fallback["intended_behavior"])

    def test_prevalidation_rejects_contract_limitations_contradictions_and_style_nits(self) -> None:
        source = """#include <stdio.h>
int main(void) {
    int count;
    if (scanf("%d", &count) != 1) return 1;
    if (count <= 0) return 1;
    for (int i = 0; i < count; ++i) { int value; if (scanf("%d", &value) != 1) return 1; }
    return 0;
}
"""
        findings = [
            {"id": "F01", "metric": "correctness", "title": "Empty Contract Ambiguity", "description": "The submission cannot be checked because no assignment contract or test cases were configured.", "line_start": 1},
            {"id": "F02", "metric": "logical", "title": "Missing scanf validation", "description": "The program does not check whether scanf read the initial count successfully.", "line_start": 4},
            {"id": "F03", "metric": "memory_safety", "title": "Loop variable bounds risk", "description": "The for-loop variable is not bounds-checked, so it may access invalid memory.", "line_start": 6},
            {"id": "F04", "metric": "logical", "title": "Unbounded loop risk", "description": "For count <= 0 the loop body is skipped, which is correct and causes no undefined behavior.", "line_start": 6},
            {"id": "F05", "metric": "readability", "title": "Variable naming", "description": "The variable names could be more descriptive.", "line_start": 3},
        ]
        cleaned, rejected = reviewer._prevalidate_rsi_findings(
            {"overall_assessment": "candidate findings", "findings": findings},
            source, {**RULES, "problem_contract": {}},
            {"problem": "grading", "correctness": {"tests_total": 0, "results": []}, "logical": {"tests_total": 0, "results": []}},
        )
        self.assertEqual(cleaned["findings"], [])
        self.assertEqual({item["finding_id"] for item in rejected}, {"F01", "F02", "F03", "F04", "F05"})

    def test_contract_alignment_does_not_keep_unsupported_partial_match(self) -> None:
        actual = (
            "The program reads an integer from stdin into count. If the input is not a single integer "
            "or count is non-positive, it returns 1. Otherwise, it reads count integers from stdin, "
            "accumulates their sum, computes the average, and prints it formatted to two decimal places."
        )
        intended = (
            "The program reads a single integer from stdin, validates it is positive, then reads count "
            "integers from stdin, accumulates the sum, computes the average, and prints the average "
            "formatted to two decimal places."
        )
        normalised = reviewer._normalise_program_understanding({
            "actual_behavior": actual,
            "intended_behavior": intended,
            "contract_alignment": "partial_match",
            "functions": [], "data_structures": [], "execution_flow": [], "assumptions": [], "ambiguities": [],
        }, 1, {1})
        self.assertTrue(normalised["contract_alignment"].startswith("aligned"))

    def test_correctness_and_logical_are_unmeasured_without_contract_or_tests(self) -> None:
        report = make_report()
        report["problem"] = "grading"
        report["correctness"] = {"status": "not_run", "score": None, "tests_total": 0, "results": []}
        report["logical"] = {"status": "not_run", "score": None, "tests_total": 0, "results": []}
        rules = {
            "problem": "grading", "problem_contract": {},
            "metrics": [
                {"name": "correctness", "enabled": True, "weight": 0.4},
                {"name": "logical", "enabled": True, "weight": 0.3},
                {"name": "syntax", "enabled": True, "weight": 0.3},
            ],
        }
        response = {"scores": {"correctness": 75, "logical": 80, "syntax": 100}}
        with patch.object(reviewer, "_request_json", return_value=(response, json.dumps(response))):
            scores, diagnostics, _latency, _calls = reviewer._score_metrics_after_rsi(
                report, rules, understanding_response(), {"findings": [], "overall_assessment": "No findings."}, [],
            )
        self.assertIsNone(scores["correctness"]["score"])
        self.assertIsNone(scores["logical"]["score"])
        self.assertEqual(scores["syntax"]["score"], 100)
        self.assertTrue(any("marked unmeasured" in message for message in diagnostics))

    def test_unavailable_finding_adjudicator_fails_closed(self) -> None:
        candidate = {
            "id": "F01", "metric": "correctness", "severity": "high",
            "title": "Wrong result", "description": "The code subtracts instead of adding.",
            "line_start": 3, "line_end": 3, "evidence_ids": [], "confidence": 0.9,
        }
        report = make_report()
        with patch.object(reviewer.settings, "LLM_GRADING_RETRY_ATTEMPTS", 2), \
             patch.object(reviewer, "_request_json", return_value=(None, "not valid JSON")) as request:
            accepted, rejected, metadata = reviewer._adjudicate_rsi_findings(
                [candidate], report=report, rules=RULES,
                program_understanding=understanding_response(), ledger=[],
            )
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(metadata["status"], "unavailable_fail_closed")
        self.assertEqual(metadata["calls"], 2)
        self.assertEqual(request.call_count, 2)

    def test_text_report_shows_rejected_findings_and_unmeasured_grade_as_na(self) -> None:
        report = {
            "problem": "grading", "submission": "main.c",
            "scoring": {"status": "failed_incomplete", "score": None, "warnings": ["Evidence unavailable."]},
            "configuration": {"enabled_metrics": [{"name": "correctness", "weight": 1.0}]},
            "correctness": {"score": None, "status": "inconclusive", "score_rationale": "No assignment contract or tests."},
            "llm_rsi_findings": [],
            "llm_rsi_rejected_findings": [{
                "metric": "correctness", "title": "Empty Contract Ambiguity", "line_start": 1,
                "reason": "An empty contract is a grading limitation, not a source defect.",
            }],
            "llm_rsi_evaluation": {"status": "completed_with_warnings", "llm_calls": 5, "latency_seconds": {}},
        }
        rendered = render_text_report(report)
        self.assertIn("Final grade: N/A", rendered)
        self.assertIn("EXCLUDED RSI CANDIDATES (NOT USED FOR SCORING)", rendered)
        self.assertNotIn("Final grade: 0.00/100", rendered)

    def test_only_adjudicated_findings_reach_metric_scoring(self) -> None:
        source = """#include <stdio.h>
int main(void) {
    int a, b;
    if (scanf("%d %d", &a, &b) != 2) return 1;
    printf("%d\n", a - b);
    return 0;
}
"""
        report = make_report()
        report["source_code"] = source
        report["correctness"]["results"] = [{"test_id": 1, "passed": False, "input": "4 1", "expected": "5", "actual": "3"}]
        finding_response = {
            "overall_assessment": "The source subtracts instead of adding the two required integers.",
            "findings": [{
                "metric": "correctness", "severity": "high", "title": "Subtraction used instead of addition",
                "description": "The assignment requires the sum of a and b, but line 5 computes a - b.",
                "line_start": 5, "line_end": 5, "evidence_ids": [], "confidence": 0.99,
            }],
            "unresolved_questions": [],
        }
        validation_response = {
            "decisions": [{
                "finding_id": "F01", "decision": "accept", "source_line_valid": True,
                "contract_relevant": True, "claim_supported": True, "logical_consistency": True,
                "evidence_valid": True,
                "reason": "Line 5 subtracts the operands while the assignment requires their sum.",
            }],
            "summary": "The finding is directly supported by source line 5 and the task contract.",
        }
        responses = [understanding_response(), finding_response, validation_response, metric_score_response(55, 100)]
        schemas_seen = []
        def fake_request(messages, *, max_tokens=None, enable_thinking=None, response_schema=None, schema_name=None, **kwargs):
            schemas_seen.append(schema_name)
            parsed = responses.pop(0)
            return parsed, json.dumps(parsed)

        with patch.object(reviewer, "_request_json", side_effect=fake_request):
            result = reviewer.evaluate_metrics_with_llm(report, RULES, max_iterations=1)
        self.assertEqual(schemas_seen[2], "post_rsi_finding_validation")
        self.assertEqual(result["findings"][0]["validation_status"], "accepted")
        self.assertEqual(result["finding_validation"]["accepted_count"], 1)
        self.assertEqual(result["llm_calls"], 4)
        self.assertEqual(result["status"], "completed")

    def test_unavailable_rsi_is_not_reported_as_no_defects(self) -> None:
        report = {
            "problem": "Sample C task",
            "submission": "main.c",
            "configuration": {"enabled_metrics": []},
            "scoring": {"status": "completed_with_warnings", "score": 50.0},
            "llm_rsi_evaluation": {
                "status": "unavailable",
                "error": "The model response did not match the required schema.",
                "llm_calls": 2,
                "latency_seconds": {"total_llm": 143.13, "program_understanding": 70.0, "rsi_requests": [73.13]},
            },
            "llm_rsi_findings": [],
        }
        rendered = render_text_report(report)
        self.assertIn("RSI findings unavailable", rendered)
        self.assertIn("does not mean the source is defect-free", rendered)
        self.assertNotIn("No concrete, supported defects were retained", rendered)
        self.assertIn("The model response did not match the required schema", rendered)



    def test_unmapped_contract_mismatch_is_mapped_to_correctness(self) -> None:
        source = "int main(void) { return 0; }\n"
        parsed = {
            "overall_assessment": "The submission does not implement the required task.",
            "findings": [{
                "metric": "unmapped",
                "title": "Incorrect program behavior",
                "description": "The program does not read two integers or compute their sum as required by the assignment contract.",
                "severity": "critical",
                "line_start": 1,
                "line_end": 1,
                "evidence_ids": [],
                "confidence": 0.99,
            }],
        }
        visible = {1}
        normalized = reviewer._normalise_global_review(
            parsed, RULES["metrics"], source, [], visible
        )
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["findings"][0]["metric"], "correctness")
        self.assertEqual(normalized["findings"][0]["reported_metric"], "unmapped")

    def test_gcc_analyzer_parser_extracts_use_after_free_location(self) -> None:
        output = (
            "/workspace/main.c:13:18: warning: pointer 'values' used after 'free' [-Wuse-after-free]\n"
            "/workspace/main.c:13:18: warning: use after 'free' of 'values' [CWE-416] [-Wanalyzer-use-after-free]\n"
        )
        from grading_agent.analyzers.memory_safety import _parse_gcc_analyzer_output
        findings = _parse_gcc_analyzer_output(output)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["category"], "use_after_free")
        self.assertEqual(findings[0]["line"], 13)

    def test_evidence_ledger_includes_gcc_and_valgrind_memory_findings(self) -> None:
        report = make_report()
        report["configuration"]["enabled_metrics"].append(
            {"name": "memory_safety", "analyzer": "memory_safety", "weight": 1.0}
        )
        report["memory_safety"] = {
            "status": "completed",
            "score": 0,
            "tests_total": 1,
            "tests_clean": 0,
            "memory_safe": False,
            "gcc_analyzer": {
                "status": "completed",
                "findings": [{
                    "category": "use_after_free", "severity": "warning", "line": 13,
                    "column": 18, "message": "pointer values used after free",
                }],
            },
            "runtime_tools": {
                "status": "completed",
                "valgrind_results": [{
                    "test_id": "case-1", "analysis_error": False, "memory_safe": False,
                    "parsed_summary": {
                        "error_count": 1,
                        "definitely_lost": {"bytes": 32, "blocks": 1},
                        "indirectly_lost": {"bytes": 0, "blocks": 0},
                        "possibly_lost": {"bytes": 0, "blocks": 0},
                        "diagnostics": ["Invalid read of size 4"],
                    },
                }],
                "asan_ubsan_results": [],
            },
        }
        rules = {
            "metrics": [{"name": "memory_safety", "enabled": True, "analyzer": "memory_safety", "weight": 1.0, "params": {}}],
            "problem_contract": {},
        }
        ledger = reviewer.build_evidence_ledger(report, rules)
        ids = {item["id"] for item in ledger}
        self.assertIn("E-GCC-ANALYZER-001", ids)
        self.assertIn("E-VALGRIND-TEST-001", ids)


    def test_llm_unavailable_still_returns_complete_fallback_evaluation(self) -> None:
        report = make_report()
        with patch.object(reviewer.settings, "LLM_GRADING_RETRY_ATTEMPTS", 1), patch.object(
            reviewer, "_request_json", side_effect=RuntimeError("simulated server outage")
        ):
            result = reviewer.evaluate_metrics_with_llm(report, RULES, max_iterations=2)

        self.assertEqual(result["status"], "completed_with_warnings")
        self.assertEqual(result["iterations_completed"], 2)
        self.assertEqual(len(result["metrics"]), len(RULES["metrics"]))
        self.assertTrue(all(0 <= item["score"] <= 100 for item in result["metrics"]))
        self.assertTrue(all(item["origin"] != "llm" for item in result["metrics"]))
        self.assertEqual(result["program_understanding"]["status"], "fallback")

    def test_context_budget_is_capped_and_large_source_is_compacted(self) -> None:
        captured = {}
        long_source = "\\n".join(f"{index:05d} | int value_{index} = {index};" for index in range(1, 400))
        payload = {
            "task": "Review the submitted source.",
            "source_line_count": 399,
            "source_was_truncated": False,
            "source_visible_line_ranges": [[1, 399]],
            "source_code_with_line_numbers": long_source,
            "program_understanding": {"actual_behavior": "A" * 2000},
            "rubric_and_analyzer_results": [],
            "evidence_ledger": [{"id": "E-1", "kind": "source_line", "statement": "source line", "facts": {"line": 2}}],
            "review_constraints": ["Be concise."],
        }

        class FakeResponse:
            ok = True
            reason = "OK"
            text = ""

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": '{"ok": true}'}}]}

        def fake_post(url, *, json, timeout):
            captured["body"] = json
            return FakeResponse()

        with patch.object(reviewer.settings, "LLAMA_CONTEXT_SIZE", 4096), patch.object(
            reviewer.requests, "post", side_effect=fake_post
        ):
            parsed, _ = reviewer._request_json(
                [
                    {"role": "system", "content": "Return one JSON object."},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                max_tokens=3800,
                enable_thinking=False,
            )

        self.assertEqual(parsed, {"ok": True})
        self.assertLess(captured["body"]["max_tokens"], 3800)
        request_user = captured["body"]["messages"][1]["content"]
        self.assertLess(len(request_user), len(json.dumps(payload)))
        self.assertIn("source_visible_line_ranges", request_user)


    def test_large_output_budget_is_capped_to_small_context_without_overflow(self) -> None:
        payload = {
            "task": "Review source.",
            "source_code_with_line_numbers": "\n".join(
                f"{line:05d} | int value_{line} = {line};" for line in range(1, 2500)
            ),
            "source_visible_line_ranges": [[1, 2499]],
            "source_line_count": 2499,
            "rubric_metrics": [{"name": "correctness", "weight": 1.0}],
        }
        captured = {}

        class FakeResponse:
            ok = True
            reason = "OK"
            text = ""

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": '{"ok": true}'}}]}

        def fake_post(url, *, json, timeout):
            captured["body"] = json
            return FakeResponse()

        with patch.object(reviewer.settings, "LLAMA_CONTEXT_SIZE", 1024), patch.object(
            reviewer.requests, "post", side_effect=fake_post
        ):
            parsed, _diagnostic = reviewer._request_json(
                [
                    {"role": "system", "content": "Review the C program and return JSON. " + ("Verbose instruction. " * 900)},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                max_tokens=4096,
                enable_thinking=False,
            )
        self.assertEqual(parsed, {"ok": True})
        self.assertLess(captured["body"]["max_tokens"], 4096)
        self.assertEqual(captured["body"]["response_format"], {"type": "json_object"})

    def test_same_model_history_compression_preserves_finding_anchors(self) -> None:
        from grading_agent.llm import context_compressor

        review = {
            "overall_assessment": "A long assessment. " * 150,
            "findings": [{
                "metric": "memory_safety", "severity": "critical", "title": "Use after free",
                "description": "A detailed explanation. " * 80, "line_start": 42, "line_end": 42,
                "evidence_ids": ["E-GCC-ANALYZER-001"], "confidence": 0.99,
            }],
            "unresolved_questions": ["Check the path."],
        }

        class FakeResponse:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": json.dumps({
                    "overall_assessment": "One verified memory-safety defect.",
                    "finding_descriptions": {"finding_0": "Pointer is dereferenced after free."},
                })}}]}

        with settings.runtime_overrides({
            "LLM_CONTEXT_COMPRESSION_THRESHOLD_CHARS": 20,
            "LLM_CONTEXT_COMPRESSION_ENABLED": True,
        }), patch.object(context_compressor.requests, "post", return_value=FakeResponse()) as post:
            compressed, metadata = context_compressor.compress_review_history(review)

        self.assertEqual(metadata["method"], "same_model")
        self.assertEqual(compressed["findings"][0]["line_start"], 42)
        self.assertEqual(compressed["findings"][0]["evidence_ids"], ["E-GCC-ANALYZER-001"])
        self.assertLess(metadata["output_chars"], metadata["input_chars"])
        self.assertTrue(post.called)

    def test_same_model_compression_failure_uses_deterministic_history_compression(self) -> None:
        from grading_agent.llm import context_compressor

        review = {
            "overall_assessment": "long " * 100,
            "findings": [{
                "metric": "correctness", "severity": "high", "title": "Wrong output",
                "description": "detailed " * 100, "line_start": 9, "line_end": 10,
                "evidence_ids": ["E-TEST-001"],
            }],
        }
        with settings.runtime_overrides({
            "LLM_CONTEXT_COMPRESSION_THRESHOLD_CHARS": 10,
            "LLM_CONTEXT_COMPRESSION_ENABLED": True,
        }), patch.object(
            context_compressor.requests, "post", side_effect=RuntimeError("grading model is offline")
        ):
            compressed, metadata = context_compressor.compress_review_history(review)
        self.assertEqual(metadata["method"], "deterministic_fallback")
        self.assertEqual(compressed["findings"][0]["line_start"], 9)
        self.assertEqual(compressed["findings"][0]["evidence_ids"], ["E-TEST-001"])


    def test_increasing_output_budget_does_not_shrink_source_context(self) -> None:
        payload = {
            "task": "Review this C source.",
            "source_code_with_line_numbers": "\n".join(
                f"{line:05d} | int value_{line} = {line};" for line in range(1, 350)
            ),
            "source_visible_line_ranges": [[1, 349]],
            "source_line_count": 349,
            "evidence_ledger": [{"id": "E-1", "kind": "source_line", "statement": "line evidence"}],
            "program_understanding": {"actual_behavior": "A concise overview."},
            "review_constraints": ["Inspect the source carefully."],
        }
        messages = [
            {"role": "system", "content": "Return a JSON object."},
            {"role": "user", "content": json.dumps(payload)},
        ]
        with patch.object(reviewer.settings, "LLAMA_CONTEXT_SIZE", 8192):
            small_prompt, small_budget, small_stats = reviewer._fit_grading_messages(messages, 256)
            large_prompt, large_budget, large_stats = reviewer._fit_grading_messages(messages, 4096)

        # The user request is a ceiling: a larger ceiling must not cause more
        # aggressive source compaction. The available output budget can grow.
        self.assertEqual(small_prompt[1]["content"], large_prompt[1]["content"])
        self.assertGreaterEqual(large_budget, small_budget)
        self.assertTrue(small_stats["prompt_fits_context_estimate"])
        self.assertTrue(large_stats["prompt_fits_context_estimate"])

    def test_semantic_same_model_compression_preserves_functions_and_analyzer_values(self) -> None:
        from grading_agent.llm import context_compressor

        payload = {
            "program_understanding": {
                "actual_behavior": "The source behavior is verbose. " * 80,
                "intended_behavior": "The task contract is verbose. " * 80,
                "functions": [{"name": "main", "role": "entry point", "line_start": 7, "line_end": 21}],
                "source_line_count": 30,
            },
            "rubric_and_analyzer_results": [{
                "name": "memory_safety", "analyzer": "memory_safety", "weight": 0.2,
                "analyzer_summary": {"score": 0.0, "finding_count": 2, "summary": "A" * 200},
            }],
            "evidence_ledger": [{"id": "E-MEM-1", "kind": "static_finding", "statement": "use after free"}],
        }

        class FakeResponse:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": json.dumps({
                    "program_understanding_prose": {
                        "actual_behavior": "Allocates and frees an array, then uses it.",
                        "intended_behavior": "Read two integers and print their sum.",
                    }
                })}}]}

        with settings.runtime_overrides({"LLM_CONTEXT_COMPRESSION_ENABLED": True}), patch.object(
            context_compressor.requests, "post", return_value=FakeResponse()
        ):
            compressed, metadata = context_compressor.compress_semantic_sections(payload, target_chars=500)

        self.assertEqual(metadata["method"], "same_model")
        self.assertEqual(compressed["program_understanding"]["functions"], payload["program_understanding"]["functions"])
        self.assertEqual(compressed["rubric_and_analyzer_results"], payload["rubric_and_analyzer_results"])
        self.assertEqual(compressed["evidence_ledger"], payload["evidence_ledger"])


    def test_report_json_sanitizer_produces_strict_json(self) -> None:
        from pathlib import Path
        from grading_agent.main import _json_compatible

        value = _json_compatible({
            "score": float("nan"),
            "path": Path("reports/sample.json"),
            "nested": {"items": [float("inf"), 3]},
        })
        serialized = json.dumps(value, allow_nan=False)
        self.assertIn('"score": null', serialized)
        self.assertIn('"reports/sample.json"', serialized)
        self.assertIn('"items": [null, 3]', serialized)

    def test_text_report_has_last_resort_formatting_fallback(self) -> None:
        malformed_report = {
            "problem": "sample",
            "configuration": {"enabled_metrics": [None]},
            "scoring": {"score": object(), "status": "completed"},
        }
        rendered = render_text_report(malformed_report)
        self.assertIn("Code Checking Assistant", rendered)
        self.assertIn("REPORT FORMATTING FALLBACK", rendered)


if __name__ == "__main__":
    unittest.main()
