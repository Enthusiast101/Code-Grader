"""Small targeted regression tests for requested final-grader behavior."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from grading_agent.config import settings  # noqa: E402
from grading_agent.llm import context_compressor, reviewer, server  # noqa: E402
from grading_agent.ui.app import build_app  # noqa: E402
from grading_agent.reporting.text_report import render_text_report  # noqa: E402
from grading_agent.scoring.overall import apply_metric_scores, calculate_overall_score  # noqa: E402
from grading_agent.ui.app import _metric_summary  # noqa: E402


class _Response:
    ok = True
    status_code = 200
    reason = "OK"
    text = ""

    def __init__(self, payload=None):
        self.payload = payload or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class TestFinalUserFixes(unittest.TestCase):
    def test_llm_score_is_authoritative_over_analyzer_score(self):
        metrics = [{"name": "correctness", "weight": 1.0, "enabled": True}]
        report = {"correctness": {"score": 0.0, "status": "inconclusive"}}
        llm = {
            "metrics": [{
                "name": "correctness", "score": 67.0,
                "rationale": "The source implements the described core behavior, with some unverified edge cases.",
                "origin": "llm",
            }]
        }
        apply_metric_scores(metrics, report, llm)
        overall = calculate_overall_score(metrics, report)
        self.assertEqual(report["correctness"]["score"], 67.0)
        self.assertEqual(overall["score"], 67.0)
        self.assertEqual(report["correctness"]["llm_rsi_score"], 67.0)
        self.assertEqual(report["correctness"]["score_origin"], "llm")
        self.assertNotIn("scoring_basis", report["correctness"])
        self.assertNotIn("scoring_basis", overall["contributions"][0])

    def test_inconclusive_zero_analyzer_score_is_not_used_as_a_zero(self):
        metrics = [{"name": "correctness", "weight": 1.0, "enabled": True}]
        report = {"correctness": {"score": 0.0, "status": "inconclusive"}}
        apply_metric_scores(metrics, report, {"metrics": []})
        overall = calculate_overall_score(metrics, report)
        self.assertIsNone(report["correctness"]["score"])
        self.assertEqual(report["correctness"]["score_origin"], "unmeasured")
        self.assertIsNone(overall["score"])
        self.assertEqual(overall["status"], "failed_incomplete")
        self.assertTrue(overall["warnings"])

    def test_inconclusive_analyzer_does_not_create_severity_based_zero_when_llm_score_is_missing(self):
        fallback = reviewer._fallback_metric_score(
            "correctness",
            {"correctness": {"score": 0.0, "status": "inconclusive"}},
            [{"metric": "correctness", "severity": "critical", "title": "Possible contract mismatch", "evidence_ids": ["E-1"]}],
        )
        self.assertIsNone(fallback["score"])
        self.assertEqual(fallback["origin"], "insufficient_evidence_fallback")
        self.assertIn("No score is assigned", fallback["rationale"])

    def test_generic_metric_rationale_is_replaced_without_discarding_score(self):
        enabled = [{"name": "syntax", "enabled": True}]
        # Scores are 0-10 rubric levels; Python converts them to points.
        generic = {"scores": {"syntax": {
            "score": 10,
            "rationale": "No metric-specific defect was retained; available analyzer evidence supports this score.",
        }}}
        specific = {"scores": {"syntax": {
            "score": 10,
            "rationale": "GCC compiled the source with the configured C11, Wall, Wextra, and Wpedantic flags and reported no diagnostics.",
        }}}
        generic_result = reviewer._normalise_metric_scores(generic, enabled, [])
        self.assertEqual(generic_result["syntax"]["score"], 100.0)
        self.assertIn("did not provide a metric-specific explanation", generic_result["syntax"]["rationale"])
        self.assertIn("syntax", reviewer._normalise_metric_scores(specific, enabled, []))

    def test_valid_llm_score_is_not_discarded_when_rationale_is_generic(self):
        parsed = {"scores": {"syntax": {
            "score": 9.6,
            "rationale": "No metric-specific defect was retained; available analyzer evidence supports this score.",
            "zero_score_reason": "",
            "confidence": 0.8,
        }}}
        result = reviewer._normalise_metric_scores(parsed, [{"name": "syntax"}], [])
        self.assertEqual(result["syntax"]["score"], 96.0)
        self.assertEqual(result["syntax"]["origin"], "llm")
        self.assertLessEqual(len(result["syntax"]["rationale"]), 160)

    def test_scoring_reads_source_and_preserves_valid_llm_zero_without_retry(self):
        source = "int main(void) { puts(\"hello\"); return 0; }\n"
        rules = {
            "problem": "Read one integer and print the integer back unchanged.",
            "problem_contract": {"requirements": ["Read one integer and print it unchanged."]},
            "metrics": [{"name": "correctness", "enabled": True, "weight": 1.0, "analyzer": "io_tests", "params": {}}],
        }
        report = {"problem": "Read one integer and print the integer back unchanged.", "source_code": source, "correctness": {"tests_total": 0, "status": "inconclusive"}}
        first = {"scores": {"correctness": {
            "score": 0,
            "rationale": "The source violates the explicitly required input and output behavior and produces none of the required result.",
            "zero_score_reason": "The only required output is absent and the required input is never read.",
            "confidence": 0.95,
        }}}
        calls = []

        def fake_request(messages, **kwargs):
            calls.append((messages, kwargs))
            return first, "{}"

        with settings.runtime_overrides({
            "LLAMA_REASONING_ENABLED": True,
            "LLM_GLOBAL_SCORE_MAX_TOKENS": 2048,
            "LLM_GLOBAL_SOURCE_MAX_CHARS": 8000,
            "LLM_GLOBAL_MAX_FINDINGS": 12,
        }), patch.object(reviewer, "_request_json", side_effect=fake_request):
            scores, diagnostics, _latencies, request_count = reviewer._score_metrics_after_rsi(
                report, rules, {}, {"findings": [], "overall_assessment": "No concrete issues found."}, []
            )

        self.assertEqual(request_count, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(scores["correctness"]["score"], 0.0)
        self.assertEqual(scores["correctness"]["origin"], "llm")
        self.assertIn("source_code_with_line_numbers", calls[0][0][1]["content"])
        self.assertIsNone(calls[0][1]["enable_thinking"])
        self.assertEqual(diagnostics, [])

    def test_compression_calls_same_model_with_direct_json_schema_mode(self):
        mock_response = _Response({"choices": [{"message": {"content": '{"summary":"compressed"}'}}]})
        with settings.runtime_overrides({
            "MODEL": "configured-grading-model",
            "LLAMA_REASONING_ENABLED": True,
            "LLAMA_REASONING_FORMAT": "deepseek",
            "LLM_CONTEXT_COMPRESSION_ENABLED": True,
            "LLM_CONTEXT_COMPRESSION_MAX_TOKENS": 512,
        }), patch.object(context_compressor.requests, "post", return_value=mock_response) as post:
            result = context_compressor._same_model_chat(
                "Compress faithfully.", {"summary": "long text"},
                {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
            )
        self.assertEqual(result, {"summary": "compressed"})
        url, = post.call_args.args
        body = post.call_args.kwargs["json"]
        self.assertEqual(url, f"{settings.LLAMA_BASE_URL}{settings.LLAMA_API_PATH}")
        self.assertEqual(body["model"], "configured-grading-model")
        # Structured compression matches grading requests: schema from the first token.
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("grammar", body)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["response_format"]["json_schema"]["schema"]["type"], "object")
        self.assertNotIn("schema", body["response_format"])  # top-level schema is ignored by llama-server

    def test_managed_server_rejects_pre_json_schema_fix_build_before_start(self):
        old_version = MagicMock(returncode=0, stdout="version: 0.0.0 (build 11999, commit oldbuild)\n", stderr="")
        with patch.object(server.shutil, "which", return_value="/usr/bin/llama-server"), \
             patch.object(server.subprocess, "run", return_value=old_version), \
             patch.object(server.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "PR #31380"):
                server.initiate_server()
        popen.assert_not_called()

    def test_managed_server_accepts_known_pr31380_commit_even_with_custom_build_number(self):
        fixed_version = MagicMock(returncode=0, stdout="version: 0.0.0 (build 1, commit 3c809f44)\n", stderr="")
        with patch.object(server.subprocess, "run", return_value=fixed_version):
            detected = server._verify_json_schema_compatible_server("/usr/bin/llama-server")
        self.assertIn("3c809f44", detected)

    def test_managed_server_rejects_unverifiable_build_by_default(self):
        unknown_version = MagicMock(returncode=0, stdout="custom llama server build\n", stderr="")
        with patch.object(server.shutil, "which", return_value="/usr/bin/llama-server"), \
             patch.object(server.subprocess, "run", return_value=unknown_version):
            with self.assertRaisesRegex(RuntimeError, "LLAMA_ALLOW_UNVERIFIED_SERVER"):
                server.initiate_server()

    def test_managed_server_starts_with_reasoning_enabled_and_budget(self):
        process = MagicMock()
        process.poll.return_value = None
        health = MagicMock(status_code=200)
        version = MagicMock(returncode=0, stdout="version: 0.0.0 (build 12005, commit 3c809f44)\n", stderr="")
        with settings.runtime_overrides({
            "MODEL": "configured-grading-model",
            "LLAMA_REASONING_ENABLED": True,
            "LLAMA_REASONING_FORMAT": "deepseek",
            "LLAMA_REASONING_BUDGET": 768,
        }), patch.object(server.shutil, "which", return_value="/usr/bin/llama-server"), \
             patch.object(server.subprocess, "run", return_value=version), \
             patch.object(server.subprocess, "Popen", return_value=process) as popen, \
             patch.object(server.requests, "get", return_value=health):
            server.initiate_server()
        command = popen.call_args.args[0]
        self.assertIn("--reasoning", command)
        self.assertEqual(command[command.index("--reasoning") + 1], "on")
        self.assertEqual(command[command.index("--reasoning-format") + 1], "deepseek")
        self.assertEqual(command[command.index("--reasoning-budget") + 1], "768")

    def test_student_reports_remove_score_basis_and_generic_rsi_boilerplate(self):
        report = {
            "problem": "grading",
            "configuration": {"enabled_metrics": [{"name": "correctness", "weight": 1.0}]},
            "correctness": {
                "score": 64.0,
                "status": "completed",
                "score_rationale": "The source has coherent behavior, but the assignment contract is underspecified.",
            },
            "scoring": {"status": "completed", "score": 64.0, "contributions": [{"metric": "correctness", "score": 64.0, "weight": 1.0, "contribution": 64.0}]},
            "llm_rsi_evaluation": {"status": "completed", "overall_assessment": "The source was reviewed against the available task details."},
            "llm_rsi_findings": [],
        }
        markdown = _metric_summary(report)
        text = render_text_report(report)
        forbidden = (
            "Score basis: LLM RSI score based on rubric, source code, and analyzer evidence",
            "RSI: No metric-specific defect was retained; available analyzer evidence supports this score.",
        )
        for phrase in forbidden:
            self.assertNotIn(phrase, markdown)
            self.assertNotIn(phrase, text)
        self.assertIn("Assessment:", markdown)
        self.assertIn("Assessment:", text)

    def test_grading_request_passes_thinking_flags_to_the_same_model(self):
        mock_response = _Response({"choices": [{"message": {"content": '{"ok":true}'}}]})
        with settings.runtime_overrides({
            "MODEL": "configured-grading-model",
            "LLAMA_REASONING_ENABLED": True,
            "LLAMA_REASONING_FORMAT": "deepseek",
        }), patch.object(reviewer.requests, "post", return_value=mock_response) as post:
            parsed, _raw = reviewer._request_json(
                [{"role": "system", "content": "Return JSON."}, {"role": "user", "content": "{}"}],
                max_tokens=512,
                enable_thinking=True,
            )
        body = post.call_args.kwargs["json"]
        self.assertEqual(parsed, {"ok": True})
        self.assertEqual(body["model"], "configured-grading-model")
        self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(body["reasoning_format"], "deepseek")

    def test_gradio_app_builds_with_same_model_controls(self):
        app = build_app()
        self.assertIsNotNone(app)

    def test_default_reasoning_and_same_model_compression_are_enabled(self):
        self.assertTrue(settings.LLAMA_REASONING_ENABLED)
        self.assertTrue(settings.LLM_CONTEXT_COMPRESSION_ENABLED)


if __name__ == "__main__":
    unittest.main(verbosity=2)

class TestStructuredLLMRetries(unittest.TestCase):
    def test_structured_request_retries_invalid_json_then_accepts_repaired_response(self):
        responses = [
            (None, "not-json"),
            ({"ok": True}, '{"ok":true}'),
        ]
        with patch.object(reviewer, "_request_json", side_effect=responses) as request:
            parsed, diagnostics, latencies, calls = reviewer._call_grading_json(
                [{"role": "system", "content": "Return JSON."}],
                max_tokens=128,
                response_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
                schema_name="retry_regression",
                validator=lambda value: value.get("ok") is True,
            )
        self.assertEqual(parsed, {"ok": True})
        self.assertEqual(calls, 2)
        self.assertEqual(len(latencies), 2)
        self.assertTrue(any("attempt 1/3" in item for item in diagnostics))
        self.assertEqual(request.call_args_list[0].kwargs["enable_thinking"], None)

    def test_score_rationale_preserves_three_sentences(self):
        from grading_agent.reporting.rationale import compact_rationale
        rationale = "The parser reads the required integer. The loop accumulates values in a 64-bit sum. The final division preserves fractional output."
        self.assertEqual(len(compact_rationale(rationale).split(". ")), 3)
