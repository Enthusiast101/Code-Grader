import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from grading_agent.benchmark.formai_predict import (
    _cppcheck_findings,
    _prediction_from_findings,
    _prediction_from_rsi_review,
    _select_hybrid_prediction,
    run,
)


class FormAIPredictionParsingTests(unittest.TestCase):
    def test_parses_vulnerability_related_cppcheck_diagnostic(self):
        output = (
            "candidate.c:42:9: warning: Array index out of bounds [arrayIndexOutOfBounds]\n"
            "candidate.c:12:2: style: unusedFunction [unusedFunction]\n"
        )
        findings = _cppcheck_findings(output)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["line"], 42)
        self.assertEqual(findings[0]["id"], "arrayIndexOutOfBounds")
        self.assertTrue(_prediction_from_findings(findings)[0])

    def test_rsi_prediction_requires_validated_vulnerability_claim_and_source_evidence(self):
        review = {
            "status": "completed",
            "claim_validation": {"status": "completed", "verified_claim_count": 1},
            "review": {"findings": [{
                "text": "Potential buffer overflow at the indexed write.",
                "kind": "potential",
                "evidence_ids": ["E-SOURCE-L004"],
            }]},
        }
        ledger = [{
            "id": "E-SOURCE-L004", "kind": "source_line",
            "statement": "Source line 4", "facts": {"line": 4, "code": "buffer[index] = value;"},
        }]
        self.assertEqual(
            _prediction_from_rsi_review(review, ledger)[:3],
            (True, 4, "source_review_finding"),
        )

    def test_rsi_does_not_accept_unreferenced_findings(self):
        review = {
            "status": "completed",
            "claim_validation": {"status": "completed", "verified_claim_count": 1},
            "review": {"findings": [{
                "text": "Potential buffer overflow at the indexed write.",
                "kind": "potential",
                "evidence_ids": ["E-UNKNOWN"],
            }]},
        }
        ledger = [{"id": "E-SOURCE-L004", "kind": "source_line", "facts": {"line": 4}}]
        self.assertEqual(_prediction_from_rsi_review(review, ledger)[:3], (False, None, None))

    def test_no_relevant_diagnostics_is_negative_prediction(self):
        findings = _cppcheck_findings(
            "candidate.c:12:2: style: unusedFunction [unusedFunction]\n"
        )
        self.assertEqual(findings, [])
        self.assertEqual(_prediction_from_findings(findings), (False, None, None))

    def test_hybrid_preserves_llm_prediction_when_docker_is_unavailable(self):
        audit = {"status": "completed", "audit": {"findings": [{
            "line": 8, "category": "buffer_overflow", "severity": "high",
            "code_excerpt": "buffer[index] = value;", "description": "Possible out-of-bounds write.",
        }]}}
        result = _select_hybrid_prediction(
            mode="hybrid", tool_status="unavailable", tool_findings=[], llm_audit=audit,
        )
        self.assertEqual(result, (True, 8, "buffer_overflow", "completed_with_warnings"))

    def test_hybrid_does_not_require_rsi_to_succeed(self):
        audit = {"status": "completed", "audit": {"findings": [{
            "line": 8, "category": "buffer_overflow", "severity": "high",
            "code_excerpt": "buffer[index] = value;", "description": "Possible out-of-bounds write.",
        }]}}
        result = _select_hybrid_prediction(
            mode="hybrid_rsi", tool_status="unavailable", tool_findings=[],
            llm_audit=audit, rsi_review={"status": "unavailable"}, rsi_findings=[],
        )
        self.assertEqual(result, (True, 8, "buffer_overflow", "completed_with_warnings"))

    def test_hybrid_emits_negative_result_when_available_methods_find_no_issue(self):
        audit = {"status": "completed", "audit": {"findings": []}}
        result = _select_hybrid_prediction(
            mode="hybrid", tool_status="unavailable", tool_findings=[], llm_audit=audit,
        )
        self.assertEqual(result, (False, None, None, "completed_with_warnings"))

    def test_hybrid_is_inconclusive_only_when_no_primary_stage_completed(self):
        result = _select_hybrid_prediction(
            mode="hybrid", tool_status="unavailable", tool_findings=[],
            llm_audit={"status": "unavailable"},
        )
        self.assertEqual(result, (None, None, None, "inconclusive"))

    def test_partial_cppcheck_finding_survives_tool_timeout(self):
        finding = {"line": 12, "id": "arrayIndexOutOfBounds", "description": "Index may exceed the array."}
        result = _select_hybrid_prediction(
            mode="hybrid", tool_status="inconclusive", tool_findings=[finding],
            llm_audit={"status": "unavailable"},
        )
        self.assertEqual(result, (True, 12, "arrayIndexOutOfBounds", "completed_with_warnings"))

    def test_hybrid_rsi_runs_when_only_cppcheck_is_available(self):
        finding = {"line": 2, "id": "arrayIndexOutOfBounds", "description": "Out-of-bounds access."}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / "inputs.jsonl"
            output = root / "predictions.jsonl"
            inputs.write_text(json.dumps({
                "file_name": "tool-only.c", "source_code": "int a[1];\nvoid f(void) { a[2] = 1; }\n",
            }) + "\n", encoding="utf-8")
            args = Namespace(
                inputs=inputs, mode="hybrid_rsi", output=output, max_samples=None,
                tool_timeout=2.0, max_llm_findings=10, rsi_iterations=1,
                memory_limit="512m",
            )
            with patch("grading_agent.benchmark.formai_predict._server_ready", return_value=True), \
                 patch("grading_agent.benchmark.formai_predict._run_tool_stage", return_value=([finding], "completed", None)), \
                 patch("grading_agent.benchmark.formai_predict._run_llm_stage", return_value={"status": "unavailable", "error": "mocked LLM outage"}), \
                 patch("grading_agent.benchmark.formai_predict.build_evidence_ledger", return_value=[]), \
                 patch("grading_agent.benchmark.formai_predict.review_evidence", return_value={"status": "completed", "claim_validation": {"status": "completed"}, "review": {"findings": []}}) as rsi:
                run(args)
            prediction = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
        rsi.assert_called_once()
        self.assertTrue(prediction["predicted_vulnerable"])
        self.assertEqual(prediction["prediction_status"], "completed_with_warnings")
        self.assertEqual(prediction["llm_audit_status"], "unavailable")
        self.assertIsNone(prediction["rsi_error"])
        self.assertIn("without additional validated", prediction["rsi_note"])


class FormAIResilienceTests(unittest.TestCase):
    @patch("grading_agent.benchmark.formai_predict._server_ready", return_value=True)
    @patch("grading_agent.benchmark.formai_predict.review_c_vulnerabilities_source")
    def test_combined_mode_runs_llm_and_writes_prediction_without_docker(self, llm_audit, _server_ready):
        llm_audit.return_value = {
            "status": "completed",
            "audit": {"summary": "Potential unsafe indexed write.", "findings": [{
                "line": 2, "category": "out_of_bounds_write", "severity": "high",
                "code_excerpt": "buffer[index] = value;",
                "description": "Potential out-of-bounds write.",
            }], "limitations": []},
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / "inputs.jsonl"
            output = root / "predictions.jsonl"
            inputs.write_text(json.dumps({
                "file_name": "sample.c",
                "source_code": "void f(int *buffer, int index, int value) {\n    buffer[index] = value;\n}\n",
            }) + "\n", encoding="utf-8")
            args = Namespace(
                inputs=inputs, mode="hybrid", output=output, max_samples=None,
                tool_timeout=2.0, max_llm_findings=10, rsi_iterations=1,
                memory_limit="512m",
            )
            run(args)
            prediction = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(prediction["predicted_vulnerable"])
        self.assertEqual(prediction["prediction_status"], "completed_with_warnings")
        self.assertEqual(prediction["prediction_source"], "llm")
        self.assertEqual(prediction["llm_audit_status"], "completed")
        llm_audit.assert_called_once()

    @patch("grading_agent.benchmark.formai_predict._server_ready", return_value=True)
    @patch("grading_agent.benchmark.formai_predict.review_c_vulnerabilities_source")
    def test_malformed_input_record_does_not_stop_following_records(self, llm_audit, _server_ready):
        llm_audit.return_value = {"status": "completed", "audit": {"summary": "No issues.", "findings": [], "limitations": []}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / "inputs.jsonl"
            output = root / "predictions.jsonl"
            good = {"file_name": "good.c", "source_code": "int main(void) { return 0; }"}
            inputs.write_text("not-json\n" + json.dumps(good) + "\n", encoding="utf-8")
            args = Namespace(
                inputs=inputs, mode="hybrid", output=output, max_samples=None,
                tool_timeout=2.0, max_llm_findings=10, rsi_iterations=1,
                memory_limit="512m",
            )
            run(args)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["prediction_status"], "inconclusive")
        self.assertEqual(rows[1]["file_name"], "good.c")
        self.assertEqual(rows[1]["predicted_vulnerable"], False)
        llm_audit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
