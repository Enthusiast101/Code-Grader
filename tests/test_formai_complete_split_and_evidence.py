from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from grading_agent.benchmark.formai_evaluate import evaluate_mode
from grading_agent.benchmark.formai_manifest import validate_benchmark_split
from grading_agent.benchmark.formai_predict import _prediction_from_rsi_review
from grading_agent.ui import app as ui_app
from grading_agent.config import settings
from unittest.mock import patch


class FormAICompleteSplitTests(unittest.TestCase):
    def _write(self, path: Path, text: str) -> Path:
        path.write_text(text, encoding="utf-8")
        return path

    def test_missing_96_predictions_report_four_percent_coverage_and_no_final_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = [
                {"file_name": f"p{i}.c", "ground_truth_category": "VULNERABLE" if i < 50 else "SAFE"}
                for i in range(100)
            ]
            predictions = [
                {"file_name": f"p{i}.c", "predicted_vulnerable": (i < 3)}
                for i in range(4)
            ]
            label_path = self._write(root / "labels.json", json.dumps(labels))
            prediction_path = self._write(
                root / "predictions.jsonl", "".join(json.dumps(item) + "\n" for item in predictions)
            )
            result = evaluate_mode(label_path, prediction_path)
            self.assertEqual(result["missing_predictions"], 96)
            self.assertEqual(result["matched_prediction_records"], 4)
            self.assertEqual(result["prediction_coverage"], 0.04)
            self.assertFalse(result["metrics_finalized"])
            self.assertIsNone(result["recall"])
            self.assertIsNotNone(result["observed_subset_metrics"]["recall"])
            # Missing records are not silently counted as false negatives.
            self.assertEqual(sum(result["confusion_matrix"].values()), 4)

    def test_full_prediction_set_finalizes_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = [
                {"file_name": f"p{i}.c", "ground_truth_category": "VULNERABLE" if i < 2 else "SAFE"}
                for i in range(4)
            ]
            predictions = [
                {"file_name": f"p{i}.c", "predicted_vulnerable": (i == 0)}
                for i in range(4)
            ]
            label_path = self._write(root / "labels.json", json.dumps(labels))
            prediction_path = self._write(
                root / "predictions.jsonl", "".join(json.dumps(item) + "\n" for item in predictions)
            )
            result = evaluate_mode(label_path, prediction_path)
            self.assertTrue(result["metrics_finalized"])
            self.assertEqual(result["prediction_coverage"], 1.0)
            self.assertEqual(result["recall"], 0.5)
            self.assertEqual(result["accuracy"], 0.75)

    def test_split_preflight_rejects_four_sources_against_100_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._write(
                root / "inputs.jsonl",
                "".join(json.dumps({"file_name": f"p{i}.c", "source_code": "int main(void){return 0;}"}) + "\n" for i in range(4)),
            )
            labels = self._write(
                root / "labels.json",
                json.dumps([{"file_name": f"p{i}.c", "ground_truth_category": "SAFE"} for i in range(100)]),
            )
            with self.assertRaisesRegex(ValueError, "not the same split"):
                validate_benchmark_split(inputs, labels)


class StrictRsiSourceLineTests(unittest.TestCase):
    def _review(self, evidence_ids):
        return {
            "status": "completed",
            "claim_validation": {"status": "completed"},
            "review": {"findings": [{
                "text": "Potential buffer overflow at the cited source operation.",
                "kind": "potential",
                "evidence_ids": evidence_ids,
            }]},
        }

    def test_rsi_claim_is_anchored_through_cited_candidate_source_line(self):
        ledger = [
            {"id": "E-MODEL-001", "kind": "model_audit_finding", "facts": {"line": 8, "category": "Buffer Overflow", "code_excerpt": "strcpy(dst, src);"}},
            {"id": "E-SOURCE-L008", "kind": "source_line", "facts": {"line": 8, "code": "strcpy(dst, src);"}},
        ]
        # Citing the candidate alone resolves its line in the ledger.
        result = _prediction_from_rsi_review(self._review(["E-MODEL-001"]), ledger)
        self.assertTrue(result[0])
        self.assertEqual(result[1], 8)
        result = _prediction_from_rsi_review(self._review(["E-MODEL-001", "E-SOURCE-L008"]), ledger)
        self.assertTrue(result[0])
        self.assertEqual(result[1], 8)

    def test_rsi_candidate_without_matching_source_line_is_rejected(self):
        no_line = [
            {"id": "E-MODEL-001", "kind": "model_audit_finding", "facts": {"line": 8, "category": "Buffer Overflow", "code_excerpt": "strcpy(dst, src);"}},
        ]
        self.assertFalse(_prediction_from_rsi_review(self._review(["E-MODEL-001"]), no_line)[0])
        wrong_excerpt = no_line + [
            {"id": "E-SOURCE-L008", "kind": "source_line", "facts": {"line": 8, "code": "int n = 0;"}},
        ]
        self.assertFalse(_prediction_from_rsi_review(self._review(["E-MODEL-001"]), wrong_excerpt)[0])

    def test_rsi_rejects_model_excerpt_that_does_not_match_cited_source_line(self):
        ledger = [
            {"id": "E-MODEL-001", "kind": "model_audit_finding", "facts": {
                "line": 12, "category": "Buffer Overflow", "code_excerpt": "recv(fd, buf, 256, 0);"
            }},
            {"id": "E-SOURCE-L012", "kind": "source_line", "facts": {
                "line": 12, "code": "#include <stdio.h>"
            }},
        ]
        self.assertFalse(_prediction_from_rsi_review(self._review(["E-MODEL-001", "E-SOURCE-L012"]), ledger)[0])

    def test_rsi_rejects_wrong_line_and_include_directive(self):
        ledger = [
            {"id": "E-MODEL-001", "kind": "model_audit_finding", "facts": {"line": 12, "category": "Buffer Overflow"}},
            {"id": "E-SOURCE-L012", "kind": "source_line", "facts": {"line": 12, "code": "#include <stdio.h>"}},
        ]
        self.assertFalse(_prediction_from_rsi_review(self._review(["E-MODEL-001", "E-SOURCE-L012"]), ledger)[0])
        mismatched = [
            {"id": "E-MODEL-001", "kind": "model_audit_finding", "facts": {"line": 12, "category": "Buffer Overflow"}},
            {"id": "E-SOURCE-L013", "kind": "source_line", "facts": {"line": 13, "code": "int x = 0;"}},
        ]
        self.assertFalse(_prediction_from_rsi_review(self._review(["E-MODEL-001", "E-SOURCE-L013"]), mismatched)[0])


class GradioFullSplitTests(unittest.TestCase):
    def test_gradio_always_runs_full_matching_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "test_inputs.jsonl"
            labels = root / "test_labels.json"
            inputs.write_text(
                "".join(json.dumps({"file_name": f"p{i}.c", "source_code": "int main(void){return 0;}"}) + "\n" for i in range(2)),
                encoding="utf-8",
            )
            labels.write_text(json.dumps([
                {"file_name": "p0.c", "ground_truth_category": "VULNERABLE"},
                {"file_name": "p1.c", "ground_truth_category": "SAFE"},
            ]), encoding="utf-8")
            output_root = root / "runs"
            seen = {}

            def fake_predict(args):
                seen["max_samples"] = args.max_samples
                with Path(args.output).open("w", encoding="utf-8") as stream:
                    for i in range(2):
                        stream.write(json.dumps({"file_name": f"p{i}.c", "predicted_vulnerable": False,
                                                 "llm_audit_status": "completed", "tool_status": "completed",
                                                 "rsi_review_status": "completed", "prediction_status": "completed"}) + "\n")

            with patch.object(settings, "FORM_AI_GRADIO_RUNS_DIR", output_root), patch.object(
                ui_app, "run_formai_prediction", side_effect=fake_predict
            ):
                summary, prediction_path, evaluation_path = ui_app.run_formai_combined(
                    str(inputs), str(labels), "hybrid_rsi", 20, 10, 2, "512m"
                )
            self.assertIsNone(seen["max_samples"])
            self.assertIn("Benchmark status:** COMPLETE", summary)
            evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
            self.assertEqual(evaluation["evaluation"]["matched_prediction_records"], 2)
            self.assertTrue(evaluation["evaluation"]["metrics_finalized"])
            self.assertEqual(len(Path(prediction_path).read_text(encoding="utf-8").splitlines()), 2)

    def test_gradio_sample_count_runs_and_evaluates_matching_random_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "test_inputs.jsonl"
            labels = root / "test_labels.json"
            records = [
                {"file_name": f"p{i}.c", "source_code": f"int main(void) {{ return {i}; }}"}
                for i in range(10)
            ]
            inputs.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            labels.write_text(json.dumps([
                {"file_name": f"p{i}.c", "ground_truth_category": "VULNERABLE" if i < 5 else "SAFE"}
                for i in range(10)
            ]), encoding="utf-8")
            output_root = root / "runs"
            seen = {}

            def fake_predict(args):
                selected = [json.loads(line) for line in Path(args.inputs).read_text(encoding="utf-8").splitlines()]
                seen["names"] = [item["file_name"] for item in selected]
                seen["input_count"] = len(selected)
                with Path(args.output).open("w", encoding="utf-8") as stream:
                    for item in selected:
                        stream.write(json.dumps({
                            "file_name": item["file_name"], "predicted_vulnerable": False,
                            "llm_audit_status": "completed", "tool_status": "completed",
                            "rsi_review_status": "completed", "prediction_status": "completed",
                        }) + "\n")

            with patch.object(settings, "FORM_AI_GRADIO_RUNS_DIR", output_root), patch.object(
                ui_app, "run_formai_prediction", side_effect=fake_predict
            ):
                summary, prediction_path, evaluation_path = ui_app.run_formai_combined(
                    str(inputs), str(labels), "hybrid_rsi", 20, 10, 2, "512m", 4
                )
            self.assertEqual(seen["input_count"], 4)
            evaluation_payload = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
            self.assertEqual(evaluation_payload["sample_selection"]["selected_count"], 4)
            self.assertEqual(evaluation_payload["evaluation"]["ground_truth_records"], 4)
            self.assertTrue(evaluation_payload["evaluation"]["metrics_finalized"])
            self.assertEqual(len(Path(prediction_path).read_text(encoding="utf-8").splitlines()), 4)
            self.assertIn("4 / 10 available", summary)

    def test_gradio_rejects_mismatched_input_and_label_files_before_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "test_inputs.jsonl"
            labels = root / "test_labels.json"
            inputs.write_text(json.dumps({"file_name": "p0.c", "source_code": "int main(void){return 0;}"}) + "\n", encoding="utf-8")
            labels.write_text(json.dumps([
                {"file_name": "p0.c", "ground_truth_category": "SAFE"},
                {"file_name": "p1.c", "ground_truth_category": "VULNERABLE"},
            ]), encoding="utf-8")
            with patch.object(ui_app, "run_formai_prediction") as runner:
                summary, prediction_path, evaluation_path = ui_app.run_formai_combined(
                    str(inputs), str(labels), "hybrid_rsi", 20, 10, 2, "512m"
                )
            runner.assert_not_called()
            self.assertIn("not the same split", summary)
            self.assertIsNone(prediction_path)
            self.assertIsNone(evaluation_path)


if __name__ == "__main__":
    unittest.main()
