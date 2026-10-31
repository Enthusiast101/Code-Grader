from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grading_agent.benchmark import formai_sample
from grading_agent.benchmark.formai_evaluate import evaluate_mode


class FormAISamplingAggregationTests(unittest.TestCase):
    def test_multiple_vulnerability_rows_are_merged_per_program(self):
        records = [
            {"file_name": "vuln.c", "source_code": "int main(void) { return 0; }", "category": "VULNERABLE", "vulnerable_line": 10, "error_type": "buffer overflow"},
            {"file_name": "vuln.c", "source_code": "int main(void) { return 0; }", "category": "VULNERABLE", "vulnerable_line": 20, "error_type": "null pointer dereference"},
            {"file_name": "safe.c", "source_code": "int main(void) { return 0; }", "category": "NON-VULNERABLE", "vulnerable_line": -1, "error_type": "N/A"},
            {"file_name": "timeout.c", "source_code": "int main(void) { return 0; }", "category": "UNKNOWN (time out)"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            db = sqlite3.connect(Path(temp_dir) / "test.sqlite3")
            try:
                formai_sample._create_database(db)
                with patch.object(formai_sample, "iter_records", return_value=iter(records)):
                    counts = formai_sample._aggregate_records(Path(temp_dir) / "unused.json", db)
                self.assertEqual(counts["records_seen"], 4)
                self.assertEqual(counts["unique_programs"], 3)
                self.assertEqual(formai_sample._program_class(1, 0, 0), "VULNERABLE")
                self.assertEqual(formai_sample._program_class(0, 1, 0), "SAFE")
                self.assertIsNone(formai_sample._program_class(0, 1, 1))
                rows = db.execute(
                    "SELECT vulnerable_line, error_type FROM vulnerability_records WHERE file_name='vuln.c' ORDER BY vulnerable_line"
                ).fetchall()
                self.assertEqual(rows, [(10, "buffer overflow"), (20, "null pointer dereference")])
            finally:
                db.close()

    def test_same_filename_with_different_source_is_rejected(self):
        records = [
            {"file_name": "same.c", "source_code": "int a;", "category": "VULNERABLE", "vulnerable_line": 1},
            {"file_name": "same.c", "source_code": "int b;", "category": "VULNERABLE", "vulnerable_line": 2},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            db = sqlite3.connect(Path(temp_dir) / "test.sqlite3")
            try:
                formai_sample._create_database(db)
                with patch.object(formai_sample, "iter_records", return_value=iter(records)):
                    with self.assertRaisesRegex(ValueError, "different source_code"):
                        formai_sample._aggregate_records(Path(temp_dir) / "unused.json", db)
            finally:
                db.close()


class FormAIEvaluationMultiLocationTests(unittest.TestCase):
    def test_localization_matches_any_vulnerable_line_and_any_error_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            labels = Path(temp_dir) / "labels.json"
            predictions = Path(temp_dir) / "predictions.jsonl"
            labels.write_text(json.dumps([
                {
                    "file_name": "multi.c",
                    "ground_truth_category": "VULNERABLE",
                    "vulnerable_lines": [10, 20, 30],
                    "error_types": ["buffer overflow", "null pointer dereference"],
                },
                {"file_name": "safe.c", "ground_truth_category": "SAFE", "vulnerable_lines": [], "error_types": []},
            ]), encoding="utf-8")
            predictions.write_text(
                json.dumps({"file_name": "multi.c", "predicted_vulnerable": True, "predicted_line": 20, "predicted_error_type": "null pointer dereference"}) + "\n"
                + json.dumps({"file_name": "safe.c", "predicted_vulnerable": False}) + "\n",
                encoding="utf-8",
            )
            result = evaluate_mode(labels, predictions, line_tolerance=0)
            self.assertEqual(result["confusion_matrix"], {"tp": 1, "fp": 0, "fn": 0, "tn": 1})
            self.assertEqual(result["localization"]["exact_line_accuracy"], 1.0)
            self.assertEqual(result["error_type_accuracy"], 1.0)

    def test_duplicate_prediction_filenames_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            labels = Path(temp_dir) / "labels.json"
            predictions = Path(temp_dir) / "predictions.jsonl"
            labels.write_text("[]", encoding="utf-8")
            item = {"file_name": "same.c", "predicted_vulnerable": False}
            predictions.write_text(json.dumps(item) + "\n" + json.dumps(item) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
                evaluate_mode(labels, predictions)


if __name__ == "__main__":
    unittest.main()
