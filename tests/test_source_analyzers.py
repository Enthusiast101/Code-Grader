import json
import tempfile
import unittest
from pathlib import Path

from grading_agent.analyzers.source_metrics import analyze_source
from grading_agent.analyzers.modularity import run_modularity
from grading_agent.analyzers.complexity_estimator import run_complexity_estimator
from grading_agent.analyzers.readability import run_readability
from grading_agent.analyzers.similarity import run_similarity_flag
from grading_agent.scoring.overall import calculate_overall_score
from grading_agent.benchmark.formai_evaluate import evaluate_mode
from grading_agent.benchmark.formai_split import split_formai


SOURCE = r'''#include <stdio.h>
int add_values(int left, int right) {
    return left + right;
}
int main(void) {
    int result = add_values(2, 3);
    printf("%d\\n", result);
    return 0;
}
'''


class SourceAnalyzerTests(unittest.TestCase):
    def test_source_metrics_detect_functions(self):
        metrics = analyze_source(SOURCE)
        self.assertEqual(metrics["function_count"], 2)
        self.assertEqual(metrics["functions"][0]["name"], "add_values")

    def test_modularity_reports_helpers(self):
        result = run_modularity(SOURCE, min_functions=2, expect_helpers=True)
        self.assertEqual(result["helper_function_count"], 1)
        self.assertEqual(result["score"], 100.0)

    def test_complexity_returns_0_100_score(self):
        result = run_complexity_estimator(SOURCE)
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)

    def test_readability_returns_0_100_score(self):
        result = run_readability(SOURCE)
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)

    def test_similarity_flags_near_identical_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            (corpus / "prior.c").write_text(SOURCE, encoding="utf-8")
            result = run_similarity_flag(SOURCE, corpus, flag_threshold=0.8)
            self.assertTrue(result["flagged_for_human_review"])
            self.assertEqual(result["max_similarity"], 1.0)


class OverallScoringTests(unittest.TestCase):
    def test_weighted_score(self):
        metrics = [
            {"name": "correctness", "analyzer": "io_tests", "weight": 0.6, "enabled": True},
            {"name": "memory_safety", "analyzer": "memory_safety", "weight": 0.4, "enabled": True},
        ]
        results = {"correctness": {"status": "completed", "score": 100}, "memory_safety": {"status": "completed", "score": 90}}
        score = calculate_overall_score(metrics, results)
        self.assertEqual(score["score"], 96.0)

    def test_missing_metric_uses_zero_and_still_returns_final_grade(self):
        metrics = [{"name": "similarity", "analyzer": "similarity_flag", "weight": 1.0, "enabled": True}]
        result = calculate_overall_score(metrics, {"similarity": {"status": "inconclusive", "score": None}})
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["warnings"]), 1)


class FormAIEvaluatorTests(unittest.TestCase):
    def test_evaluator_confusion_and_localization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = root / "truth.json"
            preds = root / "predictions.jsonl"
            truth.write_text(json.dumps([
                {"file_name": "a.c", "ground_truth_category": "VULNERABLE", "vulnerable_line": 10, "error_type": "buffer overflow"},
                {"file_name": "b.c", "ground_truth_category": "SAFE"},
            ]), encoding="utf-8")
            preds.write_text(
                json.dumps({"file_name": "a.c", "predicted_vulnerable": True, "predicted_line": 11, "predicted_error_type": "buffer overflow"}) + "\n" +
                json.dumps({"file_name": "b.c", "predicted_vulnerable": False}) + "\n",
                encoding="utf-8",
            )
            result = evaluate_mode(truth, preds, line_tolerance=1)
            self.assertEqual(result["f1_score"], 1.0)
            self.assertEqual(result["localization"]["within_tolerance_accuracy"], 1.0)


class FormAISplitTests(unittest.TestCase):
    def test_stratified_split_keeps_labels_separate_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "inputs.jsonl"
            labels_path = root / "labels.json"
            records = []
            labels = []
            with source_path.open("w", encoding="utf-8") as stream:
                for index in range(10):
                    category = "VULNERABLE" if index < 5 else "SAFE"
                    filename = f"sample-{index}.c"
                    stream.write(json.dumps({"file_name": filename, "source_code": "int main(void){return 0;}"}) + "\n")
                    labels.append({"file_name": filename, "ground_truth_category": category})
            labels_path.write_text(json.dumps(labels), encoding="utf-8")

            out1 = root / "split1"
            out2 = root / "split2"
            first = split_formai(source_path, labels_path, out1, seed=17)
            second = split_formai(source_path, labels_path, out2, seed=17)
            self.assertEqual(
                {k: v["count"] for k, v in first["splits"].items()},
                {k: v["count"] for k, v in second["splits"].items()},
            )
            for split_name in ("train", "validation", "test"):
                source_text = (out1 / f"{split_name}_inputs.jsonl").read_text(encoding="utf-8")
                self.assertNotIn("ground_truth_category", source_text)
                label_data = json.loads((out1 / f"{split_name}_labels.json").read_text(encoding="utf-8"))
                self.assertEqual(len(label_data), first["splits"][split_name]["count"])


if __name__ == "__main__":
    unittest.main()
