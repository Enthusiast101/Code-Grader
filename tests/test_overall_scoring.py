import unittest

from grading_agent.scoring.overall import apply_metric_scores, calculate_overall_score


class OverallScoringTests(unittest.TestCase):
    def setUp(self):
        self.metrics = [
            {"name": "correctness", "analyzer": "io_tests", "weight": 0.6, "enabled": True},
            {"name": "logical", "analyzer": "edge_case_runner", "weight": 0.4, "enabled": True},
        ]

    def test_llm_rsi_scores_are_used_then_python_applies_weights(self):
        report = {
            "correctness": {"status": "completed", "score": 80.0},
            "logical": {"status": "unavailable", "score": None},
        }
        llm = {"metrics": [
            {"name": "correctness", "score": 90.0, "rationale": "Test evidence reviewed."},
            {"name": "logical", "score": 70.0, "rationale": "Source-only estimate."},
        ]}
        apply_metric_scores(self.metrics, report, llm)
        self.assertEqual(report["correctness"]["score"], 90.0)
        self.assertEqual(report["logical"]["score"], 70.0)
        final = calculate_overall_score(self.metrics, report)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["score"], 82.0)
        self.assertEqual(len(final["contributions"]), 2)

    def test_missing_llm_uses_analyzer_score(self):
        report = {"correctness": {"status": "completed", "score": 88.0}, "logical": {"status": "completed", "score": 92.0}}
        apply_metric_scores(self.metrics, report, {"status": "unavailable", "metrics": []})
        self.assertEqual(report["correctness"]["score"], 88.0)
        self.assertEqual(calculate_overall_score(self.metrics, report)["score"], 89.6)

    def test_partial_llm_output_uses_analyzer_only_for_missing_metric(self):
        report = {"correctness": {"status": "completed", "score": 80.0}, "logical": {"status": "completed", "score": 92.0}}
        llm = {"metrics": [{"name": "correctness", "score": 75.0}]}
        apply_metric_scores(self.metrics, report, llm)
        self.assertEqual(report["correctness"]["score"], 75.0)
        self.assertEqual(report["logical"]["score"], 92.0)

    def test_no_scores_does_not_block_final_weighted_grade(self):
        report = {"correctness": {"status": "unavailable", "score": None}, "logical": {"status": "unavailable", "score": None}}
        apply_metric_scores(self.metrics, report, {"status": "unavailable", "metrics": []})
        final = calculate_overall_score(self.metrics, report)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["score"], 0.0)
        self.assertEqual(len(final["warnings"]), 2)


if __name__ == "__main__":
    unittest.main()
