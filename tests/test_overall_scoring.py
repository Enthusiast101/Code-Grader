import unittest

from grading_agent.scoring.overall import apply_metric_scores, calculate_overall_score


class OverallScoringTests(unittest.TestCase):
    def setUp(self):
        self.metrics = [
            {"name": "correctness", "analyzer": "io_tests", "weight": 0.6, "enabled": True},
            {"name": "logical", "analyzer": "edge_case_runner", "weight": 0.4, "enabled": True},
        ]

    def test_analyzers_are_authoritative_even_when_llm_disagrees(self):
        report = {
            "correctness": {"status": "completed", "score": 80.0},
            "logical": {"status": "completed", "score": 90.0},
        }
        llm = {"metrics": [
            {"name": "correctness", "score": 7.5, "score_band": "5-10", "origin": "llm", "rationale": "LLM proposal."},
            {"name": "logical", "score": 2.5, "score_band": "0-5", "origin": "llm", "rationale": "LLM proposal."},
        ]}
        apply_metric_scores(self.metrics, report, llm)
        self.assertEqual(report["correctness"]["score"], 80.0)
        self.assertEqual(report["logical"]["score"], 90.0)
        self.assertEqual(report["correctness"]["llm_review_score"], 7.5)
        self.assertEqual(report["logical"]["llm_review_score"], 2.5)
        self.assertEqual(calculate_overall_score(self.metrics, report)["score"], 84.0)

    def test_llm_cannot_fill_an_unmeasured_metric(self):
        report = {
            "correctness": {"status": "completed", "score": 80.0},
            "logical": {"status": "inconclusive", "score": None, "reason": "Missing test inputs."},
        }
        llm = {"metrics": [
            {"name": "correctness", "score": 92.5, "score_band": "90-95", "origin": "llm"},
            {"name": "logical", "score": 97.5, "score_band": "95-100", "origin": "llm", "rationale": "LLM guessed from source."},
        ]}
        apply_metric_scores(self.metrics, report, llm)
        self.assertIsNone(report["logical"]["score"])
        self.assertEqual(report["logical"]["llm_review_score"], 97.5)
        final = calculate_overall_score(self.metrics, report)
        self.assertEqual(final["status"], "failed_incomplete")
        self.assertIsNone(final["score"])
        self.assertEqual(final["missing_metrics"][0]["metric"], "logical")


    def test_numeric_llm_score_without_band_is_ignored(self):
        report = {"correctness": {"status": "completed", "score": 42.0}, "logical": {"status": "completed", "score": 58.0}}
        llm = {"metrics": [{"name": "correctness", "score": 99.0, "origin": "llm"}]}
        apply_metric_scores(self.metrics, report, llm)
        self.assertIsNone(report["correctness"]["llm_review_score"])
        self.assertEqual(report["correctness"]["score"], 42.0)
        self.assertEqual(calculate_overall_score(self.metrics, report)["score"], 48.4)

    def test_completed_analyzer_zero_is_a_real_zero(self):
        report = {"correctness": {"status": "completed", "score": 0.0}, "logical": {"status": "completed", "score": 100.0}}
        apply_metric_scores(self.metrics, report, None)
        final = calculate_overall_score(self.metrics, report)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(report["correctness"]["score"], 0.0)
        self.assertEqual(final["score"], 40.0)

    def test_completed_analyzer_scores_are_weighted_deterministically(self):
        report = {
            "correctness": {"status": "completed", "score": 80.0},
            "logical": {"status": "completed", "score": 90.0},
        }
        apply_metric_scores(self.metrics, report, {"status": "unavailable", "metrics": []})
        final = calculate_overall_score(self.metrics, report)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["score"], 84.0)
        self.assertTrue(all(x["score_origin"] == "deterministic_analyzer" for x in final["contributions"]))

    def test_missing_result_fails_closed_instead_of_neutral_70(self):
        report = {"correctness": {"status": "completed", "score": 80.0}}
        final = calculate_overall_score(self.metrics, report)
        self.assertEqual(final["status"], "failed_incomplete")
        self.assertIsNone(final["score"])
        self.assertEqual(final["missing_metrics"][0]["metric"], "logical")

    def test_disabled_metrics_are_not_required(self):
        metrics = [
            {"name": "correctness", "analyzer": "io_tests", "weight": 1.0, "enabled": True},
            {"name": "runtime", "analyzer": "runtime_profile", "weight": 0.0, "enabled": False},
        ]
        report = {"correctness": {"status": "completed", "score": 75.0}}
        apply_metric_scores(metrics, report, None)
        self.assertEqual(calculate_overall_score(metrics, report)["score"], 75.0)

    def test_invalid_weights_are_rejected(self):
        metrics = [
            {"name": "correctness", "analyzer": "io_tests", "weight": 0.4, "enabled": True},
            {"name": "logical", "analyzer": "edge_case_runner", "weight": 0.4, "enabled": True},
        ]
        with self.assertRaises(ValueError):
            calculate_overall_score(metrics, {})


if __name__ == "__main__":
    unittest.main()
