import unittest

from grading_agent.scoring.overall import apply_metric_scores, calculate_overall_score


class OverallScoringTests(unittest.TestCase):
    def setUp(self):
        self.metrics = [
            {"name": "correctness", "analyzer": "io_tests", "weight": 0.6, "enabled": True},
            {"name": "logical", "analyzer": "edge_case_runner", "weight": 0.4, "enabled": True},
        ]

    def test_validated_llm_scores_are_primary_when_no_test_cap_applies(self):
        report = {
            "correctness": {"status": "completed", "score": 80.0},
            "logical": {"status": "completed", "score": 90.0},
        }
        llm = {"metrics": [
            {"name": "correctness", "score": 7.5, "score_band": "5-10", "origin": "llm", "rationale": "LLM proposal."},
            {"name": "logical", "score": 2.5, "score_band": "0-5", "origin": "llm", "rationale": "LLM proposal."},
        ]}
        apply_metric_scores(self.metrics, report, llm)
        self.assertEqual(report["correctness"]["score"], 7.5)
        self.assertEqual(report["logical"]["score"], 2.5)
        self.assertEqual(report["correctness"]["llm_review_score"], 7.5)
        self.assertEqual(report["logical"]["llm_review_score"], 2.5)
        self.assertEqual(calculate_overall_score(self.metrics, report)["score"], 5.5)

    def test_executed_correctness_and_logical_test_rates_are_authoritative(self):
        report = {
            "correctness": {"status": "completed", "score": 50.0, "tests_passed": 1, "tests_total": 2},
            "logical": {"status": "completed", "score": 100.0, "tests_passed": 2, "tests_total": 2},
        }
        llm = {"metrics": [
            {"name": "correctness", "score": 85.0, "origin": "llm", "rationale": "LLM proposed a high score."},
            {"name": "logical", "score": 95.0, "origin": "llm", "rationale": "Both edge tests passed."},
        ]}
        apply_metric_scores(self.metrics, report, llm)
        self.assertEqual(report["correctness"]["llm_review_score"], 85.0)
        self.assertEqual(report["correctness"]["score"], 50.0)
        self.assertEqual(report["correctness"]["score_origin"], "deterministic_test_results")
        self.assertTrue(report["correctness"]["test_score_override_applied"])
        self.assertEqual(report["logical"]["score"], 100.0)
        self.assertEqual(report["logical"]["score_origin"], "deterministic_test_results")
        self.assertEqual(calculate_overall_score(self.metrics, report)["score"], 70.0)

    def test_validated_llm_score_can_measure_a_metric_without_analyzer_output(self):
        report = {
            "correctness": {"status": "completed", "score": 80.0},
            "logical": {"status": "inconclusive", "score": None, "reason": "Missing test inputs."},
        }
        llm = {"metrics": [
            {"name": "correctness", "score": 92.5, "score_band": "90-95", "origin": "llm"},
            {"name": "logical", "score": 97.5, "score_band": "95-100", "origin": "llm", "rationale": "LLM guessed from source."},
        ]}
        apply_metric_scores(self.metrics, report, llm)
        self.assertEqual(report["logical"]["score"], 97.5)
        self.assertEqual(report["logical"]["llm_review_score"], 97.5)
        final = calculate_overall_score(self.metrics, report)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["score"], 94.5)


    def test_numeric_llm_score_without_supplied_band_is_accepted_and_banded_locally(self):
        report = {"correctness": {"status": "completed", "score": 42.0}, "logical": {"status": "completed", "score": 58.0}}
        llm = {"metrics": [{"name": "correctness", "score": 99.0, "origin": "llm"}]}
        apply_metric_scores(self.metrics, report, llm)
        self.assertEqual(report["correctness"]["llm_review_score"], 99.0)
        self.assertEqual(report["correctness"]["score"], 99.0)
        self.assertEqual(calculate_overall_score(self.metrics, report)["score"], 82.6)

    def test_test_pass_rate_is_derived_from_rows_when_aggregate_counts_are_missing(self):
        report = {
            "correctness": {"status": "completed", "score": 90.0, "results": [
                {"passed": True}, {"passed": False},
            ]},
            "logical": {"status": "completed", "score": 100.0},
        }
        llm = {"metrics": [
            {"name": "correctness", "score": 95.0, "origin": "llm", "rationale": "Looks correct."},
            {"name": "logical", "score": 95.0, "origin": "llm", "rationale": "Looks logical."},
        ]}
        apply_metric_scores(self.metrics, report, llm)
        self.assertEqual(report["correctness"]["score"], 50.0)
        self.assertEqual(report["correctness"]["score_origin"], "deterministic_test_results")
        self.assertEqual(report["logical"]["score"], 95.0)

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
