import unittest

from grading_agent.scoring.overall import apply_metric_scores, calculate_overall_score


class OverallScoringTests(unittest.TestCase):
    def setUp(self):
        self.metrics = [
            {"name": "correctness", "analyzer": "io_tests", "weight": 0.6, "enabled": True},
            {"name": "logical", "analyzer": "edge_case_runner", "weight": 0.4, "enabled": True},
        ]

    def test_validated_llm_scores_are_primary_when_no_test_cap_applies(self):
        metrics = [
            {"name": "structure", "analyzer": "static_analysis", "weight": 0.6, "enabled": True},
            {"name": "readability", "analyzer": "readability", "weight": 0.4, "enabled": True},
        ]
        report = {
            "structure": {"status": "completed", "score": 80.0},
            "readability": {"status": "completed", "score": 90.0},
        }
        llm = {"metrics": [
            {"name": "structure", "score": 7.5, "score_band": "5-10", "origin": "llm", "rationale": "LLM proposal."},
            {"name": "readability", "score": 2.5, "score_band": "0-5", "origin": "llm", "rationale": "LLM proposal."},
        ]}
        apply_metric_scores(metrics, report, llm)
        self.assertEqual(report["structure"]["score"], 75.0)
        self.assertEqual(report["readability"]["score"], 25.0)
        self.assertEqual(report["structure"]["llm_review_score"], 7.5)
        self.assertEqual(report["readability"]["llm_review_score"], 2.5)
        self.assertEqual(calculate_overall_score(metrics, report)["score"], 55.0)

    def test_executed_correctness_and_logical_test_rates_are_authoritative(self):
        report = {
            "correctness": {"status": "completed", "score": 50.0, "tests_passed": 1, "tests_total": 2},
            "logical": {"status": "completed", "score": 100.0, "tests_passed": 2, "tests_total": 2},
        }
        llm = {"metrics": [
            {"name": "correctness", "score": 8.5, "origin": "llm", "rationale": "LLM proposed a high score."},
            {"name": "logical", "score": 9.5, "origin": "llm", "rationale": "Both edge tests passed."},
        ]}
        apply_metric_scores(self.metrics, report, llm)
        self.assertEqual(report["correctness"]["llm_review_score"], 8.5)
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
            {"name": "correctness", "score": 9.25, "score_band": "9-10", "origin": "llm"},
            {"name": "logical", "score": 9.75, "score_band": "9-10", "origin": "llm", "rationale": "LLM guessed from source."},
        ]}
        apply_metric_scores(self.metrics, report, llm)
        self.assertIsNone(report["logical"]["score"])
        self.assertEqual(report["logical"]["llm_review_score"], 9.75)
        final = calculate_overall_score(self.metrics, report)
        self.assertEqual(final["status"], "failed_incomplete")
        self.assertIsNone(final["score"])


    def test_deterministic_execution_score_overrides_contradictory_llm_score(self):
        report = {"runtime": {"status": "completed", "score": 42.0, "median_runtime_ms": 2000}}
        metrics = [{"name": "runtime", "analyzer": "runtime_profile", "weight": 1.0, "enabled": True}]
        llm = {"metrics": [{"name": "runtime", "score": 9.9, "origin": "llm"}]}
        apply_metric_scores(metrics, report, llm)
        self.assertEqual(report["runtime"]["llm_review_score"], 9.9)
        self.assertEqual(report["runtime"]["score"], 42.0)
        self.assertEqual(report["runtime"]["score_origin"], "deterministic_analyzer")
        self.assertEqual(calculate_overall_score(metrics, report)["score"], 42.0)

    def test_compiler_analyzer_score_is_authoritative_over_llm_rubric(self):
        report = {"syntax": {"analyzer": "compiler_diagnostics", "status": "completed", "score": 85.0, "exit_code": 0, "warning_count": 3}}
        metrics = [{"name": "syntax", "analyzer": "compiler_diagnostics", "weight": 1.0, "enabled": True}]
        apply_metric_scores(metrics, report, {"metrics": [{"name": "syntax", "score": 10.0, "origin": "llm"}]})
        self.assertEqual(report["syntax"]["score"], 85.0)
        self.assertEqual(report["syntax"]["score_origin"], "deterministic_analyzer")

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
        self.assertEqual(report["logical"]["score"], 100.0)

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

    def test_configured_metric_caps_convert_percent_scores_to_points(self):
        names_and_caps = [
            ("correctness", 20), ("syntax", 10), ("structure", 10),
            ("modularity", 10), ("logical", 10), ("memory_safety", 10),
            ("complexity", 10), ("readability", 10), ("originality", 10),
        ]
        metrics = [
            {"name": name, "weight": cap / 100, "max_points": cap, "enabled": True}
            for name, cap in names_and_caps
        ]
        report = {
            name: {"status": "completed", "score": 50.0 if name == "correctness" else 100.0}
            for name, _ in names_and_caps
        }
        apply_metric_scores(metrics, report, None)
        overall = calculate_overall_score(metrics, report)
        self.assertEqual(report["correctness"]["score_points"], 10.0)
        self.assertEqual(report["correctness"]["max_points"], 20.0)
        self.assertEqual(overall["score"], 90.0)
        self.assertEqual(sum(item["points_awarded"] for item in overall["contributions"]), 90.0)
        self.assertEqual(sum(item["max_points"] for item in overall["contributions"]), 100.0)

    def test_diagnostic_only_runtime_metric_does_not_change_grade_or_gate_completion(self):
        metrics = [
            {"name": "correctness", "weight": 1.0, "max_points": 100, "enabled": True},
            {"name": "runtime_efficiency", "weight": 0.0, "max_points": 0, "enabled": True, "grade_contribution": False},
        ]
        report = {
            "correctness": {"status": "completed", "score": 80.0},
            "runtime_efficiency": {"status": "inconclusive", "score": None, "reason": "No runtime tests."},
        }
        apply_metric_scores(metrics, report, None)
        overall = calculate_overall_score(metrics, report)
        self.assertEqual(overall["score"], 80.0)
        self.assertEqual([item["metric"] for item in overall["contributions"]], ["correctness"])

    def test_execution_metrics_without_measurement_reject_llm_scores(self):
        for name, analyzer in (("runtime", "runtime_profile"), ("memory", "memory_safety")):
            with self.subTest(metric=name):
                metric = {"name": name, "analyzer": analyzer, "weight": 1.0, "max_points": 100, "enabled": True}
                report = {name: {"status": "inconclusive", "score": None, "reason": "No test inputs."}}
                llm = {"metrics": [{"name": name, "score_points": 100.0, "origin": "llm", "rationale": "This metric could not be measured from available evidence."}]}
                apply_metric_scores([metric], report, llm)
                self.assertIsNone(report[name]["score"])
                overall = calculate_overall_score([metric], report)
                self.assertEqual(overall["status"], "failed_incomplete")
                self.assertIsNone(overall["score"])

    def test_llm_unmeasured_rationale_cannot_become_full_credit(self):
        metric = {"name": "structure", "analyzer": "static_analysis", "weight": 1.0, "max_points": 100, "enabled": True}
        report = {"structure": {"status": "unavailable", "score": None}}
        llm = {"metrics": [{"name": "structure", "score_points": 100.0, "origin": "llm", "rationale": "Structure could not be measured from available evidence."}]}
        apply_metric_scores([metric], report, llm)
        self.assertIsNone(report["structure"]["score"])
        self.assertEqual(calculate_overall_score([metric], report)["status"], "failed_incomplete")

    def test_completed_compiler_measurement_does_not_claim_unmeasured(self):
        metric = {"name": "syntax", "analyzer": "compiler_diagnostics", "weight": 1.0, "max_points": 100, "enabled": True}
        report = {"syntax": {"analyzer": "compiler_diagnostics", "status": "completed", "score": 100.0, "compiled": True, "exit_code": 0, "warning_count": 0, "error_count": 0}}
        apply_metric_scores([metric], report, None)
        rationale = report["syntax"]["score_rationale"].lower()
        self.assertIn("compilation succeeded", rationale)
        self.assertNotIn("could not be measured", rationale)


if __name__ == "__main__":
    unittest.main()
