import unittest

from grading_agent.reporting.text_report import render_text_report


class TextReportRSITests(unittest.TestCase):
    def test_configured_point_caps_are_used_for_metric_display(self):
        report = {
            "configuration": {"enabled_metrics": [
                {"name": "correctness", "weight": 0.2, "max_points": 20},
                {"name": "runtime_efficiency", "weight": 0, "max_points": 0, "grade_contribution": False},
            ]},
            "correctness": {"status": "completed", "score": 62.5, "score_points": 12.5, "score_origin": "deterministic_test_results"},
            "runtime_efficiency": {"status": "inconclusive", "score": None},
            "scoring": {"status": "completed", "score": 12.5, "contributions": [
                {"metric": "correctness", "score": 62.5, "weight": 0.2, "contribution": 12.5,
                 "points_awarded": 12.5, "max_points": 20, "score_origin": "deterministic_test_results"}
            ]},
        }
        rendered = render_text_report(report)
        self.assertIn("correctness: 12.50/20", rendered)
        self.assertIn("runtime_efficiency     weight=0.000  score=diagnostic only", rendered)

    def test_metric_rsi_scores_are_rendered(self):
        report = {
            "problem": "sum_two_integers",
            "submission": "main.c",
            "configuration": {"enabled_metrics": []},
            "scoring": {"status": "completed", "score": 98.8, "contributions": []},
            "llm_rsi_evaluation": {
                "status": "completed_with_warnings",
                "metrics": [{
                    "name": "correctness", "score": 9.8, "score_scale": "0-10", "origin": "llm",
                    "rationale": "All configured cases passed; additional inputs were source-reviewed.",
                }],
                "missing_metrics": ["logical"],
            },
        }
        rendered = render_text_report(report)
        self.assertIn("LLM EVIDENCE-BASED RSI", rendered)
        self.assertIn("Status: completed_with_warnings", rendered)
        self.assertIn("correctness: 9.80/10 (LLM proposal)", rendered)
        self.assertIn("logical", rendered)

    def test_unavailable_rsi_error_is_rendered(self):
        report = {
            "configuration": {"enabled_metrics": []},
            "scoring": {"status": "completed", "score": 0.0, "contributions": []},
            "llm_rsi_evaluation": {"status": "unavailable", "error": "Local model server stopped."},
        }
        rendered = render_text_report(report)
        self.assertIn("Status: unavailable", rendered)
        self.assertIn("Local model server stopped.", rendered)


if __name__ == "__main__":
    unittest.main()
