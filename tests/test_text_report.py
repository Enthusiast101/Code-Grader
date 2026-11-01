import unittest

from grading_agent.reporting.text_report import render_text_report


class TextReportRSITests(unittest.TestCase):
    def test_metric_rsi_scores_are_rendered(self):
        report = {
            "problem": "sum_two_integers",
            "submission": "main.c",
            "configuration": {"enabled_metrics": []},
            "scoring": {"status": "completed", "score": 98.8, "contributions": []},
            "llm_rsi_evaluation": {
                "status": "completed_with_warnings",
                "metrics": [{
                    "name": "correctness", "score": 98.0,
                    "rationale": "All configured cases passed; additional inputs were source-reviewed.",
                }],
                "missing_metrics": ["logical"],
            },
        }
        rendered = render_text_report(report)
        self.assertIn("LLM EVIDENCE-BASED RSI", rendered)
        self.assertIn("Status: completed_with_warnings", rendered)
        self.assertIn("correctness: 98.0/100", rendered)
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
