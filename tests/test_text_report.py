import unittest

from grading_agent.reporting.text_report import render_text_report


class TextReportLLMReviewTests(unittest.TestCase):
    def test_partial_review_summary_is_rendered(self):
        report = {
            "problem": "sum_two_integers",
            "submission": "main.c",
            "configuration": {"enabled_metrics": []},
            "scoring": {"status": "completed", "score": 98.8, "contributions": []},
            "llm_review": {
                "status": "partial",
                "final_synthesis_status": "failed_using_previous_valid_review",
                "failed_passes": [{"iteration": 3, "error": "empty model response"}],
                "review": {
                    "summary": "The valid first-pass review is retained.",
                    "findings": ["All configured tests passed."],
                    "uncertainties": ["Final synthesis did not complete."],
                },
            },
        }
        rendered = render_text_report(report)
        self.assertIn("Status: partial", rendered)
        self.assertIn("The valid first-pass review is retained.", rendered)
        self.assertIn("Final synthesis did not complete.", rendered)
        self.assertIn("empty model response", rendered)

    def test_unavailable_review_error_is_rendered(self):
        report = {
            "configuration": {"enabled_metrics": []},
            "scoring": {"status": "incomplete"},
            "llm_review": {"status": "unavailable", "error": "Local model server stopped."},
        }
        rendered = render_text_report(report)
        self.assertIn("Status: unavailable", rendered)
        self.assertIn("Local model server stopped.", rendered)


if __name__ == "__main__":
    unittest.main()
