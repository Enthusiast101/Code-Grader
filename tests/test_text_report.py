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


    def test_rejected_claims_are_visible_in_report(self):
        report = {
            "problem": "sample",
            "submission": "main.c",
            "configuration": {"enabled_metrics": []},
            "scoring": {"status": "completed", "score": 100.0, "contributions": []},
            "llm_review": {
                "status": "completed_with_warnings",
                "review": {
                    "summary": "Boundary cases passed.",
                    "unverified_claims": [{
                        "claim": "Integer overflow is handled.",
                        "reason": "An overflow-specific test failed.",
                    }],
                },
            },
        }
        rendered = render_text_report(report)
        self.assertIn("Rejected / unverified claims", rendered)
        self.assertIn("An overflow-specific test failed", rendered)

    def test_recovered_intermediate_pass_is_not_rendered_as_failure(self):
        report = {
            "problem": "sample", "submission": "main.c",
            "configuration": {"enabled_metrics": []},
            "scoring": {"status": "completed", "score": 100.0, "contributions": []},
            "llm_review": {
                "status": "completed",
                "llm_status": "completed",
                "final_synthesis_status": "llm_advisory_completed",
                "failed_passes": [],
                "recovered_passes": [{"iteration": 1, "error": "intermediate parse issue"}],
                "review": {"summary": "Deterministic review is complete.", "findings": []},
                "llm_advisory": {"summary": "Final validated advisory."},
            },
        }
        rendered = render_text_report(report)
        self.assertIn("Final validated advisory.", rendered)
        self.assertIn("Recovered intermediate LLM pass(es): 1", rendered)
        self.assertNotIn("Review pass 1 error", rendered)
        self.assertNotIn("LLM diagnostic:", rendered)

    def test_usefulness_and_limitations_section_is_removed_even_from_legacy_report(self):
        report = {
            "configuration": {"enabled_metrics": []},
            "scoring": {"status": "completed", "score": 100.0, "contributions": []},
            "usefulness_and_limitations": {
                "usefulness": [{"use_case": "legacy item", "example": "example", "benefit": "benefit"}],
                "limitations": [{"limitation": "legacy limitation", "example": "example", "mitigation": "mitigation"}],
            },
        }
        rendered = render_text_report(report)
        self.assertNotIn("USEFULNESS AND LIMITATIONS", rendered)
        self.assertNotIn("legacy item", rendered)
        self.assertNotIn("legacy limitation", rendered)

    def test_deterministic_fallback_is_distinguished_from_llm_success(self):
        from grading_agent.reporting.text_report import render_text_report
        report = {
            "problem": "sample", "submission": "main.c",
            "configuration": {"enabled_metrics": []},
            "scoring": {"status": "completed", "score": 100.0, "contributions": []},
            "llm_review": {
                "status": "completed_with_warnings",
                "review_engine": "python_deterministic_evidence_review",
                "llm_status": "unavailable",
                "llm_error": "mock model failure",
                "review": {
                    "summary": "Deterministic evidence review is available.",
                    "summary_evidence_ids": ["E-PREFLIGHT-001"],
                    "findings": [], "contradictions": [],
                    "recommendations": [], "uncertainties": [],
                },
            },
        }
        rendered = render_text_report(report)
        self.assertIn("EVIDENCE REVIEW", rendered)
        self.assertIn("python_deterministic_evidence_review", rendered)
        self.assertIn("LLM advisory: unavailable; deterministic evidence review is complete.", rendered)
        self.assertIn("Local LLM status: unavailable", rendered)
        self.assertIn("mock model failure", rendered)


if __name__ == "__main__":
    unittest.main()
