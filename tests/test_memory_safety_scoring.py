import unittest

from grading_agent.scoring.memory_safety import calculate_memory_safety_score


class MemorySafetyScoringTests(unittest.TestCase):
    def setUp(self):
        self.valgrind = {
            "compiled": True,
            "analysis_error": False,
            "tests_total": 2,
            "tests_checked": 2,
            "tests_clean": 2,
            "results": [
                {
                    "memory_safe": True,
                    "analysis_error": False,
                    "timed_out": False,
                    "log_truncated": False,
                    "log_read_error": None,
                    "parsed_summary": {"error_count": 0},
                },
                {
                    "memory_safe": True,
                    "analysis_error": False,
                    "timed_out": False,
                    "log_truncated": False,
                    "log_read_error": None,
                    "parsed_summary": {"error_count": 0},
                },
            ],
        }
        self.audit = {
            "status": "completed",
            "audit": {"findings": [], "summary": "No source-level findings."},
        }
        self.penalties = {
            "critical": 40.0,
            "high": 25.0,
            "medium": 10.0,
            "low": 3.0,
        }

    def calculate(self, valgrind=None, audit=None, vw=0.7, lw=0.3):
        return calculate_memory_safety_score(
            valgrind or self.valgrind,
            audit or self.audit,
            valgrind_weight=vw,
            llm_audit_weight=lw,
            severity_penalties=self.penalties,
        )

    def test_clean_results_score_100(self):
        result = self.calculate()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["score"], 100.0)

    def test_source_finding_uses_configured_severity_penalty(self):
        audit = {
            "status": "completed",
            "audit": {
                "findings": [{
                    "category": "out_of_bounds_write",
                    "severity": "high",
                    "line": 12,
                    "function": "copy_input",
                    "description": "Potential out-of-bounds write.",
                }]
            },
        }
        result = self.calculate(audit=audit)
        self.assertEqual(result["llm_source_audit"]["score"], 75.0)
        self.assertEqual(result["score"], 92.5)

    def test_inconclusive_valgrind_withholds_composite_score(self):
        valgrind = {**self.valgrind, "analysis_error": True}
        result = self.calculate(valgrind=valgrind)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["score"])

    def test_unavailable_llm_withholds_composite_score(self):
        audit = {"status": "unavailable", "error": "server unavailable"}
        result = self.calculate(audit=audit)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["score"])

    def test_same_finding_is_deduplicated(self):
        finding = {
            "category": "out_of_bounds_write",
            "severity": "high",
            "line": 12,
            "function": "copy_input",
            "description": "Potential out-of-bounds write.",
        }
        audit = {"status": "completed", "audit": {"findings": [finding, finding]}}
        result = self.calculate(audit=audit)
        self.assertEqual(result["llm_source_audit"]["unique_findings"], 1)


if __name__ == "__main__":
    unittest.main()
