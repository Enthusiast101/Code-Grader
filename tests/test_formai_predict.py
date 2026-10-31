import unittest

from grading_agent.benchmark.formai_predict import (
    _cppcheck_findings,
    _prediction_from_findings,
)


class FormAIPredictionParsingTests(unittest.TestCase):
    def test_parses_vulnerability_related_cppcheck_diagnostic(self):
        output = (
            "candidate.c:42:9: warning: Array index out of bounds [arrayIndexOutOfBounds]\n"
            "candidate.c:12:2: style: unusedFunction [unusedFunction]\n"
        )
        findings = _cppcheck_findings(output)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["line"], 42)
        self.assertEqual(findings[0]["id"], "arrayIndexOutOfBounds")
        self.assertTrue(_prediction_from_findings(findings)[0])

    def test_no_relevant_diagnostics_is_negative_prediction(self):
        findings = _cppcheck_findings(
            "candidate.c:12:2: style: unusedFunction [unusedFunction]\n"
        )
        self.assertEqual(findings, [])
        self.assertEqual(_prediction_from_findings(findings), (False, None, None))


if __name__ == "__main__":
    unittest.main()
