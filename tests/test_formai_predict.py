import unittest

from grading_agent.benchmark.formai_predict import (
    _cppcheck_findings,
    _prediction_from_findings,
    _prediction_from_rsi_review,
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

    def test_rsi_prediction_requires_validated_vulnerability_claim_and_source_evidence(self):
        review = {
            "status": "completed",
            "claim_validation": {"status": "completed", "verified_claim_count": 1},
            "review": {"findings": [{
                "text": "Potential buffer overflow at the indexed write.",
                "kind": "potential",
                "evidence_ids": ["E-SOURCE-L004"],
            }]},
        }
        ledger = [{
            "id": "E-SOURCE-L004", "kind": "source_line",
            "statement": "Source line 4", "facts": {"line": 4, "code": "buffer[index] = value;"},
        }]
        self.assertEqual(
            _prediction_from_rsi_review(review, ledger)[:3],
            (True, 4, "source_review_finding"),
        )

    def test_rsi_does_not_accept_unreferenced_findings(self):
        review = {
            "status": "completed",
            "claim_validation": {"status": "completed", "verified_claim_count": 1},
            "review": {"findings": [{
                "text": "Potential buffer overflow at the indexed write.",
                "kind": "potential",
                "evidence_ids": ["E-UNKNOWN"],
            }]},
        }
        ledger = [{"id": "E-SOURCE-L004", "kind": "source_line", "facts": {"line": 4}}]
        self.assertEqual(_prediction_from_rsi_review(review, ledger)[:3], (False, None, None))

    def test_no_relevant_diagnostics_is_negative_prediction(self):
        findings = _cppcheck_findings(
            "candidate.c:12:2: style: unusedFunction [unusedFunction]\n"
        )
        self.assertEqual(findings, [])
        self.assertEqual(_prediction_from_findings(findings), (False, None, None))


if __name__ == "__main__":
    unittest.main()
