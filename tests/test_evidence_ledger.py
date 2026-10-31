import unittest

from grading_agent.llm.evidence import build_evidence_ledger, validate_review_payload


class EvidenceLedgerTests(unittest.TestCase):
    def setUp(self):
        self.rules = {
            "problem_contract": {"operand_type": "int32", "sum_type": "int64"},
            "metrics": [{"name": "modularity", "enabled": True, "params": {"expect_helpers": False, "min_functions": 1}}],
        }
        self.report = {
            "submission": "main.c",
            "source_code": "int main(void) { return 0; }",
            "configuration": {"enabled_metrics": [{"name": "logical"}, {"name": "modularity"}]},
            "logical": {
                "status": "completed", "score": 50, "weight": 0.5,
                "tests_passed": 1, "tests_total": 2,
                "results": [
                    {"test_id": "E01", "passed": True, "input": "2147483647 0", "expected": "2147483647", "actual": "2147483647"},
                    {"test_id": "E02", "passed": False, "input": "2147483647 1", "expected": "2147483648", "actual": "-2147483648", "tags": ["signed_int32_overflow"]},
                ],
            },
            "modularity": {
                "status": "completed", "score": 100, "weight": 0.5,
                "function_count": 1, "helper_function_count": 0,
                "minimum_functions": 1, "helpers_expected": False,
                "functions": [{"name": "main"}],
            },
        }
        self.ledger = build_evidence_ledger(self.report, self.rules)
        self.by_id = {item["id"]: item for item in self.ledger}

    def _review(self):
        return {
            "summary": "The boundary test passed, but the overflow-tagged test failed.",
            "summary_evidence_ids": ["E-LOGICAL-SUMMARY-001", "E-LOGICAL-TEST-002"],
            "findings": [{
                "text": "The configured signed 32-bit overflow case failed.",
                "kind": "observed",
                "evidence_ids": ["E-LOGICAL-TEST-002"],
            }],
            "contradictions": [],
            "recommendations": [{
                "text": "Use a 64-bit sum type to meet the problem contract.",
                "basis": "rubric",
                "evidence_ids": ["E-RULE-CONTRACT-001", "E-LOGICAL-TEST-002"],
            }],
            "uncertainties": [],
            "evidence_references": ["E-LOGICAL-TEST-002"],
        }

    def test_ledger_classifies_overflow_test_and_has_stable_ids(self):
        record = self.by_id["E-LOGICAL-TEST-002"]
        self.assertIn("signed_int32_overflow", record["facts"]["tags"])
        self.assertEqual(record["facts"]["passed"], False)
        self.assertIn("E-SOURCE-L001", self.by_id)

    def test_valid_claims_require_known_evidence_ids(self):
        cleaned, validation = validate_review_payload(self._review(), self.ledger, self.rules)
        self.assertIsNotNone(cleaned)
        self.assertEqual(validation["status"], "completed")
        self.assertEqual(validation["verified_claim_count"], 2)

    def test_positive_overflow_claim_without_passed_overflow_tests_is_rejected(self):
        review = self._review()
        review["summary"] = "The program correctly handles integer overflow."
        cleaned, validation = validate_review_payload(review, self.ledger, self.rules)
        self.assertIsNone(cleaned)
        self.assertIn("overflow", validation["reason"])

    def test_claim_that_overflow_tests_passed_is_rejected_when_an_overflow_case_failed(self):
        review = self._review()
        review["findings"] = [{
            "text": "The signed integer overflow test passed.",
            "kind": "observed",
            "evidence_ids": ["E-LOGICAL-TEST-002"],
        }]
        cleaned, validation = validate_review_payload(review, self.ledger, self.rules)
        self.assertIsNotNone(cleaned)
        self.assertEqual(validation["rejected_claim_count"], 1)
        self.assertFalse(cleaned["findings"])

    def test_helper_recommendation_conflicting_with_rules_is_rejected(self):
        review = self._review()
        review["recommendations"] = [{
            "text": "Add a helper function to improve modularity.",
            "basis": "optional",
            "evidence_ids": ["E-RULE-MODULARITY-001", "E-MODULARITY-001"],
        }]
        cleaned, validation = validate_review_payload(review, self.ledger, self.rules)
        self.assertIsNotNone(cleaned)
        self.assertEqual(validation["status"], "completed_with_warnings")
        self.assertEqual(cleaned["recommendations"], [])
        self.assertEqual(validation["rejected_claim_count"], 1)

    def test_claim_with_unknown_evidence_is_rejected(self):
        review = self._review()
        review["findings"][0]["evidence_ids"] = ["E-FAKE-001"]
        cleaned, validation = validate_review_payload(review, self.ledger, self.rules)
        self.assertIsNotNone(cleaned)
        self.assertEqual(validation["rejected_claim_count"], 1)
        self.assertEqual(cleaned["findings"], [])


if __name__ == "__main__":
    unittest.main()
