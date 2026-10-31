import unittest
from unittest.mock import patch

from grading_agent.analyzers.registry import ANALYZER_REGISTRY, run_analyzer


class AnalyzerRegistryTests(unittest.TestCase):
    def test_expected_analyzers_are_registered(self):
        self.assertEqual(
            set(ANALYZER_REGISTRY),
            {"io_tests", "memory_safety", "static_analysis"},
        )

    def test_dispatches_registered_handler_and_adds_metric_metadata(self):
        metric = {
            "name": "correctness",
            "analyzer": "io_tests",
            "weight": 0.6,
            "params": {},
        }
        expected_result = {"compiled": True, "tests_passed": 1}

        with patch.dict(
            ANALYZER_REGISTRY,
            {"io_tests": lambda context: dict(expected_result)},
        ):
            result = run_analyzer(
                "io_tests",
                sandbox=object(),
                metric=metric,
                rules={},
                source_code="int main(void) { return 0; }",
                source_filename="main.c",
            )

        self.assertTrue(result["compiled"])
        self.assertEqual(result["metric_name"], "correctness")
        self.assertEqual(result["analyzer"], "io_tests")
        self.assertEqual(result["weight"], 0.6)

    def test_unknown_analyzer_fails_with_available_names(self):
        with self.assertRaisesRegex(ValueError, "Registered analyzers"):
            run_analyzer(
                "unknown",
                sandbox=object(),
                metric={"name": "x", "weight": 1, "params": {}},
                rules={},
                source_code="",
                source_filename="main.c",
            )


if __name__ == "__main__":
    unittest.main()
