"""Unit tests for the instructor rule loader (run with python -m unittest)."""

import json
import tempfile
import unittest
from pathlib import Path

from grading_agent.config.rule_loader import load_rules, RuleValidationError


class RuleLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "examples/sum").mkdir(parents=True)
        (self.root / "tests/sum").mkdir(parents=True)
        (self.root / "examples/sum/main.c").write_text(
            "int main(void) { return 0; }\n", encoding="utf-8"
        )
        (self.root / "tests/sum/testcases.json").write_text(
            json.dumps([{"input": "1 2\n", "expected": "3\n"}]),
            encoding="utf-8",
        )
        self.rules = {
            "problem": "sum",
            "time_limit_s": 2,
            "memory_limit_mb": 512,
            "submission": {
                "language": "c",
                "source_file": "examples/sum/main.c",
            },
            "metrics": [
                {
                    "name": "correctness",
                    "analyzer": "io_tests",
                    "weight": 0.6,
                    "enabled": True,
                    "params": {"testset_path": "tests/sum/testcases.json"},
                },
                {
                    "name": "memory_safety",
                    "analyzer": "valgrind_memcheck",
                    "weight": 0.4,
                    "enabled": True,
                    "params": {"testset_path": "tests/sum/testcases.json"},
                },
            ],
        }
        self.rules_path = self.root / "rules.json"
        self._write_rules()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_rules(self):
        self.rules_path.write_text(
            json.dumps(self.rules), encoding="utf-8"
        )

    def test_loads_valid_rules_and_resolves_paths(self):
        loaded = load_rules(self.rules_path, self.root)
        self.assertEqual(loaded["problem"], "sum")
        self.assertEqual(len(loaded["metrics"]), 2)
        self.assertEqual(
            loaded["submission"]["resolved_source_path"],
            str((self.root / "examples/sum/main.c").resolve()),
        )
        self.assertAlmostEqual(
            sum(metric["weight"] for metric in loaded["metrics"]), 1.0
        )

    def test_rejects_weights_that_do_not_sum_to_one(self):
        self.rules["metrics"][0]["weight"] = 0.7
        self._write_rules()
        with self.assertRaises(RuleValidationError):
            load_rules(self.rules_path, self.root)

    def test_rejects_unknown_analyzer(self):
        self.rules["metrics"][0]["analyzer"] = "invented_analyzer"
        self._write_rules()
        with self.assertRaises(RuleValidationError):
            load_rules(self.rules_path, self.root)

    def test_rejects_path_traversal(self):
        self.rules["submission"]["source_file"] = "../outside.c"
        self._write_rules()
        with self.assertRaises(RuleValidationError):
            load_rules(self.rules_path, self.root)


    def test_memory_safety_uses_default_component_weights(self):
        self.rules["metrics"][1]["analyzer"] = "memory_safety"
        self.rules["metrics"][1]["params"] = {
            "testset_path": "tests/sum/testcases.json"
        }
        self._write_rules()

        loaded = load_rules(self.rules_path, self.root)
        memory_metric = loaded["metrics"][1]
        self.assertAlmostEqual(
            memory_metric["params"]["valgrind_weight"], 0.70
        )
        self.assertAlmostEqual(
            memory_metric["params"]["llm_audit_weight"], 0.30
        )
        self.assertEqual(
            memory_metric["params"]["severity_penalties"],
            {
                "critical": 40.0,
                "high": 25.0,
                "medium": 10.0,
                "low": 3.0,
            },
        )

    def test_memory_safety_complements_single_explicit_subweight(self):
        self.rules["metrics"][1]["analyzer"] = "memory_safety"
        self.rules["metrics"][1]["params"] = {
            "testset_path": "tests/sum/testcases.json",
            "valgrind_weight": 0.80,
        }
        self._write_rules()

        loaded = load_rules(self.rules_path, self.root)
        params = loaded["metrics"][1]["params"]
        self.assertAlmostEqual(params["valgrind_weight"], 0.80)
        self.assertAlmostEqual(params["llm_audit_weight"], 0.20)

    def test_memory_safety_rejects_component_weights_not_summing_to_one(self):
        self.rules["metrics"][1]["analyzer"] = "memory_safety"
        self.rules["metrics"][1]["params"] = {
            "testset_path": "tests/sum/testcases.json",
            "valgrind_weight": 0.70,
            "llm_audit_weight": 0.20,
        }
        self._write_rules()

        with self.assertRaises(RuleValidationError):
            load_rules(self.rules_path, self.root)

    def test_legacy_valgrind_analyzer_is_normalized_to_composite(self):
        loaded = load_rules(self.rules_path, self.root)
        self.assertEqual(loaded["metrics"][1]["analyzer"], "memory_safety")
        self.assertIn("llm_audit_weight", loaded["metrics"][1]["params"])

    def test_rejects_invalid_test_case_schema(self):
        (self.root / "tests/sum/testcases.json").write_text(
            json.dumps([{"input": 123, "expected": "3"}]),
            encoding="utf-8",
        )
        with self.assertRaises(RuleValidationError):
            load_rules(self.rules_path, self.root)


if __name__ == "__main__":
    unittest.main()
