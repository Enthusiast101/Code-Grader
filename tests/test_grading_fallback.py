import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grading_agent.main import grade_with_rules_file, run_configured_analyzers, _review_with_local_llm


class GradingFlowTests(unittest.TestCase):
    def _make_project(self, root: Path) -> tuple[Path, Path]:
        source = root / "main.c"
        source.write_text(
            "int helper(void) { return 7; }\n"
            "int main(void) { return helper(); }\n",
            encoding="utf-8",
        )
        rules_file = root / "rules.json"
        rules = {
            "problem": "fallback-test",
            "time_limit_s": 5,
            "memory_limit_mb": 512,
            "submission": {"language": "c", "source_file": "main.c"},
            "metrics": [
                {
                    "name": "modularity", "analyzer": "modularity", "enabled": True,
                    "weight": 0.5, "params": {"min_functions": 2, "expect_helpers": True},
                },
                {
                    "name": "syntax", "analyzer": "compiler_diagnostics", "enabled": True,
                    "weight": 0.5, "params": {},
                },
            ],
        }
        rules_file.write_text(json.dumps(rules), encoding="utf-8")
        return source, rules_file

    def _block_docker_import(self, name, *args, **kwargs):
        if name == "grading_agent.sandbox.docker_sandbox":
            raise ImportError("simulated missing Docker SDK")
        return self._original_import(name, *args, **kwargs)

    @patch("grading_agent.main.evaluate_metrics_with_llm")
    @patch("grading_agent.main._is_llm_server_ready", return_value=True)
    def test_rsi_scores_metrics_without_docker(self, _server_ready, evaluator):
        evaluator.return_value = {
            "status": "completed",
            "metrics": [{"name": "correctness", "score": 91.0, "rationale": "Source and available evidence reviewed."}],
        }
        report = {
            "source_code": "int main(void) { return 0; }",
            "configuration": {"enabled_metrics": []},
        }
        rules = {"metrics": [{"name": "correctness", "enabled": True, "weight": 1.0}]}
        result = _review_with_local_llm(report, rules=rules)
        self.assertEqual(result["status"], "completed")
        evaluator.assert_called_once_with(report, rules)

    def test_source_analyzers_continue_when_docker_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, rules_file = self._make_project(root)
            rules = json.loads(rules_file.read_text())
            rules.update({
                "submission": {**rules["submission"], "resolved_source_path": str(root / "main.c")},
                "rules_path": str(rules_file),
                "project_root": str(root),
                "metrics": [
                    {**metric, "params": dict(metric["params"])}
                    for metric in rules["metrics"]
                ],
            })
            self._original_import = builtins.__import__
            with patch("builtins.__import__", side_effect=self._block_docker_import):
                report = run_configured_analyzers(rules)
        self.assertEqual(report["sandbox"]["status"], "unavailable")
        self.assertEqual(report["modularity"]["status"], "completed")
        self.assertEqual(report["syntax"]["status"], "unavailable")
        self.assertEqual(report["preflight"]["status"], "unavailable")

    @patch("grading_agent.main._review_with_local_llm", side_effect=RuntimeError("model unavailable"))
    def test_llm_failure_still_returns_analyzer_weighted_grade(self, _llm_rsi):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, rules_file = self._make_project(root)
            self._original_import = builtins.__import__
            with patch("builtins.__import__", side_effect=self._block_docker_import):
                report, _, _ = grade_with_rules_file(
                    rules_file, project_root=root, report_dir=root / "reports"
                )
        self.assertEqual(report["scoring"]["status"], "completed")
        self.assertGreaterEqual(report["scoring"]["score"], 0.0)
        self.assertEqual(report["modularity"]["scoring_basis"], "deterministic analyzer score (LLM RSI score unavailable)")
        self.assertEqual(report["llm_rsi_evaluation"]["status"], "unavailable")

    @patch("grading_agent.main._review_with_local_llm")
    def test_final_grade_uses_rsi_scores_and_python_weights(self, llm_rsi):
        llm_rsi.return_value = {
            "status": "completed",
            "metrics": [
                {"name": "modularity", "score": 80.0, "rationale": "The code is adequately decomposed."},
                {"name": "syntax", "score": 90.0, "rationale": "Source inspection suggests valid C syntax."},
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, rules_file = self._make_project(root)
            report_dir = root / "reports"
            self._original_import = builtins.__import__
            with patch("builtins.__import__", side_effect=self._block_docker_import):
                report, json_path, text_path = grade_with_rules_file(
                    rules_file, project_root=root, report_dir=report_dir
                )
                self.assertTrue(json_path.is_file())
                self.assertTrue(text_path.is_file())

        llm_rsi.assert_called_once()
        self.assertEqual(report["modularity"]["analyzer_score"], 100.0)
        self.assertEqual(report["modularity"]["llm_rsi_score"], 80.0)
        self.assertEqual(report["modularity"]["score"], 80.0)
        self.assertIsNone(report["syntax"]["analyzer_score"])
        self.assertEqual(report["syntax"]["llm_rsi_score"], 90.0)
        self.assertEqual(report["syntax"]["score"], 90.0)
        self.assertEqual(report["scoring"]["status"], "completed")
        self.assertEqual(report["scoring"]["score"], 85.0)


if __name__ == "__main__":
    unittest.main()
