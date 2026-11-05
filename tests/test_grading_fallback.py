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


    def test_pre_generated_suite_is_used_without_running_generator_again(self):
        import sys
        import types

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "main.c"
            source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
            rules_path = root / "rules.json"
            rules_path.write_text("{}\n", encoding="utf-8")
            suite_path = root / "suite.json"
            suite_path.write_text(json.dumps({"test_cases": []}), encoding="utf-8")
            testset_path = root / "all_cases.json"
            testset_path.write_text(json.dumps([{"id": "all-1", "input": "", "expected": ""}]), encoding="utf-8")

            class FakeSandbox:
                image = "test-image"

                def __init__(self, **_kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

            sandbox_module = types.ModuleType("grading_agent.sandbox.docker_sandbox")
            sandbox_module.DockerSandbox = FakeSandbox
            metrics = [{
                "name": "correctness", "analyzer": "io_tests", "enabled": True,
                "weight": 1.0, "params": {"resolved_testset_path": str(testset_path)},
            }]
            rules = {
                "problem": "Read input and print it.",
                "time_limit_s": 2,
                "memory_limit_mb": 256,
                "submission": {"resolved_source_path": str(source)},
                "rules_path": str(rules_path),
                "project_root": str(root),
                "metrics": metrics,
                "test_generation": {
                    "enabled": True,
                    "pre_generated_suite": {
                        "status": "completed",
                        "suite_path": str(suite_path),
                        "metric_testset_paths": {"correctness": str(testset_path)},
                        "instructor_test_count": 1,
                        "generated_test_count": 12,
                        "requested_generated_test_count": 12,
                        "total_test_count": 13,
                    },
                },
            }

            with patch.dict(sys.modules, {"grading_agent.sandbox.docker_sandbox": sandbox_module}), \
                 patch("grading_agent.main._collect_tool_versions", return_value={"gcc": "fake"}), \
                 patch("grading_agent.main.run_preflight", return_value={
                     "status": "passed", "compiled": True, "smoke_tested": False,
                 }), \
                 patch("grading_agent.main.run_analyzer", return_value={
                     "status": "completed", "score": 100.0, "results": [],
                 }) as analyzer, \
                 patch("grading_agent.testing.test_generation.prepare_assignment_test_suite", side_effect=AssertionError("grading must not regenerate")):
                report = run_configured_analyzers(rules)

        self.assertEqual(report["test_generation"]["status"], "completed")
        self.assertEqual(report["test_generation"]["source"], "pre_generated_by_ui")
        self.assertEqual(report["test_generation"]["generated_test_count"], 12)
        analyzer.assert_called_once()
        self.assertEqual(analyzer.call_args.kwargs["metric"]["params"]["resolved_testset_path"], str(testset_path))


if __name__ == "__main__":
    unittest.main()
