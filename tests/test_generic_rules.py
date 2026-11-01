from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from grading_agent.config.rule_loader import load_rules
from grading_agent.analyzers.registry import AnalyzerContext
from grading_agent.analyzers.io_tests import run_io_tests
from grading_agent.analyzers.runtime_profile import run_runtime_profile
from grading_agent.analyzers.memory_safety import run_memory_safety

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_rules_template_is_assignment_neutral_and_requires_prompt():
    template = json.loads((PROJECT_ROOT / "rules.json").read_text(encoding="utf-8"))
    assert "addition" not in template["problem"].lower()
    assert "sum" not in template["problem"].lower()
    assert template["problem_contract"] == {}
    assert sum(metric["weight"] for metric in template["metrics"] if metric["enabled"]) == 1.0

    # A template with no task prompt must not be used accidentally.
    try:
        load_rules(PROJECT_ROOT / "rules.json", PROJECT_ROOT)
    except ValueError as exc:
        assert "placeholder" in str(exc).lower()
    else:
        raise AssertionError("Unedited task placeholder should be rejected")

    template["problem"] = "Write a program that sorts an integer sequence in ascending order."
    temp_root = PROJECT_ROOT / "tests" / ".generic_rules_tmp"
    temp_root.mkdir(exist_ok=True)
    (temp_root / "submission.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
    template["submission"]["source_file"] = "submission.c"
    rules_path = temp_root / "rules.json"
    rules_path.write_text(json.dumps(template), encoding="utf-8")
    try:
        rules = load_rules(rules_path, temp_root)
    finally:
        for child in temp_root.iterdir():
            child.unlink()
        temp_root.rmdir()

    for metric in rules["metrics"]:
        if metric["analyzer"] in {"io_tests", "edge_case_runner", "runtime_profile", "memory_safety"}:
            assert "resolved_testset_path" not in metric["params"]


def test_task_contract_accepts_non_addition_assignment(tmp_path):
    source = tmp_path / "submission.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    rules = json.loads((PROJECT_ROOT / "rules.json").read_text(encoding="utf-8"))
    rules["problem"] = "Sort a sequence of integers in nondecreasing order."
    rules["problem_contract"] = {
        "input_format": "n followed by n integers",
        "output_format": "sorted integers",
        "constraints": {"n_min": 0, "n_max": 100000},
        "stability_required": False,
    }
    rules["submission"]["source_file"] = str(source.relative_to(tmp_path))
    # Use a rules path within tmp_path; project-relative source validation remains enforced.
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules), encoding="utf-8")
    loaded = load_rules(rules_path, tmp_path)
    assert loaded["problem"].startswith("Sort a sequence")
    assert loaded["problem_contract"]["constraints"]["n_max"] == 100000


def test_missing_test_path_returns_empty_test_cases_in_context():
    context = AnalyzerContext(
        sandbox=None,
        metric={"name": "correctness", "analyzer": "io_tests", "params": {}},
        rules={},
        source_code="int main(void) { return 0; }",
        source_filename="submission.c",
    )
    assert context.get_test_cases() == []


def test_io_and_runtime_analyzers_do_not_assign_false_zero_without_cases():
    sandbox = Mock()
    io_result = run_io_tests(sandbox, "int main(void) { return 0; }", [])
    runtime_result = run_runtime_profile(sandbox, "int main(void) { return 0; }", [])
    assert io_result["status"] == "inconclusive" and io_result["score"] is None
    assert runtime_result["status"] == "inconclusive" and runtime_result["score"] is None
    sandbox.write_file.assert_not_called()
    sandbox.run.assert_not_called()


def test_memory_safety_still_runs_gcc_analyzer_without_runtime_cases():
    class FakeSandbox:
        def __init__(self):
            self.commands = []

        def write_file(self, filename, content):
            return None

        def run(self, command, timeout_s):
            self.commands.append(command)
            if "-fanalyzer" in command:
                return {
                    "output": "submission.c:4:9: warning: use after free [CWE-416]\n",
                    "exit_code": 0,
                    "timed_out": False,
                }
            return {"output": "", "exit_code": 0, "timed_out": False}

    sandbox = FakeSandbox()
    result = run_memory_safety(sandbox, "int main(void) { return 0; }", [])
    assert result["status"] == "inconclusive"
    assert result["score"] is None
    assert result["gcc_analyzer"]["status"] == "completed"
    assert len(sandbox.commands) == 2
    assert "-fanalyzer" in sandbox.commands[1]

def test_configured_assignment_testset_is_still_resolved_and_validated(tmp_path):
    source = tmp_path / "submission.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    test_dir = tmp_path / "tests" / "sorting"
    test_dir.mkdir(parents=True)
    case_path = test_dir / "io_cases.json"
    case_path.write_text(json.dumps([{"id": "empty", "input": "0\n", "expected": "\n"}]), encoding="utf-8")

    rules = json.loads((PROJECT_ROOT / "rules.json").read_text(encoding="utf-8"))
    rules["problem"] = "Read a sequence and print its values in sorted order."
    rules["submission"]["source_file"] = "submission.c"
    rules["metrics"][0]["params"]["testset_path"] = "tests/sorting/io_cases.json"
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules), encoding="utf-8")

    loaded = load_rules(rules_path, tmp_path)
    assert loaded["metrics"][0]["params"]["resolved_testset_path"] == str(case_path.resolve())

