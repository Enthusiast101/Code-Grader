from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grading_agent.testing.test_generation import (
    TestGenerationError, assignment_fingerprint, generate_assignment_test_spec,
    materialize_test_cases, write_frozen_suite, load_frozen_suite,
)


def sample_response():
    return {
        "spec_summary": "Read two integers and print their sum.",
        "ambiguities": [],
        "reference_solution_c": "#include <stdio.h>\nint main(void){long long a,b;if(scanf(\"%lld%lld\",&a,&b)!=2)return 1;printf(\"%lld\\n\",a+b);return 0;}\n",
        "test_inputs": [
            {"id": "basic", "input": "2 3\n", "tags": ["normal"], "rationale": "Ordinary positive operands."},
            {"id": "negative", "input": "-4 7\n", "tags": ["negative"], "rationale": "Mixed-sign operands."},
        ],
        "coverage_claims": ["normal", "mixed_sign"],
    }


def test_generation_runs_three_sequential_passes_and_rechecks_prior_response():
    calls = []
    def request(messages, schema, stage):
        calls.append((messages, schema, stage))
        payload = json.loads(messages[1]["content"])
        if len(calls) > 1:
            assert "preceding_complete_response" in payload
            assert payload["preceding_complete_response"]["spec_summary"] == "Read two integers and print their sum."
        return sample_response()
    spec, history = generate_assignment_test_spec(
        problem="Read two integers and print their sum.", problem_contract={"sum_type": "int64"},
        request_json=request,
    )
    assert len(calls) == 3
    assert [x["pass"] for x in history] == [1, 2, 3]
    assert len(spec["test_inputs"]) == 2


def test_generation_fails_closed_on_ambiguous_contract():
    response = sample_response()
    response["ambiguities"] = ["The assignment does not define overflow behavior."]
    with pytest.raises(TestGenerationError, match="Unresolved assignment ambiguities"):
        generate_assignment_test_spec(
            problem="Add two integers.", problem_contract={},
            request_json=lambda *_: response,
        )


def test_instructor_cases_are_preserved_and_generated_outputs_use_oracle():
    spec = sample_response()
    merged = materialize_test_cases(
        spec,
        instructor_cases=[{"id": "teacher-1", "input": "2 3\n", "expected": "5\n"}],
        run_reference=lambda source, stdin: {"stdout": "3\n" if stdin.startswith("-4") else "5\n", "exit_code": 0, "timed_out": False},
    )
    assert len(merged) == 2  # duplicate generated input did not replace instructor case
    assert merged[0]["origin"] == "instructor"
    assert merged[1]["origin"] == "llm_generated_reference_oracle"
    assert merged[1]["expected"] == "3\n"


def test_generated_oracle_must_agree_with_instructor_expected_output():
    with pytest.raises(TestGenerationError, match="disagrees with instructor"):
        materialize_test_cases(
            sample_response(),
            instructor_cases=[{"id": "teacher", "input": "2 3\n", "expected": "6\n"}],
            run_reference=lambda source, stdin: {"stdout": "5\n", "exit_code": 0, "timed_out": False},
        )


def test_reference_oracle_failure_is_not_treated_as_expected_output():
    with pytest.raises(TestGenerationError, match="expected output is untrusted"):
        materialize_test_cases(
            sample_response(),
            run_reference=lambda *_: {"stdout": "", "exit_code": 1, "timed_out": False},
        )


def test_assignment_fingerprint_and_frozen_suite_are_stable(tmp_path):
    fingerprint = assignment_fingerprint(
        problem="Add two numbers", problem_contract={"type": "int"}, model_id="Qwen3.5-2B-Q4_K_M"
    )
    cases = [{"id": "t1", "input": "1 2\n", "expected": "3\n"}]
    spec = sample_response()
    path = write_frozen_suite(tmp_path, fingerprint=fingerprint, cases=cases, generated_spec=spec, model_id="Qwen3.5-2B-Q4_K_M")
    assert load_frozen_suite(path, fingerprint) == cases
    with pytest.raises(TestGenerationError, match="does not match"):
        load_frozen_suite(path, "different-fingerprint")
    with pytest.raises(TestGenerationError, match="differs"):
        write_frozen_suite(tmp_path, fingerprint=fingerprint, cases=[], generated_spec=spec, model_id="Qwen3.5-2B-Q4_K_M")


def test_prepare_suite_generates_once_then_reuses_frozen_artifact(tmp_path):
    from grading_agent.testing.test_generation import prepare_assignment_test_suite

    class FakeSandbox:
        def __init__(self):
            self.source = ""
        def write_file(self, filename, content):
            self.source = content
        def run(self, command, timeout_s, stdin=None):
            if command[0] == "gcc":
                return {"output": "", "exit_code": 0, "timed_out": False}
            output = "5\n" if stdin.startswith("2") else "3\n"
            return {"output": output, "exit_code": 0, "timed_out": False, "output_may_be_truncated": False}

    rules = {
        "problem": "Read two integers and print their sum.",
        "problem_contract": {"sum_type": "int64"},
        "project_root": str(tmp_path),
        "time_limit_s": 2,
        "metrics": [
            {"name": "correctness", "analyzer": "io_tests", "enabled": True, "params": {}},
            {"name": "logical", "analyzer": "edge_case_runner", "enabled": True, "params": {}},
            {"name": "memory_safety", "analyzer": "memory_safety", "enabled": True, "params": {}},
        ],
        "test_generation": {"enabled": True, "rsi_passes": 3, "seed": 42, "max_cases": 10},
    }
    response = sample_response()
    response["test_inputs"][0]["input"] = "2 3\n"
    response["test_inputs"][1]["input"] = "-4 7\n"
    calls = []
    def request(messages, schema, stage):
        calls.append(stage)
        return response

    sandbox = FakeSandbox()
    first = prepare_assignment_test_suite(rules, sandbox=sandbox, request_json=request, project_root=tmp_path, model_id="qwen-test")
    assert first["status"] == "completed"
    assert first["source"] == "generated_and_frozen"
    assert first["total_test_count"] == 2
    assert first["edge_test_count"] == 1
    assert len(calls) == 3
    assert all(Path(v).is_file() for v in first["metric_testset_paths"].values())

    second = prepare_assignment_test_suite(
        rules, sandbox=sandbox,
        request_json=lambda *_: (_ for _ in ()).throw(AssertionError("must reuse frozen suite")),
        project_root=tmp_path, model_id="qwen-test",
    )
    assert second["source"] == "frozen_cache"
    assert second["assignment_fingerprint"] == first["assignment_fingerprint"]
    assert second["total_test_count"] == 2


def test_prepare_suite_fails_if_required_logical_cases_are_not_generated(tmp_path):
    from grading_agent.testing.test_generation import prepare_assignment_test_suite

    class FakeSandbox:
        def write_file(self, filename, content): pass
        def run(self, command, timeout_s, stdin=None):
            if command[0] == "gcc":
                return {"output": "", "exit_code": 0, "timed_out": False}
            return {"output": "5\n", "exit_code": 0, "timed_out": False, "output_may_be_truncated": False}

    response = sample_response()
    response["test_inputs"] = [
        {"id": "normal", "input": "2 3\n", "tags": ["normal"], "rationale": "Normal"},
    ]
    rules = {
        "problem": "Read two integers and print their sum.", "problem_contract": {},
        "project_root": str(tmp_path), "time_limit_s": 2,
        "metrics": [{"name": "logical", "analyzer": "edge_case_runner", "enabled": True, "params": {}}],
        "test_generation": {"enabled": True, "rsi_passes": 3, "max_cases": 10},
    }
    with pytest.raises(TestGenerationError, match="no identifiable edge/boundary"):
        prepare_assignment_test_suite(rules, sandbox=FakeSandbox(), request_json=lambda *_: response, project_root=tmp_path, model_id="qwen-test")


def test_fingerprint_changes_when_generation_configuration_changes():
    base = assignment_fingerprint(
        problem="Sort integers.", problem_contract={"max_n": 100},
        model_id="Qwen3.5-2B-Q4_K_M", generation_seed=42,
        rsi_passes=3, max_cases=60, temperature=0.0,
    )
    changed = assignment_fingerprint(
        problem="Sort integers.", problem_contract={"max_n": 100},
        model_id="Qwen3.5-2B-Q4_K_M", generation_seed=42,
        rsi_passes=2, max_cases=60, temperature=0.0,
    )
    assert base != changed


def test_strict_mode_refuses_to_regenerate_a_missing_frozen_suite(tmp_path):
    from grading_agent.testing.test_generation import prepare_assignment_test_suite

    rules = {
        "problem": "Read an integer and print it.", "problem_contract": {},
        "project_root": str(tmp_path), "time_limit_s": 2,
        "metrics": [{"name": "correctness", "analyzer": "io_tests", "enabled": True, "params": {}}],
        "test_generation": {"enabled": True, "require_frozen_suite": True, "rsi_passes": 3, "seed": 42, "temperature": 0.0},
    }
    with pytest.raises(TestGenerationError, match="requires the pre-generated assignment suite"):
        prepare_assignment_test_suite(
            rules, sandbox=object(), request_json=lambda *_: pytest.fail("must not regenerate"),
            project_root=tmp_path, model_id="qwen-test", generation_seed=42, temperature=0.0,
        )
