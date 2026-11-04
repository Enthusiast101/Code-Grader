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
        "inferred_contract": {
            "input_format": "Two signed integers separated by whitespace.",
            "output_format": "Print their sum followed by a newline.",
            "constraints": [],
            "semantics": ["Compute the mathematical sum of the two input integers."],
            "formatting": ["One integer result per output line."],
        },
        "reference_solution_c": "#include <stdio.h>\nint main(void){long long a,b;if(scanf(\"%lld%lld\",&a,&b)!=2)return 1;printf(\"%lld\\n\",a+b);return 0;}\n",
        "test_inputs": [
            {"id": "basic", "input": "2 3\n", "tags": ["normal"], "rationale": "Ordinary positive operands."},
            {"id": "negative", "input": "-4 7\n", "tags": ["negative"], "rationale": "Mixed-sign operands."},
        ],
        "coverage_claims": ["normal", "mixed_sign"],
    }


def test_submission_understanding_stage_reads_source_before_test_design():
    from grading_agent.testing.test_generation import analyze_submission_for_test_generation

    source = '#include <stdio.h>\nint main(void) { int n; scanf("%d", &n); printf("%d\\n", n); }\n'
    calls = []

    def request(messages, schema, stage):
        calls.append((messages, schema, stage))
        payload = json.loads(messages[1]["content"])
        assert payload["submission_source"] == source
        assert payload["assignment"] == "Read an integer and print it."
        assert "test_inputs" not in schema["properties"]
        return {
            "implementation_summary": "Reads one integer and prints it unchanged.",
            "input_parsing_and_validation": ["scanf return value is not checked."],
            "control_flow_and_data_types": ["Uses a signed int."],
            "likely_weak_points": ["Malformed input is not rejected."],
            "recommended_test_dimensions": ["Zero, negative, and integer limits."],
        }

    analysis = analyze_submission_for_test_generation(
        problem="Read an integer and print it.",
        problem_contract={"input_range": "signed 32-bit"},
        submission_source=source,
        request_json=request,
        submission_filename="student.c",
    )

    assert calls[0][2] == "assignment_submission_understanding_attempt_1"
    assert analysis["likely_weak_points"] == ["Malformed input is not rejected."]


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
        request_json=request, max_cases=2,
    )
    assert len(calls) == 3
    assert [x["pass"] for x in history] == [1, 2, 3]
    assert len(spec["test_inputs"]) == 2


def test_trusted_reference_generation_builds_requested_suite_in_small_batches():
    from grading_agent.testing.test_generation import generate_assignment_test_spec

    calls = []

    def request(messages, schema, stage):
        payload = json.loads(messages[1]["content"])
        requested = schema["properties"]["test_inputs"]["minItems"]
        assert schema["properties"]["test_inputs"]["maxItems"] == requested
        assert requested <= 4
        calls.append((payload, stage, requested))
        batch_number = len(calls)
        return {
            "test_inputs": [
                {
                    "id": f"case-{batch_number}-{index}",
                    "input": f"{batch_number * 10 + index} 1\n",
                    "tags": ["generated"],
                    "rationale": "Distinct valid input.",
                }
                for index in range(requested)
            ]
        }

    trusted_reference = (
        '#include <stdio.h>\n'
        'int main(void) { int a, b; if (scanf("%d%d", &a, &b) != 2) return 1; '
        'printf("%d\\n", a + b); return 0; }\n'
    )
    spec, history = generate_assignment_test_spec(
        problem="Read two integers and print their sum.",
        problem_contract={"sum_type": "int32"},
        request_json=request,
        passes=3,
        max_cases=5,
        trusted_reference_solution=trusted_reference,
        instructor_examples=[{"id": "teacher", "input": "1 2\n", "expected": "3\n"}],
    )

    assert len(spec["test_inputs"]) == 5
    assert all(case["input"] != "1 2\n" for case in spec["test_inputs"])
    assert len({case["input"] for case in spec["test_inputs"]}) == 5
    assert spec["reference_solution_c"] == trusted_reference
    assert [item["test_count"] for item in history] == [5, 5, 5]
    assert len(calls) == 6  # two small batches in each of three RSI passes
    assert all("preceding_complete_response" in payload for payload, _, _ in calls[2:])


def test_generation_fails_closed_on_ambiguous_contract():
    response = sample_response()
    response["ambiguities"] = ["The assignment does not define overflow behavior."]
    with pytest.raises(TestGenerationError, match="Unresolved assignment ambiguities"):
        generate_assignment_test_spec(
            problem="Read two integers from input and print their arithmetic sum.", problem_contract={},
            request_json=lambda *_: response,
        )


def test_generation_refuses_a_title_without_an_assignment_contract():
    with pytest.raises(TestGenerationError, match="only a title"):
        generate_assignment_test_spec(
            problem="grading", problem_contract={}, request_json=lambda *_: sample_response(),
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
            numbers = [int(value) for value in stdin.split()]
            output = f"{sum(numbers)}\n"
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
    calls = []
    def request(messages, schema, stage):
        calls.append(stage)
        requested = schema["properties"]["test_inputs"]["maxItems"]
        generated = dict(response)
        generated["test_inputs"] = [
            {
                "id": f"sum-{index}",
                "input": f"{index + 10} {index + 11}\n",
                "tags": ["edge"] if index == 0 else ["normal"],
                "rationale": "Unique pair of integers.",
            }
            for index in range(requested)
        ]
        return generated

    sandbox = FakeSandbox()
    first = prepare_assignment_test_suite(rules, sandbox=sandbox, request_json=request, project_root=tmp_path, model_id="qwen-test")
    assert first["status"] == "completed"
    assert first["source"] == "generated_and_frozen"
    assert first["total_test_count"] == 10
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
    assert second["total_test_count"] == 10


def test_prepare_suite_understands_submission_before_generating_additional_cases(tmp_path):
    from grading_agent.testing.test_generation import prepare_assignment_test_suite

    class FakeSandbox:
        def write_file(self, filename, content):
            self.source = content

        def run(self, command, timeout_s, stdin=None):
            if command[0] == "gcc":
                return {"output": "", "exit_code": 0, "timed_out": False}
            left, right = (int(value) for value in stdin.split())
            return {
                "output": f"{left + right}\n", "exit_code": 0,
                "timed_out": False, "output_may_be_truncated": False,
            }

    source = '#include <stdio.h>\nint main(void) { int a, b; scanf("%d%d", &a, &b); printf("%d\\n", a + b); }\n'
    reference = (
        '#include <stdio.h>\n'
        'int main(void) { int a, b; if (scanf("%d%d", &a, &b) != 2) return 1; '
        'printf("%d\\n", a + b); return 0; }\n'
    )
    (tmp_path / "reference.c").write_text(reference, encoding="utf-8")
    (tmp_path / "instructor.json").write_text(
        json.dumps([{"id": "teacher", "input": "1 2\n", "expected": "3\n"}]),
        encoding="utf-8",
    )
    rules = {
        "problem": "Read two integers and print their sum.",
        "problem_contract": {"input_format": "Two signed integers", "output_rule": "Their sum"},
        "time_limit_s": 2,
        "metrics": [{
            "name": "correctness", "analyzer": "io_tests", "enabled": True,
            "params": {"testset_path": "instructor.json", "resolved_testset_path": str(tmp_path / "instructor.json")},
        }],
        "test_generation": {
            "enabled": True, "rsi_passes": 1, "seed": 42,
            "max_cases": 2, "reference_solution_path": "reference.c",
        },
    }
    analysis = {
        "implementation_summary": "The submission scans two integers and adds them using int.",
        "input_parsing_and_validation": ["The scanf result is unchecked."],
        "control_flow_and_data_types": ["Two signed int values are added."],
        "likely_weak_points": ["Large values may overflow int."],
        "recommended_test_dimensions": ["Large positive and negative operands."],
    }
    stages = []
    generation_batch = 0

    def request(messages, schema, stage):
        nonlocal generation_batch
        stages.append(stage)
        payload = json.loads(messages[1]["content"])
        if stage.startswith("assignment_submission_understanding"):
            assert payload["submission_source"] == source
            assert "test_inputs" not in schema["properties"]
            return analysis
        assert "submitted_implementation_analysis" in payload
        assert payload["submitted_implementation_analysis"] == analysis
        generation_batch += 1
        requested = schema["properties"]["test_inputs"]["maxItems"]
        return {
            "test_inputs": [
                {
                    "id": f"generated-{index}",
                    "input": f"{20 + (generation_batch - 1) * 4 + index} {1 + index}\n",
                    "tags": ["boundary"],
                    "rationale": "Additional large-value case.",
                }
                for index in range(requested)
            ]
        }

    result = prepare_assignment_test_suite(
        rules, sandbox=FakeSandbox(), request_json=request, project_root=tmp_path,
        model_id="qwen-test", submission_source=source,
    )

    assert result["status"] == "completed"
    assert stages[0] == "assignment_submission_understanding_attempt_1"
    assert all(stage.startswith("assignment_test_generation_rsi_1_batch_") for stage in stages[1:])
    assert result["submission_analysis"] == analysis
    assert result["generated_test_count"] == 2
    assert result["instructor_test_count"] == 1
    frozen = json.loads(Path(result["suite_path"]).read_text(encoding="utf-8"))
    assert frozen["submission_analysis"] == analysis


def test_prepare_suite_binds_all_instructor_and_generated_cases_to_correctness(tmp_path):
    from grading_agent.testing.test_generation import prepare_assignment_test_suite

    class FakeSandbox:
        def write_file(self, filename, content):
            self.source = content

        def run(self, command, timeout_s, stdin=None):
            if command[0] == "gcc":
                return {"output": "", "exit_code": 0, "timed_out": False}
            numbers = [int(value) for value in stdin.split()]
            return {
                "output": f"{sum(numbers)}\n",
                "exit_code": 0,
                "timed_out": False,
                "output_may_be_truncated": False,
            }

    teacher_cases = [
        {"id": "teacher-a", "input": "1 2\n", "expected": "3\n"},
        {"id": "teacher-b", "input": "3 4\n", "expected": "7\n"},
    ]
    edge_cases = [
        {"id": "edge-a", "input": "1 2\n", "expected": "3\n"},
        {"id": "edge-b", "input": "-3 8\n", "expected": "5\n"},
    ]
    memory_cases = [
        {"id": "memory-a", "input": "3 4\n", "expected": "7\n"},
    ]
    for filename, cases in (
        ("correctness.json", teacher_cases),
        ("edge.json", edge_cases),
        ("memory.json", memory_cases),
    ):
        (tmp_path / filename).write_text(json.dumps(cases), encoding="utf-8")
    reference_path = tmp_path / "reference.c"
    reference_path.write_text(
        '#include <stdio.h>\n'
        'int main(void) { int a, b; if (scanf("%d%d", &a, &b) != 2) return 1; '
        'printf("%d\\n", a + b); return 0; }\n',
        encoding="utf-8",
    )

    rules = {
        "problem": "Read two integers and print their sum.",
        "problem_contract": {"sum_type": "int32"},
        "project_root": str(tmp_path),
        "time_limit_s": 2,
        "metrics": [
            {"name": "correctness", "analyzer": "io_tests", "enabled": True, "weight": 1.0,
             "params": {"testset_path": "correctness.json", "resolved_testset_path": str(tmp_path / "correctness.json")}},
            {"name": "logical", "analyzer": "edge_case_runner", "enabled": True,
             "params": {"testset_path": "edge.json", "resolved_testset_path": str(tmp_path / "edge.json")}},
            {"name": "memory_safety", "analyzer": "memory_safety", "enabled": True,
             "params": {"testset_path": "memory.json", "resolved_testset_path": str(tmp_path / "memory.json")}},
        ],
        "test_generation": {
            "enabled": True,
            "rsi_passes": 3,
            "seed": 42,
            "max_cases": 5,
            "reference_solution_path": "reference.c",
        },
    }

    request_number = 0

    def request(messages, schema, stage):
        nonlocal request_number
        request_number += 1
        requested = schema["properties"]["test_inputs"]["minItems"]
        return {
            "test_inputs": [
                {
                    "id": f"generated-{request_number}-{index}",
                    "input": f"{request_number * 10 + index} 1\n",
                    "tags": ["edge"] if index == 0 else ["normal"],
                    "rationale": "Additional valid input.",
                }
                for index in range(requested)
            ]
        }

    result = prepare_assignment_test_suite(
        rules,
        sandbox=FakeSandbox(),
        request_json=request,
        project_root=tmp_path,
        model_id="qwen-test",
    )

    correctness_path = Path(result["metric_testset_paths"]["correctness"])
    correctness_cases = json.loads(correctness_path.read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["instructor_test_count"] == 3  # one duplicate input is merged
    assert result["generated_test_count"] == 5
    assert result["total_test_count"] == 8
    assert len(correctness_cases) == 8
    assert sum(case["origin"] == "instructor" for case in correctness_cases) == 3
    assert sum(case["origin"] == "llm_generated_reference_oracle" for case in correctness_cases) == 5
    assert rules["metrics"][0]["params"]["resolved_testset_path"] == str(correctness_path)

    # Exercise the real correctness analyzer through its registry binding:
    # it must execute the combined instructor + generated JSON, not just
    # the original correctness.json file.
    from grading_agent.analyzers.registry import run_analyzer

    correctness_result = run_analyzer(
        "io_tests",
        sandbox=FakeSandbox(),
        metric=rules["metrics"][0],
        rules=rules,
        source_code="int main(void) { return 0; }",
        source_filename="submission.c",
    )
    assert correctness_result["tests_total"] == 8
    assert correctness_result["tests_passed"] == 8


def test_prepare_suite_fails_if_required_logical_cases_are_not_generated(tmp_path):
    from grading_agent.testing.test_generation import prepare_assignment_test_suite

    class FakeSandbox:
        def write_file(self, filename, content): pass
        def run(self, command, timeout_s, stdin=None):
            if command[0] == "gcc":
                return {"output": "", "exit_code": 0, "timed_out": False}
            return {"output": "5\n", "exit_code": 0, "timed_out": False, "output_may_be_truncated": False}

    response = sample_response()
    rules = {
        "problem": "Read two integers and print their sum.", "problem_contract": {},
        "project_root": str(tmp_path), "time_limit_s": 2,
        "metrics": [{"name": "logical", "analyzer": "edge_case_runner", "enabled": True, "params": {}}],
        "test_generation": {"enabled": True, "rsi_passes": 3, "max_cases": 10},
    }
    with pytest.raises(TestGenerationError, match="no identifiable edge/boundary"):
        def request(messages, schema, stage):
            requested = schema["properties"]["test_inputs"]["maxItems"]
            generated = dict(response)
            generated["test_inputs"] = [
                {"id": f"normal-{index}", "input": f"{index + 1} {index + 2}\n",
                 "tags": ["normal"], "rationale": "Normal pair."}
                for index in range(requested)
            ]
            return generated

        prepare_assignment_test_suite(rules, sandbox=FakeSandbox(), request_json=request, project_root=tmp_path, model_id="qwen-test")


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
