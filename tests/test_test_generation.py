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
    TestGenerationError,
    assignment_fingerprint,
    generate_assignment_test_spec,
    analyze_submission_for_test_generation,
    materialize_test_cases,
    write_frozen_suite,
    load_frozen_suite,
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
        "test_inputs": [
            {"id": "basic", "input": "2 3\n", "expected_output": "5\n", "tags": ["normal"]},
            {"id": "negative", "input": "-4 7\n", "expected_output": "3\n", "tags": ["negative", "edge"]},
        ],
    }


def verifier_response(cases):
    verifications = []
    for case in cases:
        values = [int(token) for token in case["input"].split()]
        verifications.append({
            "id": case["id"],
            "input": case["input"],
            "input_is_valid": True,
            "expected_output": f"{sum(values)}\n",
            "reason": "",
        })
    return {"verifications": verifications}


def test_submission_understanding_stage_reads_source_before_test_design():
    source = '#include <stdio.h>\nint main(void) { int n; scanf("%d", &n); printf("%d\\n", n); }\n'
    calls = []

    def request(messages, schema, stage):
        calls.append((messages, schema, stage))
        payload = json.loads(messages[1]["content"])
        assert payload["submission_source"] == source
        assert payload["assignment"] == "Read an integer and print it."
        assert schema["properties"]["implementation_summary"]["maxLength"] == 700
        assert schema["properties"]["input_parsing_and_validation"]["maxItems"] == 8
        assert schema["properties"]["likely_weak_points"]["maxItems"] == 6
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

    assert len(calls) == 1
    assert calls[0][2] == "assignment_submission_understanding_single_pass"
    assert analysis["likely_weak_points"] == ["Malformed input is not rejected."]


def test_generation_proposes_pairs_then_second_llm_independently_verifies_them():
    calls = []
    teacher = {"id": "teacher-example", "input": "1 2\n", "expected": "3\n"}
    response = sample_response()

    def request(messages, schema, stage):
        payload = json.loads(messages[1]["content"])
        calls.append(stage)
        if stage == "assignment_test_generation_single_pass":
            assert "expected_output" in schema["properties"]["test_inputs"]["items"]["properties"]
            assert payload["instructor_examples_as_hints"] == [{
                "id": "teacher-example", "input": "1 2\n", "expected_output": "3\n"
            }]
            assert "Do not write, propose, or return C code" in messages[0]["content"]
            return response
        assert stage == "assignment_expected_output_verification_single_pass"
        assert payload["instructor_examples_as_format_and_behavior_hints"][0]["expected_output"] == "3\n"
        assert payload["test_inputs_to_solve"] == [
            {"id": "basic", "input": "2 3\n"},
            {"id": "negative", "input": "-4 7\n"},
        ]
        # The verifier must not see the first model's expected outputs or student code.
        assert "expected_output" not in json.dumps(payload["test_inputs_to_solve"])
        assert "submitted_implementation_analysis" not in payload
        assert "submission_source" not in payload
        return verifier_response(payload["test_inputs_to_solve"])

    analysis = {
        "implementation_summary": "Reads two integers and prints their sum.",
        "likely_weak_points": ["May mishandle negative numbers."],
    }
    spec, history = generate_assignment_test_spec(
        problem="Read two integers and print their sum.",
        problem_contract={"sum_type": "int64"},
        request_json=request,
        max_cases=2,
        instructor_examples=[teacher],
        submission_analysis=analysis,
    )

    assert calls == [
        "assignment_test_generation_single_pass",
        "assignment_expected_output_verification_single_pass",
    ]
    assert [case["expected_output"] for case in spec["test_inputs"]] == ["5\n", "3\n"]
    assert spec["generation_mode"] == "generate_then_independently_verify"
    assert spec["verification_summary"]["verified"] == 2
    assert spec["verification_response_count"] == 1
    assert history[0]["verification"]["method"] == "independent_second_llm_call"



@pytest.mark.parametrize(
    "bad_field, bad_value, expected_reason",
    [
        ("input", None, "input is missing or is not text"),
        ("input", "x" * 20_001, "input exceeds 20000 characters"),
        ("expected_output", None, "expected_output is missing or is not text"),
        ("expected_output", "x" * 20_001, "expected_output exceeds 20000 characters"),
    ],
)
def test_malformed_candidates_are_rejected_without_discarding_valid_cases(
    bad_field, bad_value, expected_reason
):
    generated = sample_response()
    generated["test_inputs"][0]["id"] = "normal"
    generated["test_inputs"][0][bad_field] = bad_value

    def request(messages, schema, stage):
        if stage == "assignment_test_generation_single_pass":
            input_schema = schema["properties"]["test_inputs"]["items"]["properties"]["input"]
            assert "minLength" not in input_schema
            return generated
        payload = json.loads(messages[1]["content"])
        assert payload["test_inputs_to_solve"] == [
            {"id": "negative", "input": "-4 7\n"}
        ]
        return verifier_response(payload["test_inputs_to_solve"])

    spec, history = generate_assignment_test_spec(
        problem="Read two integers and print their sum.",
        problem_contract={},
        request_json=request,
        max_cases=1,
    )

    assert [case["id"] for case in spec["test_inputs"]] == ["negative"]
    assert spec["rejected_generated_cases"] == [
        {"id": "normal", "reason": expected_reason}
    ]
    assert history[0]["rejected_test_count"] == 1


@pytest.mark.parametrize("stdin_text", ["", " \n"])
def test_empty_or_whitespace_stdin_is_preserved_for_independent_verification(stdin_text):
    generated = {
        "spec_summary": "Print READY without reading stdin.",
        "ambiguities": [],
        "inferred_contract": {
            "input_format": "No standard input is read.",
            "output_format": "Print READY followed by a newline.",
            "constraints": [],
            "semantics": ["Print a fixed readiness message."],
            "formatting": ["One line containing READY."],
        },
        "test_inputs": [{
            "id": "no-input",
            "input": stdin_text,
            "expected_output": "READY\n",
            "tags": ["normal"],
        }],
    }

    calls = []

    def request(messages, schema, stage):
        calls.append(stage)
        if stage == "assignment_test_generation_single_pass":
            input_schema = schema["properties"]["test_inputs"]["items"]["properties"]["input"]
            assert "minLength" not in input_schema
            return generated
        assert stage == "assignment_expected_output_verification_single_pass"
        payload = json.loads(messages[1]["content"])
        assert payload["test_inputs_to_solve"] == [{"id": "no-input", "input": stdin_text}]
        return {"verifications": [{
            "id": "no-input",
            "input": stdin_text,
            "input_is_valid": True,
            "expected_output": "READY\n",
            "reason": "The program does not read stdin.",
        }]}

    spec, _ = generate_assignment_test_spec(
        problem="Print READY and a newline without reading any standard input.",
        problem_contract={},
        request_json=request,
        max_cases=3,
    )

    assert calls == [
        "assignment_test_generation_single_pass",
        "assignment_expected_output_verification_single_pass",
    ]
    assert spec["test_inputs"][0]["input"] == stdin_text
    assert spec["test_inputs"][0]["expected_output"] == "READY\n"


def test_explicit_input_contract_still_rejects_missing_required_stdin():
    from grading_agent.testing.test_generation import _validate_input_against_contract

    assert _validate_input_against_contract("", {}) is None
    assert "exactly 1" in _validate_input_against_contract("", {"input_token_count": 1})
    assert "missing required field" in _validate_input_against_contract(
        "",
        {"input_schema": {"fields": [{"name": "value", "type": "integer"}]}},
    )


def test_all_invalid_candidates_get_one_bounded_refill_then_independent_verification():
    initial = sample_response()
    initial["test_inputs"] = [
        {"id": "normal_case", "input": "", "expected_output": "5\n", "tags": ["normal"]},
        {"id": "boundary_case", "input": " \n", "expected_output": "3\n", "tags": ["edge"]},
    ]
    calls = []
    verification_calls = 0

    def request(messages, schema, stage):
        nonlocal verification_calls
        calls.append(stage)
        payload = json.loads(messages[1]["content"])
        if stage == "assignment_test_generation_single_pass":
            return initial
        if stage == "assignment_expected_output_verification_single_pass":
            verification_calls += 1
            if verification_calls == 1:
                return {"verifications": [
                    {
                        "id": case["id"],
                        "input": case["input"],
                        "input_is_valid": False,
                        "expected_output": "",
                        "reason": "This assignment requires two integer tokens on stdin.",
                    }
                    for case in payload["test_inputs_to_solve"]
                ]}
            return verifier_response(payload["test_inputs_to_solve"])
        if stage == "assignment_test_generation_repair":
            assert payload["requested_replacement_count"] == 2
            assert payload["previously_used_inputs"] == ["", " \n"]
            assert schema["properties"]["test_inputs"]["maxItems"] == 2
            assert "expected_output" not in json.dumps(payload["previously_rejected_candidates"])
            return {
                "spec_summary": "Read two integers and print their sum.",
                "ambiguities": [],
                "inferred_contract": sample_response()["inferred_contract"],
                "test_inputs": [
                    {"id": "normal", "input": "8 9\n", "expected_output": "17\n", "tags": ["normal"]},
                    {"id": "negative", "input": "-1 1\n", "expected_output": "0\n", "tags": ["edge"]},
                ],
            }
        raise AssertionError(f"Unexpected request stage: {stage}")

    spec, history = generate_assignment_test_spec(
        problem="Read two integers and print their sum.",
        problem_contract={},
        request_json=request,
        max_cases=2,
    )

    assert calls == [
        "assignment_test_generation_single_pass",
        "assignment_expected_output_verification_single_pass",
        "assignment_test_generation_repair",
        "assignment_expected_output_verification_single_pass",
    ]
    assert [case["input"] for case in spec["test_inputs"]] == ["8 9\n", "-1 1\n"]
    assert [case["id"] for case in spec["test_inputs"]] == ["refill-1-normal", "refill-2-negative"]
    assert spec["generation_response_count"] == 2
    assert spec["verification_response_count"] == 2
    assert len(history) == 2


def test_refill_failure_is_reported_with_diagnostics():
    calls = []

    def request(messages, schema, stage):
        calls.append(stage)
        if stage == "assignment_test_generation_single_pass":
            return sample_response()
        if stage == "assignment_expected_output_verification_single_pass":
            payload = json.loads(messages[1]["content"])
            return {"verifications": [
                {
                    "id": case["id"],
                    "input": case["input"],
                    "input_is_valid": False,
                    "expected_output": "",
                    "reason": "Rejected by the assignment verifier.",
                }
                for case in payload["test_inputs_to_solve"]
            ]}
        if stage == "assignment_test_generation_repair":
            raise RuntimeError("temporary model endpoint failure")
        raise AssertionError(f"Unexpected stage: {stage}")

    with pytest.raises(TestGenerationError, match="temporary model endpoint failure"):
        generate_assignment_test_spec(
            problem="Read two integers and print their sum.",
            problem_contract={},
            request_json=request,
            max_cases=2,
        )

    assert calls == [
        "assignment_test_generation_single_pass",
        "assignment_expected_output_verification_single_pass",
        "assignment_test_generation_repair",
    ]


def test_edge_case_runner_recognizes_boundary_in_test_id(tmp_path):
    from grading_agent.testing.test_generation import prepare_assignment_test_suite

    rules = {
        "problem": "Read two integers and print their sum.",
        "problem_contract": {},
        "project_root": str(tmp_path),
        "metrics": [{
            "name": "logical",
            "analyzer": "edge_case_runner",
            "enabled": True,
            "params": {},
        }],
        "test_generation": {"enabled": True, "seed": 42, "max_cases": 1},
    }

    def request(messages, schema, stage):
        if stage == "assignment_test_generation_single_pass":
            response = sample_response()
            response["test_inputs"] = [{
                "id": "boundary_case",
                "input": "9 1\n",
                "expected_output": "10\n",
                "tags": ["normal"],
            }]
            return response
        payload = json.loads(messages[1]["content"])
        return verifier_response(payload["test_inputs_to_solve"])

    result = prepare_assignment_test_suite(
        rules,
        sandbox=None,
        request_json=request,
        project_root=tmp_path,
        model_id="qwen-test",
    )

    assert result["generated_test_count"] == 1
    assert result["edge_test_count"] == 1
    edge_cases = json.loads(Path(result["edge_testset_path"]).read_text(encoding="utf-8"))
    assert edge_cases[0]["id"] == "generated-boundary_case"


def test_independent_verifier_rejects_wrong_output_and_ambiguous_input():

def test_missing_verifier_result_fails_closed():
    def request(messages, schema, stage):
        if stage == "assignment_test_generation_single_pass":
            return sample_response()
        return {"verifications": [
            {"id": "basic", "input": "2 3\n", "input_is_valid": True, "expected_output": "5\n", "reason": ""}
        ]}

    with pytest.raises(TestGenerationError, match="omitted test negative"):
        generate_assignment_test_spec(
            problem="Read two integers and print their sum.",
            problem_contract={},
            request_json=request,
            max_cases=2,
        )


def test_generation_fails_closed_on_ambiguous_contract():
    response = sample_response()
    response["ambiguities"] = ["The assignment does not define overflow behavior."]

    def request(messages, schema, stage):
        return response

    with pytest.raises(TestGenerationError, match="Unresolved assignment ambiguities"):
        generate_assignment_test_spec(
            problem="Read two integers from input and print their arithmetic sum.",
            problem_contract={},
            request_json=request,
        )


def test_explicit_input_schema_filters_out_of_range_and_malformed_cases():
    from grading_agent.testing.test_generation import _validate_input_against_contract

    contract = {
        "input_schema": {
            "fields": [
                {"name": "count", "type": "integer", "min": 1, "max": 10},
                {"name": "mode", "type": "string", "enum": ["fast", "slow"]},
            ]
        }
    }
    assert _validate_input_against_contract("5 fast\n", contract) is None
    assert "minimum" in _validate_input_against_contract("0 fast\n", contract)
    assert "integer" in _validate_input_against_contract("five fast\n", contract)
    assert "allowed values" in _validate_input_against_contract("5 invalid\n", contract)
    assert "unexpected token" in _validate_input_against_contract("5 fast extra\n", contract)


def test_generation_refuses_a_title_without_an_assignment_contract():
    with pytest.raises(TestGenerationError, match="only a title"):
        generate_assignment_test_spec(
            problem="grading",
            problem_contract={},
            request_json=lambda *_: sample_response(),
        )


def test_materialize_merges_instructor_hints_and_verified_pairs_without_running_c_code():
    spec = sample_response()
    merged = materialize_test_cases(
        spec,
        instructor_cases=[{"id": "teacher-1", "input": "1 2\n", "expected": "3\n"}],
    )
    assert len(merged) == 3
    assert merged[0]["origin"] == "instructor"
    assert merged[1]["origin"] == "llm_generated_independently_verified"
    assert merged[1]["expected"] == "5\n"
    assert merged[2]["expected"] == "3\n"


def test_generated_pair_cannot_disagree_with_an_instructor_example():
    spec = {
        "test_inputs": [
            {"id": "case", "input": "1 2\n", "expected_output": "999\n", "tags": []}
        ]
    }
    with pytest.raises(TestGenerationError, match="disagrees with an instructor example"):
        materialize_test_cases(
            spec,
            instructor_cases=[{"id": "teacher", "input": "1 2\n", "expected": "3\n"}],
        )


def test_conflicting_instructor_examples_are_rejected():
    with pytest.raises(TestGenerationError, match="Instructor examples disagree"):
        materialize_test_cases(
            {"test_inputs": []},
            instructor_cases=[
                {"id": "a", "input": "1 2\n", "expected": "3\n"},
                {"id": "b", "input": "1 2\n", "expected": "4\n"},
            ],
        )


def test_assignment_fingerprint_and_frozen_suite_are_stable(tmp_path):
    fingerprint = assignment_fingerprint(
        problem="Add two numbers",
        problem_contract={"type": "int"},
        model_id="Qwen-test",
    )
    cases = [{"id": "t1", "input": "1 2\n", "expected": "3\n"}]
    spec = sample_response()
    spec["verification_summary"] = {"method": "independent_second_llm_call", "verified": 2}
    path = write_frozen_suite(
        tmp_path,
        fingerprint=fingerprint,
        cases=cases,
        generated_spec=spec,
        model_id="Qwen-test",
        oracle_kind="two-pass-llm-verified-expected-outputs",
    )
    frozen = json.loads(path.read_text(encoding="utf-8"))
    assert frozen["schema_version"] == 3
    assert frozen["expected_output_verification"]["verified"] == 2
    assert "reference_solution_c" not in frozen
    assert load_frozen_suite(path, fingerprint) == cases
    with pytest.raises(TestGenerationError, match="does not match"):
        load_frozen_suite(path, "different-fingerprint")
    with pytest.raises(TestGenerationError, match="differs"):
        write_frozen_suite(
            tmp_path,
            fingerprint=fingerprint,
            cases=[],
            generated_spec=spec,
            model_id="Qwen-test",
        )


def test_prepare_suite_uses_two_llm_output_calls_and_never_compiles_a_reference(tmp_path):
    from grading_agent.testing.test_generation import prepare_assignment_test_suite

    class NoReferenceSandbox:
        def write_file(self, *args, **kwargs):
            raise AssertionError("test-suite generation must not write a generated C reference file")

        def run(self, *args, **kwargs):
            raise AssertionError("test-suite generation must not compile or execute a reference program")

    source = '#include <stdio.h>\nint main(void) { int a, b; scanf("%d%d", &a, &b); printf("%d\\n", a + b); }\n'
    analysis = {
        "implementation_summary": "Reads two integers and prints their sum.",
        "input_parsing_and_validation": ["Does not check scanf result."],
        "control_flow_and_data_types": ["Adds two signed integers."],
        "likely_weak_points": ["Potential integer overflow."],
        "recommended_test_dimensions": ["Negative values and numeric limits."],
    }
    calls = []

    def request(messages, schema, stage):
        calls.append(stage)
        payload = json.loads(messages[1]["content"])
        if stage == "assignment_submission_understanding_single_pass":
            assert payload["submission_source"] == source
            return analysis
        if stage == "assignment_test_generation_single_pass":
            assert payload["submitted_implementation_analysis"] == analysis
            return sample_response()
        assert stage == "assignment_expected_output_verification_single_pass"
        assert "submission_source" not in payload
        assert "submitted_implementation_analysis" not in payload
        return verifier_response(payload["test_inputs_to_solve"])

    rules = {
        "problem": "Read two integers and print their sum.",
        "problem_contract": {"sum_type": "int64"},
        "project_root": str(tmp_path),
        "time_limit_s": 2,
        "metrics": [{
            "name": "correctness", "analyzer": "io_tests", "enabled": True,
            "params": {},
        }],
        "test_generation": {"enabled": True, "seed": 42, "max_cases": 2},
    }
    result = prepare_assignment_test_suite(
        rules,
        sandbox=NoReferenceSandbox(),
        request_json=request,
        project_root=tmp_path,
        model_id="qwen-test",
        submission_source=source,
    )

    assert calls == [
        "assignment_submission_understanding_single_pass",
        "assignment_test_generation_single_pass",
        "assignment_expected_output_verification_single_pass",
    ]
    assert result["status"] == "completed"
    assert result["generated_test_count"] == 2
    assert result["expected_output_verification"]["verified"] == 2
    frozen = json.loads(Path(result["suite_path"]).read_text(encoding="utf-8"))
    assert frozen["schema_version"] == 3
    assert "reference_solution_c" not in frozen
    all_cases = json.loads(Path(result["all_testset_path"]).read_text(encoding="utf-8"))
    assert [case["expected"] for case in all_cases] == ["5\n", "3\n"]


def test_prepare_suite_uses_instructor_cases_as_hints_for_all_metrics(tmp_path):
    from grading_agent.testing.test_generation import prepare_assignment_test_suite

    class NoReferenceSandbox:
        def write_file(self, *args, **kwargs):
            raise AssertionError("no C reference should be written")

        def run(self, *args, **kwargs):
            raise AssertionError("no C reference should be compiled or executed")

    teacher_cases = [
        {"id": "teacher-a", "input": "1 2\n", "expected": "3\n"},
        {"id": "teacher-b", "input": "3 4\n", "expected": "7\n"},
    ]
    edge_cases = [
        {"id": "edge-a", "input": "1 2\n", "expected": "3\n"},
        {"id": "edge-b", "input": "-3 8\n", "expected": "5\n"},
    ]
    memory_cases = [{"id": "memory-a", "input": "3 4\n", "expected": "7\n"}]
    for filename, cases in (
        ("correctness.json", teacher_cases),
        ("edge.json", edge_cases),
        ("memory.json", memory_cases),
    ):
        (tmp_path / filename).write_text(json.dumps(cases), encoding="utf-8")

    rules = {
        "problem": "Read two integers and print their sum.",
        "problem_contract": {"sum_type": "int32"},
        "project_root": str(tmp_path),
        "metrics": [
            {"name": "correctness", "analyzer": "io_tests", "enabled": True,
             "params": {"testset_path": "correctness.json", "resolved_testset_path": str(tmp_path / "correctness.json")}},
            {"name": "logical", "analyzer": "edge_case_runner", "enabled": True,
             "params": {"testset_path": "edge.json", "resolved_testset_path": str(tmp_path / "edge.json")}},
            {"name": "memory_safety", "analyzer": "memory_safety", "enabled": True,
             "params": {"testset_path": "memory.json", "resolved_testset_path": str(tmp_path / "memory.json")}},
        ],
        "test_generation": {
            "enabled": True, "seed": 42, "max_cases": 2,
            # A legacy key must not cause the generator to load or compile C.
            "reference_solution_path": "this-file-is-intentionally-absent.c",
        },
    }
    calls = []

    def request(messages, schema, stage):
        calls.append(stage)
        payload = json.loads(messages[1]["content"])
        if stage == "assignment_test_generation_single_pass":
            assert len(payload["instructor_examples_as_hints"]) == 3
            assert payload["instructor_examples_as_hints"][0]["expected_output"] == "3\n"
            return {
                "spec_summary": "Read two integers and print their sum.",
                "ambiguities": [],
                "inferred_contract": sample_response()["inferred_contract"],
                "test_inputs": [
                    {"id": "generated-edge", "input": "10 1\n", "expected_output": "11\n", "tags": ["edge", "boundary"]},
                    {"id": "generated-normal", "input": "20 2\n", "expected_output": "22\n", "tags": ["normal"]},
                ],
            }
        assert stage == "assignment_expected_output_verification_single_pass"
        assert len(payload["instructor_examples_as_format_and_behavior_hints"]) == 3
        return verifier_response(payload["test_inputs_to_solve"])

    result = prepare_assignment_test_suite(
        rules,
        sandbox=NoReferenceSandbox(),
        request_json=request,
        project_root=tmp_path,
        model_id="qwen-test",
    )

    assert calls == [
        "assignment_test_generation_single_pass",
        "assignment_expected_output_verification_single_pass",
    ]
    correctness_path = Path(result["metric_testset_paths"]["correctness"])
    correctness_cases = json.loads(correctness_path.read_text(encoding="utf-8"))
    assert result["instructor_test_count"] == 3  # teacher, edge, and memory inputs are deduplicated
    assert result["generated_test_count"] == 2
    assert result["total_test_count"] == 5
    assert len(correctness_cases) == 5
    assert sum(case["origin"] == "instructor" for case in correctness_cases) == 3
    assert sum(case["origin"] == "llm_generated_independently_verified" for case in correctness_cases) == 2
    assert rules["metrics"][0]["params"]["resolved_testset_path"] == str(correctness_path)


def test_prepare_suite_fails_if_required_logical_cases_are_not_generated(tmp_path):
    from grading_agent.testing.test_generation import prepare_assignment_test_suite

    class NoReferenceSandbox:
        def write_file(self, *args, **kwargs):
            raise AssertionError("must not write reference code")

        def run(self, *args, **kwargs):
            raise AssertionError("must not run reference code")

    rules = {
        "problem": "Read two integers and print their sum.",
        "problem_contract": {},
        "project_root": str(tmp_path),
        "metrics": [{"name": "logical", "analyzer": "edge_case_runner", "enabled": True, "params": {}}],
        "test_generation": {"enabled": True, "seed": 42, "max_cases": 2},
    }

    def request(messages, schema, stage):
        if stage == "assignment_test_generation_single_pass":
            response = sample_response()
            response["test_inputs"] = [
                {"id": "normal-a", "input": "1 2\n", "expected_output": "3\n", "tags": ["normal"]},
                {"id": "normal-b", "input": "3 4\n", "expected_output": "7\n", "tags": ["normal"]},
            ]
            return response
        payload = json.loads(messages[1]["content"])
        return verifier_response(payload["test_inputs_to_solve"])

    with pytest.raises(TestGenerationError, match="no identifiable edge/boundary"):
        prepare_assignment_test_suite(
            rules,
            sandbox=NoReferenceSandbox(),
            request_json=request,
            project_root=tmp_path,
            model_id="qwen-test",
        )


def test_fingerprint_changes_when_generation_configuration_changes():
    base = assignment_fingerprint(
        problem="Sort integers.", problem_contract={"max_n": 100},
        model_id="Qwen-test", generation_seed=42, rsi_passes=3, max_cases=60, temperature=0.0,
    )
    changed = assignment_fingerprint(
        problem="Sort integers.", problem_contract={"max_n": 100},
        model_id="Qwen-test", generation_seed=42, rsi_passes=2, max_cases=60, temperature=0.0,
    )
    assert base != changed


def test_strict_mode_refuses_to_regenerate_a_missing_frozen_suite(tmp_path):
    from grading_agent.testing.test_generation import prepare_assignment_test_suite

    rules = {
        "problem": "Read an integer and print it.",
        "problem_contract": {},
        "project_root": str(tmp_path),
        "metrics": [{"name": "correctness", "analyzer": "io_tests", "enabled": True, "params": {}}],
        "test_generation": {
            "enabled": True, "require_frozen_suite": True,
            "seed": 42, "temperature": 0.0,
        },
    }
    with pytest.raises(TestGenerationError, match="requires the pre-generated assignment suite"):
        prepare_assignment_test_suite(
            rules,
            sandbox=object(),
            request_json=lambda *_: pytest.fail("must not regenerate"),
            project_root=tmp_path,
            model_id="qwen-test",
            generation_seed=42,
            temperature=0.0,
        )
