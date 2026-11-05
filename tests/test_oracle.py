"""Execution-oracle tests with a fake sandbox that runs Python programs locally."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grading_agent.testing.oracle import ExecutionOracle, build_llm_oracle  # noqa: E402
from grading_agent.testing.test_generation import (  # noqa: E402
    _build_expected_output_oracle,
    generate_assignment_test_spec,
    oracle_caveat,
)

SUM_PROGRAM = "import sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)\n"
WRONG_PROGRAM = "import sys\na, b = map(int, sys.stdin.read().split())\nprint(a - b)\n"


class LocalPythonSandbox:
    """Implements the sandbox calls the oracle uses, executing Python locally."""

    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp())

    def write_file(self, name: str, content: str) -> None:
        (self.dir / name).write_text(content, encoding="utf-8")

    def _exec(self, command, stdin=None, timeout_s=10):
        command = [sys.executable if part == "python3" else part for part in command]
        try:
            done = subprocess.run(command, input=stdin, capture_output=True, text=True, cwd=self.dir, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"exit_code": 124, "output": "", "stdout": "", "stderr": "", "timed_out": True}
        return {"exit_code": done.returncode, "output": done.stdout + done.stderr, "stdout": done.stdout,
                "stderr": done.stderr, "timed_out": False, "output_may_be_truncated": False}

    def run(self, command, timeout_s=None, stdin=None, **_):
        return self._exec(command, stdin, timeout_s or 10)

    def run_with_streams(self, command, timeout_s=None, stdin=None):
        return self._exec(command, stdin, timeout_s or 10)


def generation_response(cases):
    return {
        "spec_summary": "Read two integers and print their sum.",
        "ambiguities": [],
        "inferred_contract": {
            "input_format": "Two signed integers separated by whitespace.",
            "output_format": "Their sum followed by a newline.",
            "constraints": [], "semantics": ["Print a + b."], "formatting": ["One line."],
        },
        "test_inputs": cases,
    }


INSTRUCTOR = [{"id": "teacher", "input": "2 3\n", "expected": "5\n"}]


def test_trusted_oracle_output_replaces_wrong_generator_answer_and_skips_model_check():
    oracle = ExecutionOracle(LocalPythonSandbox(), SUM_PROGRAM, language="python", kind="instructor_reference")
    assert oracle.prepare() is None
    stages = []

    def request(messages, schema, stage):
        stages.append(stage)
        assert stage == "assignment_test_generation_single_pass", "no model verification with a trusted oracle"
        return generation_response([
            {"id": "big", "input": "40 2\n", "expected_output": "7\n"},          # generator is wrong
            {"id": "bad", "input": "not numbers\n", "expected_output": "0\n"},  # reference rejects it
        ])

    spec, _ = generate_assignment_test_spec(
        problem="Read two integers and print their sum.", problem_contract={}, request_json=request,
        max_cases=2, instructor_examples=INSTRUCTOR, oracle=oracle, oracle_trusted=True,
    )
    assert [(c["id"], c["expected_output"]) for c in spec["test_inputs"]] == [("big", "42\n")]
    assert spec["test_inputs"][0]["expected_output_source"] == "instructor_reference"
    assert any(r["id"] == "bad" and "reference program rejected" in r["reason"] for r in spec["rejected_generated_cases"])


def test_untrusted_oracle_needs_agreement_with_independent_check():
    oracle = ExecutionOracle(LocalPythonSandbox(), SUM_PROGRAM, language="python", kind="llm_reference")
    assert oracle.prepare() is None

    def request(messages, schema, stage):
        if stage == "assignment_test_generation_single_pass":
            return generation_response([
                {"id": "agree", "input": "1 1\n", "expected_output": "2\n"},
                {"id": "disagree", "input": "5 5\n", "expected_output": "10\n"},
            ])
        payload = json.loads(messages[1]["content"])
        answers = {"agree": "2\n", "disagree": "11\n"}  # the independent check is wrong on one
        return {"verifications": [
            {"id": item["id"], "input": item["input"], "input_is_valid": True,
             "expected_output": answers[item["id"]], "reason": "solved"}
            for item in payload["test_inputs_to_solve"]
        ]}

    spec, _ = generate_assignment_test_spec(
        problem="Read two integers and print their sum.", problem_contract={}, request_json=request,
        max_cases=2, oracle=oracle, oracle_trusted=False,
    )
    assert [c["id"] for c in spec["test_inputs"]] == ["agree"]
    assert any(r["id"] == "disagree" and "reference program output disagreed" in r["reason"]
               for r in spec["rejected_generated_cases"])


def test_llm_reference_is_trusted_only_after_matching_instructor_cases():
    programs = iter([WRONG_PROGRAM, SUM_PROGRAM])
    feedback_seen = []

    def request(messages, schema, stage):
        assert stage == "assignment_reference_program"
        payload = json.loads(messages[1]["content"])
        if "previous_program_problems" in payload:
            feedback_seen.append(payload["previous_program_problems"])
        return {"approach": "add", "program": next(programs)}

    oracle, record = build_llm_oracle(
        sandbox=LocalPythonSandbox(), problem="Read two integers and print their sum.", problem_contract={},
        instructor_cases=INSTRUCTOR, hint_examples=[], request_json=request,
    )
    assert oracle is not None and record["trusted"] is True and record["attempts"] == 2
    assert "wrong answers on instructor test cases" in feedback_seen[0]
    assert oracle.run("7 8\n")["stdout"] == "15\n"


def test_llm_reference_that_never_matches_is_not_used():
    oracle, record = build_llm_oracle(
        sandbox=LocalPythonSandbox(), problem="Read two integers and print their sum.", problem_contract={},
        instructor_cases=INSTRUCTOR, hint_examples=[],
        request_json=lambda messages, schema, stage: {"approach": "subtract", "program": WRONG_PROGRAM},
    )
    assert oracle is None and record["trusted"] is False


def test_without_sandbox_outputs_are_flagged_as_unexecuted():
    oracle, record = _build_expected_output_oracle(
        sandbox=None, rules={}, reference_source="", reference_path="", instructor_cases=[], request_json=None,
    )
    assert oracle is None and record["kind"] == "none"
    assert "NOT confirmed by executing" in oracle_caveat(record)


def test_instructor_reference_build_failure_is_reported():
    from grading_agent.testing.test_generation import TestGenerationError

    with pytest.raises(TestGenerationError, match="did not build"):
        _build_expected_output_oracle(
            sandbox=LocalPythonSandbox(), rules={}, reference_source="this is not python(\n",
            reference_path="reference_solution.py", instructor_cases=[], request_json=None,
        )
