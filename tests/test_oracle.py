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


GRID_PROBLEM = (
    "Read R and C (1 <= R,C <= 30), followed by an R by C grid whose cells are 0 (open) or 1 (blocked). "
    "Count right/down paths from the top-left to the bottom-right avoiding blocked cells; print the count."
)
GRID_PROGRAM = (
    "import sys\nit = iter(sys.stdin.read().split())\nr, c = int(next(it)), int(next(it))\n"
    "g = [[int(next(it)) for _ in range(c)] for _ in range(r)]\n"
    "dp = [[0] * c for _ in range(r)]\n"
    "for i in range(r):\n    for j in range(c):\n        if g[i][j]: continue\n"
    "        dp[i][j] = 1 if i == j == 0 else (dp[i-1][j] if i else 0) + (dp[i][j-1] if j else 0)\n"
    "print(dp[-1][-1])\n"
)


def test_oracle_backed_generation_omits_outputs_rejects_miscounted_grids_and_keeps_ambiguous_refill():
    # Regression for the grid-paths failure: a miscounted grid reached the
    # reference program (StopIteration), and a refill reporting an irrelevant
    # ambiguity was discarded although the oracle decides every output.
    oracle = ExecutionOracle(LocalPythonSandbox(), GRID_PROGRAM, language="python", kind="instructor_reference")
    assert oracle.prepare() is None
    schemas = {}

    def request(messages, schema, stage):
        schemas[stage] = schema
        if stage == "assignment_test_generation_single_pass":
            return generation_response([
                {"id": "short-grid", "input": "2 3\n0 0 0\n0 0\n"},  # 5 of 6 cells
                {"id": "half-grid", "input": "2 2\n0 0\n"},          # 2 of 4 cells
            ])
        assert stage == "assignment_test_generation_repair"
        response = generation_response([{"id": "wall", "input": "2 2\n0 1\n1 0\n"}])
        response["ambiguities"] = ["Is the grid 0-indexed or 1-indexed?"]
        return response

    spec, _ = generate_assignment_test_spec(
        problem=GRID_PROBLEM, problem_contract={}, request_json=request, max_cases=2,
        # No instructor grids: no structural variants, so the refill path runs.
        instructor_examples=[{"id": "t", "input": "2 2\n0 0\n0 0\n", "expected": "2\n"}][:0],
        oracle=oracle, oracle_trusted=True,
    )
    item_schema = schemas["assignment_test_generation_single_pass"]["properties"]["test_inputs"]["items"]
    assert "expected_output" not in item_schema["properties"] and item_schema["required"] == ["id", "input"]
    reasons = {r["id"]: r["reason"] for r in spec["rejected_generated_cases"]}
    assert "2 x 3 grid (6 cells) but supplies 5" in reasons["short-grid"]
    assert ("refill-1-wall", "0\n") in [(c["id"], c["expected_output"]) for c in spec["test_inputs"]]
    assert "Is the grid 0-indexed or 1-indexed?" in spec["ambiguities"]


def test_large_case_targets_are_generated_in_bounded_batches_within_the_reply_limit():
    # Regression: one reply had to hold the contract plus every requested grid
    # case and was cut off at 4096 tokens (finish_reason=length) twice.
    oracle = ExecutionOracle(LocalPythonSandbox(), GRID_PROGRAM, language="python", kind="instructor_reference")
    assert oracle.prepare() is None
    serial = iter(range(1, 1000))
    stages = []

    def grid(n):  # a distinct 6 x 6 grid per call, about 80 characters
        cells = [[(n >> (i * 6 + j - 1)) & 1 if 0 < i * 6 + j < 35 else 0 for j in range(6)] for i in range(6)]
        return "6 6\n" + "\n".join(" ".join(str(v) for v in row) for row in cells) + "\n"

    def request(messages, schema, stage):
        stages.append(stage)
        count = schema["properties"]["test_inputs"]["maxItems"]
        cases = [{"id": f"g{n}", "input": grid(n), "tags": ["grid"]} for n in (next(serial) for _ in range(count))]
        response = generation_response(cases) if "inferred_contract" in schema["properties"] else {"test_inputs": cases}
        if stage == "assignment_test_generation_batch" and stages.count(stage) == 2:
            raise RuntimeError("Malformed final assistant JSON; finish_reason=length")
        if len(json.dumps(response)) > 4096:  # ~1 token per character for digit grids
            raise RuntimeError("Malformed final assistant JSON; finish_reason=length")
        return response

    spec, _ = generate_assignment_test_spec(
        problem=GRID_PROBLEM, problem_contract={}, request_json=request, max_cases=12,
        oracle=oracle, oracle_trusted=True,
    )
    assert stages[0] == "assignment_test_generation_single_pass"
    assert "assignment_test_generation_single_pass_retry" not in stages
    assert len(spec["test_inputs"]) == 12, "one failed batch keeps earlier cases and the rest still fill the target"
    assert spec["generation_batches"]["failed_batches"] and spec["generation_batches"]["additional_batches"] >= 3


def test_model_that_only_repeats_instructor_cases_still_yields_verified_structural_variants():
    # Regression: every generated case duplicated an instructor case (or was a
    # miscounted 30 x 30 grid), so the suite failed with nothing verified.
    oracle = ExecutionOracle(LocalPythonSandbox(), GRID_PROGRAM, language="python", kind="instructor_reference")
    assert oracle.prepare() is None
    instructor = [
        {"id": "center", "input": "3 3\n0 0 0\n0 1 0\n0 0 0\n", "expected": "2\n"},
        {"id": "open", "input": "2 2\n0 0\n0 0\n", "expected": "2\n"},
        {"id": "wall", "input": "2 2\n0 1\n0 0\n", "expected": "1\n"},
        {"id": "one", "input": "1 1\n0\n", "expected": "1\n"},
        {"id": "start", "input": "1 1\n1\n", "expected": "0\n"},
    ]
    seen_payloads = []

    def request(messages, schema, stage):
        seen_payloads.append(json.loads(messages[1]["content"]))
        copies = [{"id": f"copy-{c['id']}", "input": c["input"]} for c in instructor]
        count = schema["properties"]["test_inputs"]["maxItems"]
        cases = copies[:count]
        return generation_response(cases) if "inferred_contract" in schema["properties"] else {"test_inputs": cases}

    spec, _ = generate_assignment_test_spec(
        problem=GRID_PROBLEM, problem_contract={}, request_json=request, max_cases=8,
        instructor_examples=instructor, oracle=oracle, oracle_trusted=True,
    )
    # The model is told about all five instructor inputs, not only the three hints.
    assert len(seen_payloads[0]["instructor_inputs_already_covered"]) == 5
    # A batch that only repeats existing inputs ends model generation early.
    assert spec["generation_batches"]["additional_batches"] == 1
    assert all(r["reason"].startswith("duplicate") for r in spec["rejected_generated_cases"])
    cases = spec["test_inputs"]
    assert len(cases) == 8 and spec["structural_variants_added"] == 8
    assert all("structural-variant" in c["tags"] for c in cases)
    instructor_inputs = {c["input"] for c in instructor}
    assert not instructor_inputs & {c["input"] for c in cases}
    assert any(" 1" in c["input"].split("\n", 1)[1] or "1 " in c["input"].split("\n", 1)[1] for c in cases), \
        "variants must include blocked cells, not only open grids"
    # Expected outputs come from executing the reference program.
    assert all(c["expected_output_source"] == "instructor_reference" for c in cases)


def test_variation_focus_is_an_explicit_instruction_and_steers_structural_variants():
    # Regression: the focus sat in the payload as a key no instruction named,
    # and deterministic variants ignored it, so the field had no visible effect.
    oracle = ExecutionOracle(LocalPythonSandbox(), GRID_PROGRAM, language="python", kind="instructor_reference")
    assert oracle.prepare() is None
    instructor = [{"id": "center", "input": "3 3\n0 0 0\n0 1 0\n0 0 0\n", "expected": "2\n"}]
    instructions = []

    def request(messages, schema, stage):
        instructions.append(json.loads(messages[1]["content"])["instruction"])
        cases = [{"id": "copy", "input": instructor[0]["input"]}]
        return generation_response(cases) if "inferred_contract" in schema["properties"] else {"test_inputs": cases}

    focus = "dense obstacles and large grids"
    spec, _ = generate_assignment_test_spec(
        problem=GRID_PROBLEM, problem_contract={}, request_json=request, max_cases=4,
        instructor_examples=instructor, oracle=oracle, oracle_trusted=True, additional_focus=focus,
    )
    assert instructions and all(f'coverage focus: "{focus}"' in text for text in instructions)
    cases = spec["test_inputs"]
    assert len(cases) == 4 and all("focus" in c["tags"] for c in cases)
    sizes = [tuple(map(int, c["input"].split()[:2])) for c in cases]
    assert sizes[0] == (6, 6), "large-grid focus puts the largest hand-writable grid first"
    blocked = [c["input"].split("\n", 1)[1].split().count("1") for c in cases if "mixed" in c["tags"]]
    assert blocked and min(blocked) >= 5, "dense-obstacle focus raises the blocked-cell density"
