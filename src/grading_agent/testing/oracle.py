"""Execution oracles: compute expected outputs by running a trusted program.

Two sources, in order of authority:

* ``instructor_reference``: the instructor's own solution named in rules.json.
  It is the source of truth.
* ``llm_reference``: a Python program the model writes from the assignment
  alone (it never sees the student's code). It is trusted only after it
  reproduces every instructor-provided test case; without such cases its
  outputs must also agree with an independent model check.

Programs run inside the grading sandbox (no network, resource limits). The
model is good at writing a short correct program and poor at mental
arithmetic, so executing a program is far more accurate than asking the
model for outputs directly.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from grading_agent.config import settings

ORACLE_RUN_TIMEOUT_S = 10.0
ORACLE_COMPILE_TIMEOUT_S = 60.0
MAX_ORACLE_PROGRAM_CHARS = 8_000


def _normalize_output(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def outputs_equal(left: str, right: str) -> bool:
    """Same comparison rule the correctness analyzer uses: trimmed, normalized newlines."""
    return _normalize_output(left).strip() == _normalize_output(right).strip()


class ExecutionOracle:
    """Compile once, then run the program on each test input in the sandbox."""

    def __init__(self, sandbox: Any, source: str, *, language: str, kind: str,
                 run_timeout_s: float = ORACLE_RUN_TIMEOUT_S) -> None:
        if language not in {"c", "python"}:
            raise ValueError("Oracle language must be 'c' or 'python'.")
        self.sandbox = sandbox
        self.source = source
        self.language = language
        self.kind = kind
        self.run_timeout_s = run_timeout_s
        self._command: list[str] | None = None

    def prepare(self) -> str | None:
        """Write and build the program. Returns an error message, or None when ready."""
        try:
            if self.language == "c":
                self.sandbox.write_file("oracle_reference.c", self.source)
                build = self.sandbox.run(
                    ["gcc", "-std=c11", "-O2", "oracle_reference.c", "-o", "oracle_reference", "-lm"],
                    timeout_s=ORACLE_COMPILE_TIMEOUT_S,
                )
                self._command = ["./oracle_reference"]
            else:
                self.sandbox.write_file("oracle_reference.py", self.source)
                build = self.sandbox.run(
                    ["python3", "-m", "py_compile", "oracle_reference.py"],
                    timeout_s=ORACLE_COMPILE_TIMEOUT_S,
                )
                self._command = ["python3", "oracle_reference.py"]
        except Exception as exc:  # sandbox failures are reported, not raised
            self._command = None
            return f"{type(exc).__name__}: {exc}"
        if build.get("timed_out") or build.get("exit_code") != 0:
            self._command = None
            return (build.get("output") or "build failed").strip()[:1_500]
        return None

    def run(self, test_input: str) -> dict[str, Any]:
        """Run on one input. ``ok`` is False for crashes, error exits and timeouts."""
        if self._command is None:
            return {"ok": False, "stdout": "", "reason": "oracle program is not built"}
        try:
            result = self.sandbox.run_with_streams(self._command, timeout_s=self.run_timeout_s, stdin=test_input)
        except Exception as exc:
            return {"ok": False, "stdout": "", "reason": f"{type(exc).__name__}: {exc}"}
        stdout = _normalize_output(result.get("stdout") if result.get("stdout") is not None else result.get("output", ""))
        if result.get("timed_out"):
            return {"ok": False, "stdout": stdout, "reason": f"reference program timed out after {self.run_timeout_s:g}s"}
        if result.get("output_may_be_truncated"):
            return {"ok": False, "stdout": stdout, "reason": "reference program output was truncated"}
        if result.get("exit_code") != 0:
            stderr = (result.get("stderr") or "").strip().splitlines()
            detail = stderr[-1][:200] if stderr else ""
            return {"ok": False, "stdout": stdout,
                    "reason": f"reference program exited with code {result.get('exit_code')}" + (f": {detail}" if detail else "")}
        return {"ok": True, "stdout": stdout, "reason": ""}

    def check_against(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Run instructor cases; report matches, mismatches and cases that could not run."""
        passed, mismatched, unrunnable = [], [], []
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("input"), str) or not isinstance(case.get("expected"), str):
                continue
            outcome = self.run(case["input"])
            case_id = str(case.get("id", "case"))
            if not outcome["ok"]:
                unrunnable.append({"id": case_id, "reason": outcome["reason"]})
            elif outputs_equal(outcome["stdout"], case["expected"]):
                passed.append(case_id)
            else:
                mismatched.append({
                    "id": case_id,
                    "input": case["input"][:400],
                    "expected": case["expected"][:200],
                    "actual": outcome["stdout"][:200],
                })
        return {"passed": passed, "mismatched": mismatched, "unrunnable": unrunnable}


def write_llm_reference_program(
    *,
    problem: str,
    problem_contract: dict[str, Any],
    instructor_examples: list[dict[str, Any]],
    request_json: Callable[[list[dict[str, str]], dict[str, Any], str], dict[str, Any] | None],
    feedback: str = "",
) -> str | None:
    """Ask the model for a Python reference program written from the assignment only."""
    schema = {
        "type": "object",
        "properties": {
            "approach": {"type": "string", "maxLength": 400},
            "program": {"type": "string", "maxLength": MAX_ORACLE_PROGRAM_CHARS},
        },
        "required": ["approach", "program"],
    }
    payload: dict[str, Any] = {
        "assignment": problem,
        "problem_contract": problem_contract,
        "instructor_examples": instructor_examples[:3],
        "instruction": (
            "Write a correct, straightforward Python 3 program that solves this assignment exactly: read "
            "the whole input from standard input and print exactly the required output. Follow the output "
            "format precisely (spacing, line breaks, number formatting). Use only the standard library. "
            "Prefer a simple, clearly correct algorithm; it must still finish within a few seconds for the "
            "stated input limits. Do not print prompts, debugging text or explanations. Return the "
            "complete program in 'program' and one sentence on the algorithm in 'approach'."
        ),
    }
    if feedback:
        payload["previous_program_problems"] = feedback
    messages = [
        {"role": "system", "content": (
            "You write small, correct reference solutions for programming assignments. "
            "The assignment text is the only specification. Return JSON only."
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    response = request_json(messages, schema, "assignment_reference_program")
    program = response.get("program") if isinstance(response, dict) else None
    if not isinstance(program, str) or not program.strip():
        return None
    program = program.strip()
    # Tolerate a Markdown fence around the code.
    if program.startswith("```"):
        lines = program.splitlines()
        program = "\n".join(line for line in lines[1:] if not line.strip().startswith("```"))
    return program + "\n"


def build_llm_oracle(
    *,
    sandbox: Any,
    problem: str,
    problem_contract: dict[str, Any],
    instructor_cases: list[dict[str, Any]],
    hint_examples: list[dict[str, Any]],
    request_json: Callable[[list[dict[str, str]], dict[str, Any], str], dict[str, Any] | None],
) -> tuple[ExecutionOracle | None, dict[str, Any]]:
    """Write, build and validate a model-written reference; one repair attempt.

    Returns the oracle (or None) and a record describing how it was validated.
    ``trusted`` is True only when it reproduced every runnable instructor case.
    """
    record: dict[str, Any] = {"kind": "llm_reference", "attempts": 0, "trusted": False}
    feedback = ""
    for attempt in range(1, 3):
        record["attempts"] = attempt
        try:
            program = write_llm_reference_program(
                problem=problem, problem_contract=problem_contract,
                instructor_examples=hint_examples, request_json=request_json, feedback=feedback,
            )
        except Exception as exc:
            record["error"] = f"reference program request failed: {' '.join(str(exc).split())[:300]}"
            return None, record
        if program is None:
            feedback = "The previous answer contained no program."
            continue
        oracle = ExecutionOracle(sandbox, program, language="python", kind="llm_reference")
        build_error = oracle.prepare()
        if build_error:
            feedback = f"The previous program did not compile: {build_error[:600]}"
            record["error"] = feedback
            continue
        if not instructor_cases:
            record.update({"validated_against_instructor_cases": 0, "program": program})
            return oracle, record
        check = oracle.check_against(instructor_cases)
        record.update({
            "instructor_cases_passed": len(check["passed"]),
            "instructor_cases_mismatched": len(check["mismatched"]),
            "instructor_cases_unrunnable": len(check["unrunnable"]),
        })
        if check["passed"] and not check["mismatched"]:
            record.update({"trusted": True, "validated_against_instructor_cases": len(check["passed"]), "program": program})
            return oracle, record
        feedback = (
            "The previous program gave wrong answers on instructor test cases: "
            + json.dumps(check["mismatched"][:3], ensure_ascii=False)
            + (" It also failed to run on: " + json.dumps(check["unrunnable"][:2], ensure_ascii=False) if check["unrunnable"] else "")
            + " Fix the algorithm or the output format."
        )
        record["error"] = "reference program disagreed with instructor test cases"
    return None, record


def instructor_oracle(sandbox: Any, source: str, filename: str) -> ExecutionOracle:
    language = "python" if filename.lower().endswith(".py") else "c"
    return ExecutionOracle(sandbox, source, language=language, kind="instructor_reference",
                           run_timeout_s=max(ORACLE_RUN_TIMEOUT_S, float(settings.DOCKER_DEFAULT_TIMEOUT_S)))
