"""Single-pass, source-aware test generation with deterministic candidate validation.

The LLM first studies the submitted implementation once, then proposes test
inputs once. If an independent reference program is omitted, a focused
reference-only request may supply it; test inputs are never regenerated.
Invalid or duplicate candidates are discarded; accepted inputs must also
execute successfully against a trusted or independent reference solution in
the restricted sandbox. The resulting suite is frozen by a source-aware
fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

MAX_GENERATED_CASES = 80
MAX_INPUT_CHARS = 20_000
MAX_REFERENCE_SOURCE_CHARS = 100_000


def _project_relative_path(project_root: Path, value: Any, field_name: str) -> Path:
    """Resolve slash-agnostic project-relative paths on Windows and POSIX."""
    if not isinstance(value, str) or not value.strip():
        raise TestGenerationError("invalid_project_path", f"{field_name} must be a non-empty project-relative path.")
    raw = value.strip()
    candidate = Path(raw.replace("\\", "/"))
    windows_candidate = PureWindowsPath(raw)
    root = Path(project_root).resolve()
    if candidate.is_absolute() or windows_candidate.is_absolute() or windows_candidate.drive:
        raise TestGenerationError("invalid_project_path", f"{field_name} must be project-relative.")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise TestGenerationError("invalid_project_path", f"{field_name} must remain under the project root.")
    return resolved


class TestGenerationError(RuntimeError):
    """A deterministic test suite could not be safely produced."""
    __test__ = False

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def assignment_fingerprint(
    *, problem: str, problem_contract: dict[str, Any], model_id: str,
    input_format: Any = None, output_format: Any = None,
    generator_version: str = "1", generation_seed: int = 42,
    rsi_passes: int = 3, max_cases: int = 60, temperature: float = 0.0,
    additional_focus: str = "", submission_source_sha256: str = "",
) -> str:
    """Stable identity for the assignment, selected count, and submitted source."""
    payload = {
        "problem": problem.strip(), "problem_contract": problem_contract,
        "input_format": input_format, "output_format": output_format,
        "model_id": model_id, "generator_version": generator_version,
        "generation_seed": generation_seed, "rsi_passes": rsi_passes,
        "max_cases": max_cases, "temperature": temperature,
    }
    if additional_focus.strip():
        payload["additional_focus"] = additional_focus.strip()
    if submission_source_sha256:
        payload["submission_source_sha256"] = submission_source_sha256
    try:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TestGenerationError("invalid_fingerprint_data", f"Assignment fingerprint data is not valid JSON: {exc}") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_candidate(
    value: Any,
    pass_number: int,
    *,
    allow_ambiguities: bool = False,
    fallback_summary: str = "",
    fallback_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate generated input/output pairs before independent verification."""
    if not isinstance(value, dict):
        raise TestGenerationError("invalid_response", f"Generation response {pass_number} did not return an object.")
    summary = value.get("spec_summary", value.get("summary"))
    ambiguity = value.get("ambiguities", value.get("unresolved_ambiguities", []))
    cases = value.get("test_inputs", value.get("test_cases", value.get("tests")))
    if not isinstance(summary, str) or not summary.strip():
        summary = fallback_summary.strip()
        if not summary:
            raise TestGenerationError("missing_spec", f"Generation response {pass_number} omitted the formal task summary.")
    if not isinstance(ambiguity, list) or not all(isinstance(x, str) for x in ambiguity):
        raise TestGenerationError("invalid_ambiguities", f"Generation response {pass_number} returned malformed ambiguity details.")
    if ambiguity and not allow_ambiguities:
        raise TestGenerationError("ambiguous_assignment", "Unresolved assignment ambiguities: " + "; ".join(ambiguity[:5]))

    inferred = value.get("inferred_contract", value.get("io_contract"))
    fallback_contract = fallback_contract if isinstance(fallback_contract, dict) else {}
    contract_source = "model"
    if not isinstance(inferred, dict):
        inferred = dict(fallback_contract)
        contract_source = "assignment_and_instructor_contract"
        if not inferred:
            raise TestGenerationError("missing_inferred_contract", f"Generation response {pass_number} omitted the assignment-derived I/O contract.")
    else:
        inferred = dict(inferred)
        for field in ("input_format", "output_format"):
            if not isinstance(inferred.get(field), str) or len(inferred[field].strip()) < 8:
                fallback_value = fallback_contract.get(field)
                if not isinstance(fallback_value, str) or len(fallback_value.strip()) < 8:
                    raise TestGenerationError("invalid_inferred_contract", f"Generation response {pass_number} must infer a meaningful {field}.")
                inferred[field] = fallback_value
                contract_source = "assignment_and_instructor_contract"
        for field in ("constraints", "semantics", "formatting"):
            entries = inferred.get(field)
            if not isinstance(entries, list) or not all(isinstance(item, str) and item.strip() for item in entries):
                fallback_entries = fallback_contract.get(field)
                if not isinstance(fallback_entries, list):
                    raise TestGenerationError("invalid_inferred_contract", f"Generation response {pass_number} returned invalid inferred_contract.{field}.")
                inferred[field] = fallback_entries
                contract_source = "assignment_and_instructor_contract"
        if not inferred["semantics"]:
            inferred["semantics"] = fallback_contract.get("semantics", [])
            contract_source = "assignment_and_instructor_contract"
            if not inferred["semantics"]:
                raise TestGenerationError("invalid_inferred_contract", f"Generation response {pass_number} must identify at least one task semantic.")

    if not isinstance(cases, list) or not cases or len(cases) > MAX_GENERATED_CASES:
        raise TestGenerationError("invalid_test_count", f"Generation response {pass_number} must return 1..{MAX_GENERATED_CASES} input/output pairs.")
    cleaned, seen_ids, seen_inputs = [], set(), set()
    for index, item in enumerate(cases, 1):
        if not isinstance(item, dict):
            raise TestGenerationError("invalid_test", f"Generated test {index} must contain input and expected_output strings.")
        case_id = item.get("id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", case_id):
            case_id = f"generated-{index}"
        if case_id in seen_ids:
            raise TestGenerationError("duplicate_test_id", f"Duplicate generated test ID: {case_id}.")
        test_input = item.get("input", item.get("stdin"))
        expected_output = item.get("expected_output", item.get("expected"))
        if not isinstance(test_input, str) or not test_input.strip() or len(test_input) > MAX_INPUT_CHARS:
            raise TestGenerationError("invalid_test_input", f"Generated test {case_id} has invalid input.")
        if not isinstance(expected_output, str) or len(expected_output) > MAX_INPUT_CHARS:
            raise TestGenerationError("invalid_expected_output", f"Generated test {case_id} must include a bounded expected_output string.")
        tags = item.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(x, str) for x in tags):
            tags = []
        tags = sorted(set(tag.strip()[:24] for tag in tags if tag.strip()))[:3] or ["generated"]
        rationale = item.get("rationale", item.get("description", ""))
        if not isinstance(rationale, str):
            rationale = ""
        seen_ids.add(case_id)
        if test_input in seen_inputs:
            continue
        seen_inputs.add(test_input)
        cleaned.append({
            "id": case_id,
            "input": test_input,
            "expected_output": expected_output,
            "tags": tags,
            "description": rationale.strip()[:240],
        })
    if not cleaned:
        raise TestGenerationError("empty_after_deduplication", "All generated inputs were duplicates.")
    return {
        "spec_summary": summary.strip()[:2_000],
        "spec_summary_source": "assignment_statement" if value.get("spec_summary") != summary else "model",
        "inferred_contract_source": contract_source,
        "ambiguities": [item.strip()[:500] for item in ambiguity if item.strip()][:10],
        "inferred_contract": {
            "input_format": inferred["input_format"].strip()[:1_000],
            "output_format": inferred["output_format"].strip()[:1_000],
            "constraints": [item.strip()[:400] for item in inferred["constraints"][:25]],
            "semantics": [item.strip()[:400] for item in inferred["semantics"][:25]],
            "formatting": [item.strip()[:400] for item in inferred["formatting"][:25]],
        },
        "test_inputs": cleaned,
    }

def _assignment_statement_is_actionable(problem: str, problem_contract: dict[str, Any]) -> bool:
    """Do not let the model invent a task from a title or from student source."""
    actionable = {
        "requirements", "input", "output", "constraints", "specification",
        "expected_behavior", "function_signature", "acceptance_criteria",
    }
    if any(str(key).casefold() in actionable and value not in (None, "", [], {})
           for key, value in problem_contract.items()):
        return True
    text = " ".join(problem.split())
    words = text.split()
    has_task_verb = re.search(
        r"\b(read|given|accept|take|input|output|compute|calculate|find|determine|print|return|"
        r"sort|count|check|implement|write|convert|generate|traverse|search|sum|average|"
        r"minimum|maximum|largest|smallest|solve|identify|produce)\b",
        text, re.IGNORECASE,
    )
    return len(text) >= 32 and len(words) >= 6 and bool(has_task_verb)


def analyze_submission_for_test_generation(
    *,
    problem: str,
    problem_contract: dict[str, Any],
    submission_source: str,
    request_json: Callable[[list[dict[str, str]], dict[str, Any], str], dict[str, Any] | None],
    submission_filename: str = "submission.c",
) -> dict[str, Any]:
    """Understand the submitted C implementation with exactly one LLM call."""
    if not isinstance(submission_source, str) or not submission_source.strip():
        raise TestGenerationError("missing_submission_source", "A non-empty C submission is required before generating tests.")
    if len(submission_source.encode("utf-8")) > MAX_REFERENCE_SOURCE_CHARS:
        raise TestGenerationError(
            "submission_source_too_large",
            f"The C source is too large for the test-design stage (limit: {MAX_REFERENCE_SOURCE_CHARS} bytes).",
        )
    schema = {
        "type": "object",
        "properties": {
            "implementation_summary": {"type": "string", "maxLength": 700},
            "input_parsing_and_validation": {
                "type": "array", "maxItems": 8,
                "items": {"type": "string", "maxLength": 180},
            },
            "control_flow_and_data_types": {
                "type": "array", "maxItems": 8,
                "items": {"type": "string", "maxLength": 180},
            },
            "likely_weak_points": {
                "type": "array", "maxItems": 6,
                "items": {"type": "string", "maxLength": 180},
            },
            "recommended_test_dimensions": {
                "type": "array", "maxItems": 8,
                "items": {"type": "string", "maxLength": 180},
            },
        },
        "required": [
            "implementation_summary", "input_parsing_and_validation",
            "control_flow_and_data_types", "likely_weak_points",
            "recommended_test_dimensions",
        ],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are the implementation-understanding stage of a C test-case generator. "
                "Read the assignment and C source carefully. Explain what the submitted program "
                "actually parses and computes, its branches, data types, assumptions, and likely "
                "failure points. Recommend concrete test dimensions that exercise its paths. "
                "Be concise: use at most 8 short bullets per list (6 for weak points), keep each "
                "bullet under 180 characters, and keep the summary under 700 characters. Prefer "
                "distinct, actionable observations over exhaustive narration. The assignment and "
                "instructor contract are authoritative: do not reinterpret requirements to match "
                "the student's implementation. Do not execute the source, produce expected outputs, "
                "or treat the source's behavior as correct. Do not produce a generic weakness checklist. "
                "Before claiming a validation is missing, inspect the relevant source path and check whether "
                "that guard is actually present (for example, a bounds comparison or scanf return-value check). "
                "The likely_weak_points list must not contradict checks you identified in the other fields. "
                "Only report concrete uncovered paths supported by the source; if unsure, omit the weakness. "
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "assignment": problem,
                    "problem_contract": problem_contract,
                    "submission_filename": submission_filename,
                    "submission_source": submission_source,
                    "instruction": (
                        "Understand this exact implementation in one pass. Return a concise summary "
                        "and only the most useful distinct observations within the schema limits. Identify "
                        "likely edge cases and missing checks that a separate test-generation call should "
                        "probe. Do not supply test inputs or expected outputs."
                    ),
                },
                sort_keys=True,
                ensure_ascii=False,
            ),
        },
    ]
    result = request_json(messages, schema, "assignment_submission_understanding_single_pass")
    if not isinstance(result, dict):
        raise TestGenerationError(
            "submission_understanding_failed",
            "The single implementation-understanding response was not a JSON object; no retry was performed.",
        )
    summary = result.get("implementation_summary")
    list_fields = (
        "input_parsing_and_validation", "control_flow_and_data_types",
        "likely_weak_points", "recommended_test_dimensions",
    )
    if not isinstance(summary, str) or not summary.strip() or any(
        not isinstance(result.get(field), list)
        or not all(isinstance(item, str) for item in result[field])
        for field in list_fields
    ):
        raise TestGenerationError(
            "submission_understanding_failed",
            "The single implementation-understanding response omitted required fields or returned malformed lists; no retry was performed.",
        )
    return {
        "implementation_summary": summary.strip()[:700],
        **{
            field: [item.strip()[:180] for item in result[field] if item.strip()][:8]
            for field in list_fields
        },
    }


def _outputs_match(generated: str, verified: str) -> bool:
    """Use the same whitespace rule as the correctness analyzer."""
    normalize = lambda value: value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalize(generated) == normalize(verified)


def _verify_expected_outputs(
    *,
    problem: str,
    problem_contract: dict[str, Any],
    inferred_contract: dict[str, Any],
    candidate_cases: list[dict[str, Any]],
    instructor_examples: list[dict[str, Any]] | None,
    request_json: Callable[[list[dict[str, str]], dict[str, Any], str], dict[str, Any] | None],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    """Independently solve generated inputs without showing the verifier proposed outputs or student code."""
    schema = {
        "type": "object",
        "properties": {
            "verifications": {
                "type": "array",
                "minItems": len(candidate_cases),
                "maxItems": len(candidate_cases),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "input_is_valid": {"type": "boolean"},
                        "expected_output": {"type": "string"},
                        "reason": {"type": "string", "maxLength": 240},
                    },
                    "required": ["id", "input_is_valid", "expected_output", "reason"],
                },
            },
        },
        "required": ["verifications"],
    }
    examples = [
        {
            "id": str(case.get("id", f"instructor-{index}")),
            "input": case["input"],
            "expected_output": case["expected"],
        }
        for index, case in enumerate(instructor_examples or [], 1)
        if isinstance(case, dict)
        and isinstance(case.get("input"), str)
        and isinstance(case.get("expected"), str)
    ][:20]
    payload = {
        "assignment": problem,
        "problem_contract": problem_contract,
        "inferred_contract": inferred_contract,
        "instructor_examples_as_format_and_behavior_hints": examples,
        "test_inputs_to_solve": [
            {"id": str(case["id"]), "input": case["input"]}
            for case in candidate_cases
        ],
        "instruction": (
            "Independently solve every supplied test input using only the assignment and problem_contract. "
            "The instructor examples are hints for input conventions, output formatting, and known examples; "
            "they are not an exhaustive list of requirements. Treat inferred_contract only as a provisional "
            "restatement: if it conflicts with assignment or problem_contract, follow the latter and flag uncertainty. "
            "Do not infer answers from student code or from any prior model output. For every id, state whether the input is valid and unambiguous "
            "under the assignment, and provide the exact expected stdout for a correct solution, including "
            "required spaces and line breaks. If the specification does not determine a unique answer, "
            "set input_is_valid=false and explain briefly. Return exactly one verification for every id."
        ),
    }
    result = request_json(
        [
            {
                "role": "system",
                "content": (
                    "You are an independent expected-output verifier for programming tests. "
                    "You are not reviewing or executing the student's code. Derive each answer from "
                    "the assignment specification yourself, using instructor examples only to understand "
                    "the intended conventions. Return JSON only."
                ),
            },
            {"role": "user", "content": json.dumps(payload, sort_keys=True, ensure_ascii=False)},
        ],
        schema,
        "assignment_expected_output_verification_single_pass",
    )
    if not isinstance(result, dict) or not isinstance(result.get("verifications"), list):
        raise TestGenerationError(
            "expected_output_verification_failed",
            "The independent verifier did not return the required verifications array. No generated outputs were accepted.",
        )
    checks: dict[str, dict[str, Any]] = {}
    for item in result["verifications"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise TestGenerationError(
                "expected_output_verification_failed",
                "The independent verifier returned a malformed verification item. No generated outputs were accepted.",
            )
        case_id = item["id"]
        if case_id in checks:
            raise TestGenerationError(
                "expected_output_verification_failed",
                f"The independent verifier returned duplicate results for test {case_id}.",
            )
        checks[case_id] = item

    expected_ids = {str(case["id"]) for case in candidate_cases}
    missing = sorted(expected_ids - set(checks))
    unexpected = sorted(set(checks) - expected_ids)
    if missing:
        raise TestGenerationError(
            "expected_output_verification_failed",
            f"The independent verifier omitted test {missing[0]}. No generated outputs were accepted.",
        )
    if unexpected:
        raise TestGenerationError(
            "expected_output_verification_failed",
            f"The independent verifier returned unexpected test IDs {unexpected[:5]}. No generated outputs were accepted.",
        )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for case in candidate_cases:
        case_id = str(case["id"])
        check = checks.get(case_id)
        if check is None:
            raise TestGenerationError(
                "expected_output_verification_failed",
                f"The independent verifier omitted test {case_id}. No generated outputs were accepted.",
            )
        if check.get("input_is_valid") is not True:
            reason = check.get("reason")
            rejected.append({
                "id": case_id,
                "reason": ("independent verifier found an invalid or ambiguous input: " + str(reason))[:300],
            })
            continue
        verified_output = check.get("expected_output")
        if not isinstance(verified_output, str) or len(verified_output) > MAX_INPUT_CHARS:
            raise TestGenerationError(
                "expected_output_verification_failed",
                f"The independent verifier returned an invalid expected_output for test {case_id}.",
            )
        if not _outputs_match(case["expected_output"], verified_output):
            rejected.append({
                "id": case_id,
                "reason": "generated expected_output disagreed with the independent verifier",
            })
            continue
        canonical_output = verified_output.replace("\r\n", "\n").replace("\r", "\n")
        accepted.append({**case, "expected_output": canonical_output})

    summary = {
        "method": "independent_second_llm_call",
        "requested": len(candidate_cases),
        "verified": len(accepted),
        "rejected": len(rejected),
        "comparison": "trimmed exact text with normalized line endings",
    }
    return accepted, rejected, summary


def generate_assignment_test_spec(
    *, problem: str, problem_contract: dict[str, Any],
    request_json: Callable[[list[dict[str, str]], dict[str, Any], str], dict[str, Any] | None],
    passes: int = 3, generation_seed: int = 42, max_cases: int = 60,
    additional_focus: str = "",
    instructor_examples: list[dict[str, Any]] | None = None,
    trusted_reference_solution: str | None = None,
    submission_analysis: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate input/output pairs once, then independently verify them once.

    trusted_reference_solution remains an ignored compatibility argument. No
    generated or configured C reference source is used by this pipeline.
    """
    if not isinstance(problem, str) or not problem.strip():
        raise TestGenerationError("missing_assignment", "A complete assignment statement is required.")
    if not isinstance(problem_contract, dict):
        raise TestGenerationError("invalid_contract", "problem_contract must be an object.")
    if not _assignment_statement_is_actionable(problem, problem_contract):
        raise TestGenerationError(
            "underspecified_assignment",
            "The assignment is only a title or lacks actionable requirements. Provide the complete natural-language problem statement or an instructor problem_contract; the student's source cannot be used to guess the task.",
        )
    if isinstance(passes, bool) or not isinstance(passes, int) or not 1 <= passes <= 5:
        raise ValueError("passes must be an integer from 1 to 5")
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or not 1 <= max_cases <= MAX_GENERATED_CASES:
        raise ValueError(f"max_cases must be from 1 to {MAX_GENERATED_CASES}")
    if not isinstance(additional_focus, str) or len(additional_focus) > 500:
        raise ValueError("additional_focus must be text of at most 500 characters")

    fallback_contract = {
        "input_format": str(problem_contract.get("input_format") or problem),
        "output_format": str(problem_contract.get("output_format") or problem_contract.get("output_rule") or problem),
        "constraints": [
            f"{key}: {value}" for key, value in problem_contract.items()
            if any(marker in str(key).casefold() for marker in ("constraint", "range", "limit"))
            and value not in (None, "", [], {})
        ],
        "semantics": [
            str(problem_contract.get("description") or problem_contract.get("requirements") or problem)
        ],
        "formatting": [
            str(problem_contract.get("formatting") or problem_contract.get("output_rule") or problem)
        ],
    }
    examples = [
        {
            "id": str(case.get("id", f"instructor-{index}")),
            "input": case["input"],
            "expected_output": case["expected"],
        }
        for index, case in enumerate(instructor_examples or [], 1)
        if isinstance(case, dict)
        and isinstance(case.get("input"), str)
        and isinstance(case.get("expected"), str)
    ][:20]
    base = {
        "assignment": problem,
        "problem_contract": problem_contract,
        "instructor_examples_as_hints": examples,
        "generation_seed": generation_seed,
        "maximum_cases": max_cases,
        "requested_case_count": max_cases,
        "additional_coverage_focus": additional_focus.strip(),
        "student_submission_included": bool(submission_analysis),
    }
    if submission_analysis:
        base["submitted_implementation_analysis"] = submission_analysis

    generation_schema = {
        "type": "object",
        "properties": {
            "spec_summary": {"type": "string", "maxLength": 2_000},
            "ambiguities": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
            "inferred_contract": {
                "type": "object",
                "properties": {
                    "input_format": {"type": "string"},
                    "output_format": {"type": "string"},
                    "constraints": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
                    "semantics": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
                    "formatting": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
                },
                "required": ["input_format", "output_format", "constraints", "semantics", "formatting"],
            },
            "test_inputs": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_cases,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "input": {"type": "string", "maxLength": MAX_INPUT_CHARS},
                        "expected_output": {"type": "string", "maxLength": MAX_INPUT_CHARS},
                        "tags": {"type": "array", "items": {"type": "string", "maxLength": 24}, "maxItems": 3},
                    },
                    "required": ["id", "input", "expected_output"],
                },
            },
        },
        "required": ["ambiguities", "inferred_contract", "test_inputs"],
    }
    system = (
        "You generate test input and expected-output pairs for a C programming assignment. "
        "The assignment and instructor problem_contract are the only authority for correctness. "
        "Use submitted_implementation_analysis only to choose useful test dimensions; never infer "
        "expected outputs from the student's implementation. Instructor examples are guidance for "
        "input grammar and output conventions, not an exhaustive list of cases or requirements. "
        "First formalize the I/O contract. Generate diverse valid inputs and independently solve each "
        "input from the assignment to provide expected_output. Do not write, propose, or return C code. "
        "Do not invent requirements or guess ambiguous behaviour. If a required rule is genuinely "
        "ambiguous, report it in ambiguities. Include normal, boundary, empty/singleton, negative/zero, "
        "and task-specific cases only when valid. Prefer concise inputs and outputs. Return JSON only."
    )
    payload = dict(
        base,
        instruction=(
            f"Generate up to {max_cases} additional distinct test cases. Each test case must contain "
            "id, input, expected_output, and up to three short tags. Derive expected_output from the "
            "assignment contract, not from the submitted code. Follow the style shown by instructor "
            "examples while exploring new inputs. Do not repeat instructor inputs. Do not include "
            "C source, coverage narration, or long rationales."
        ),
    )
    result = request_json(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, sort_keys=True, ensure_ascii=False)},
        ],
        generation_schema,
        "assignment_test_generation_single_pass",
    )
    candidate = _validate_candidate(
        result,
        1,
        allow_ambiguities=False,
        fallback_summary=problem,
        fallback_contract=fallback_contract,
    )

    rejected: list[dict[str, str]] = []
    contract_valid_cases: list[dict[str, Any]] = []
    instructor_inputs = {
        case["input"] for case in (instructor_examples or [])
        if isinstance(case, dict) and isinstance(case.get("input"), str)
    }
    for case in candidate["test_inputs"]:
        reason = _validate_input_against_contract(case["input"], problem_contract)
        if reason:
            rejected.append({"id": case["id"], "reason": reason})
        elif case["input"] in instructor_inputs:
            rejected.append({"id": case["id"], "reason": "duplicate of an instructor example"})
        else:
            contract_valid_cases.append(case)
    if not contract_valid_cases:
        raise TestGenerationError(
            "no_valid_generated_inputs",
            "The generator did not produce any distinct inputs that passed deterministic contract validation.",
        )

    verified_cases, verification_rejections, verification_summary = _verify_expected_outputs(
        problem=problem,
        problem_contract=problem_contract,
        inferred_contract=candidate["inferred_contract"],
        candidate_cases=contract_valid_cases,
        instructor_examples=instructor_examples,
        request_json=request_json,
    )
    rejected.extend(verification_rejections)
    if not verified_cases:
        preview = "; ".join(f"{item['id']}: {item['reason']}" for item in rejected[:5])
        raise TestGenerationError(
            "no_generated_cases_verified",
            "No generated input/output pair passed independent LLM verification. "
            + (preview or "Review the assignment contract and instructor examples."),
        )

    candidate["test_inputs"] = verified_cases
    candidate["generation_mode"] = "generate_then_independently_verify"
    candidate["generation_response_count"] = 1
    candidate["verification_response_count"] = 1
    candidate["verification_summary"] = verification_summary
    candidate["rejected_generated_cases"] = rejected
    history = [{
        "pass": 1,
        "status": "completed",
        "mode": "generate_then_independently_verify",
        "test_count": len(verified_cases),
        "requested_test_count": max_cases,
        "rejected_test_count": len(rejected),
        "ambiguities_reported": len(candidate["ambiguities"]),
        "verification": verification_summary,
    }]
    candidate["generation_passes"] = history
    return candidate, history

def _validate_input_against_contract(test_input: str, problem_contract: dict[str, Any]) -> str | None:
    """Validate stdin against explicit, machine-readable instructor constraints.

    Free-form prose cannot be interpreted as a formal grammar. Strict structural
    validation is available through input_schema/input_fields token descriptors.
    """
    if not isinstance(test_input, str) or not test_input.strip():
        return "input must be a non-empty string"
    if len(test_input) > MAX_INPUT_CHARS:
        return "input exceeds the maximum size"
    if "\x00" in test_input or any(ord(ch) < 32 and ch not in "\r\n\t" for ch in test_input):
        return "input contains a forbidden control character"
    if not isinstance(problem_contract, dict):
        return None

    lines = test_input.splitlines()
    for key, actual in (("input_lines_min", len(lines)), ("input_lines_max", len(lines))):
        limit = problem_contract.get(key)
        if isinstance(limit, int) and not isinstance(limit, bool):
            if key.endswith("_min") and actual < limit:
                return f"input has fewer than {limit} required lines"
            if key.endswith("_max") and actual > limit:
                return f"input has more than {limit} allowed lines"

    tokens = test_input.split()
    exact_count = problem_contract.get("input_token_count", problem_contract.get("expected_token_count"))
    min_count = problem_contract.get("min_input_tokens")
    max_count = problem_contract.get("max_input_tokens")
    if isinstance(exact_count, int) and not isinstance(exact_count, bool) and len(tokens) != exact_count:
        return f"expected exactly {exact_count} whitespace-delimited input tokens"
    if isinstance(min_count, int) and len(tokens) < min_count:
        return f"expected at least {min_count} whitespace-delimited input tokens"
    if isinstance(max_count, int) and len(tokens) > max_count:
        return f"expected at most {max_count} whitespace-delimited input tokens"

    schema = problem_contract.get("input_schema", problem_contract.get("input_fields"))
    if schema is None:
        return None
    allow_extra = False
    fields = schema
    if isinstance(schema, dict):
        fields = schema.get("fields", schema.get("tokens"))
        allow_extra = bool(schema.get("allow_extra", False))
    if not isinstance(fields, list) or not all(isinstance(field, dict) for field in fields):
        return "input_schema must contain a fields list"
    cursor = 0
    for field_index, field in enumerate(fields, 1):
        name = str(field.get("name", f"field_{field_index}"))
        count = field.get("count", 1)
        optional = bool(field.get("optional", False))
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return f"input_schema count for {name} must be a non-negative integer"
        remaining = len(tokens) - cursor
        if remaining < count and not optional:
            return f"input is missing required field {name}"
        values = tokens[cursor:cursor + min(count, max(0, remaining))]
        cursor += len(values)
        kind = str(field.get("type", "string")).casefold()
        for token in values:
            number = None
            if kind in {"int", "integer", "signed_integer", "long", "long_long"}:
                if not re.fullmatch(r"[+-]?\d+", token):
                    return f"field {name} must be an integer"
                number = int(token)
            elif kind in {"uint", "unsigned", "unsigned_integer", "natural"}:
                if not re.fullmatch(r"\+?\d+", token):
                    return f"field {name} must be a non-negative integer"
                number = int(token)
            elif kind in {"float", "double", "number"}:
                try:
                    number = float(token)
                except ValueError:
                    return f"field {name} must be numeric"
                if not math.isfinite(number):
                    return f"field {name} must be finite"
            if number is not None:
                lower = field.get(
                    "min", field.get(
                        "minimum",
                        0 if kind in {"uint", "unsigned", "unsigned_integer", "natural"} else None,
                    ),
                )
                upper = field.get("max", field.get("maximum"))
                if lower is not None and number < lower:
                    return f"field {name} is below its minimum"
                if upper is not None and number > upper:
                    return f"field {name} exceeds its maximum"
            allowed = field.get("enum", field.get("allowed_values"))
            if isinstance(allowed, list) and token not in {str(value) for value in allowed}:
                return f"field {name} is not one of its allowed values"
            min_length, max_length = field.get("min_length"), field.get("max_length")
            if isinstance(min_length, int) and len(token) < min_length:
                return f"field {name} is shorter than its minimum length"
            if isinstance(max_length, int) and len(token) > max_length:
                return f"field {name} exceeds its maximum length"
    if not allow_extra and cursor != len(tokens):
        return f"input contains {len(tokens) - cursor} unexpected token(s)"
    return None


def materialize_test_cases(
    generated_spec: dict[str, Any], *,
    instructor_cases: list[dict[str, Any]] | None = None,
    rejected_generated_cases: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Merge instructor examples with independently verified generated pairs."""
    generated = generated_spec.get("test_inputs")
    if not isinstance(generated, list):
        raise TestGenerationError("invalid_spec", "Generated test inputs are missing.")
    merged: list[dict[str, Any]] = []
    seen_inputs: dict[str, str] = {}
    seen_ids: set[str] = set()

    for index, case in enumerate(instructor_cases or [], 1):
        if not isinstance(case, dict) or not isinstance(case.get("input"), str) or not isinstance(case.get("expected"), str):
            raise TestGenerationError("invalid_instructor_test", f"Instructor test {index} is malformed.")
        test_input, expected = case["input"], case["expected"]
        if test_input in seen_inputs:
            if not _outputs_match(seen_inputs[test_input], expected):
                raise TestGenerationError("conflicting_instructor_tests", f"Instructor examples disagree about expected output for case {case.get('id', index)}.")
            for prior in merged:
                if prior.get("input") == test_input:
                    origins = prior.setdefault("origin_metrics", [])
                    for origin in case.get("origin_metrics", []):
                        if origin not in origins:
                            origins.append(origin)
                    tags = prior.setdefault("tags", [])
                    for tag in case.get("tags", []):
                        if tag not in tags:
                            tags.append(tag)
                    break
            continue
        final_id = f"instructor-{case.get('id', f'instructor-{index}')}"
        if final_id in seen_ids:
            raise TestGenerationError("duplicate_test_id", f"Duplicate instructor test ID: {final_id}.")
        seen_inputs[test_input] = expected
        seen_ids.add(final_id)
        merged.append({**case, "id": final_id, "input": test_input, "expected": expected, "origin": "instructor"})

    for index, case in enumerate(generated, 1):
        if not isinstance(case, dict):
            raise TestGenerationError("invalid_generated_test", f"Generated test {index} is malformed.")
        test_input = case.get("input")
        expected = case.get("expected_output", case.get("expected"))
        case_id = str(case.get("id", f"generated-{index}"))
        if not isinstance(test_input, str) or not isinstance(expected, str):
            if rejected_generated_cases is not None:
                rejected_generated_cases.append({"id": case_id, "reason": "verified input/expected_output pair is malformed"})
                continue
            raise TestGenerationError("invalid_generated_test", f"Generated test {case_id} lacks a string input or expected output.")
        if test_input in seen_inputs:
            if not _outputs_match(seen_inputs[test_input], expected):
                raise TestGenerationError("generated_instructor_disagreement", f"Generated expected output disagrees with an instructor example for test {case_id}.")
            continue
        final_id = f"generated-{case_id}"
        if final_id in seen_ids:
            raise TestGenerationError("duplicate_test_id", f"Generated test ID collides after normalization: {case_id}.")
        seen_inputs[test_input] = expected
        seen_ids.add(final_id)
        merged.append({
            "id": final_id,
            "input": test_input,
            "expected": expected,
            "tags": list(case.get("tags", [])),
            "description": str(case.get("description", ""))[:240],
            "origin": "llm_generated_independently_verified",
        })
    if not merged:
        raise TestGenerationError("empty_suite", "No instructor or independently verified generated cases are available.")
    return merged

def write_frozen_suite(
    cache_dir: Path, *, fingerprint: str, cases: list[dict[str, Any]],
    generated_spec: dict[str, Any], model_id: str,
    assignment_statement: str = "", problem_contract: dict[str, Any] | None = None,
    oracle_kind: str = "llm-generated-outputs-independently-verified",
) -> Path:
    """Freeze verified expected-output pairs; never silently replace a conflicting suite."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{fingerprint}.json"
    payload = {
        "schema_version": 3,
        "assignment_fingerprint": fingerprint,
        "model_id": model_id,
        "test_cases": cases,
        "generation_passes": generated_spec.get("generation_passes", generated_spec.get("test_generation_rsi", [])),
        "rejected_generated_cases": generated_spec.get("rejected_generated_cases", []),
        "generation_ambiguities": generated_spec.get("ambiguities", []),
        "spec_summary": generated_spec.get("spec_summary", ""),
        "submission_analysis": generated_spec.get("submission_analysis", {}),
        "inferred_contract": generated_spec.get("inferred_contract", {}),
        "expected_output_verification": generated_spec.get("verification_summary", {}),
        "oracle_kind": oracle_kind,
        "assignment_statement": assignment_statement,
        "problem_contract": problem_contract or {},
        "oracle_caveat": (
            "Generated expected outputs were independently rechecked by a second LLM call against the assignment specification. "
            "When available, instructor-provided examples guided input and output conventions; they are not an exhaustive oracle."
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise TestGenerationError("frozen_suite_conflict", "A frozen suite already exists for this fingerprint but differs; do not overwrite silently.")
        return path
    import os
    import tempfile
    fd, temporary_name = tempfile.mkstemp(prefix=f"{fingerprint}.", suffix=".tmp", dir=cache_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing != encoded:
                raise TestGenerationError("frozen_suite_conflict", "A frozen suite was created concurrently with different contents; review the verification record before grading.")
            os.unlink(temporary_name)
            return path
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path

def load_frozen_suite(path: Path, expected_fingerprint: str) -> list[dict[str, Any]]:
    """Load a frozen suite only when its assignment fingerprint matches."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TestGenerationError("suite_unreadable", f"Cannot read frozen test suite: {exc}") from exc
    if not isinstance(data, dict):
        raise TestGenerationError("suite_invalid", "Frozen test suite must contain a JSON object.")
    if data.get("assignment_fingerprint") != expected_fingerprint:
        raise TestGenerationError("suite_fingerprint_mismatch", "Frozen suite does not match the current assignment fingerprint.")
    cases = data.get("test_cases")
    if not isinstance(cases, list) or not cases:
        raise TestGenerationError("suite_empty", "Frozen suite is empty or malformed.")
    return cases




def prepare_assignment_test_suite(
    rules: dict[str, Any],
    *,
    sandbox: Any,
    request_json: Callable[[list[dict[str, str]], dict[str, Any], str], dict[str, Any] | None],
    project_root: Path,
    model_id: str,
    generation_seed: int = 42,
    temperature: float = 0.0,
    submission_source: str = "",
    submission_filename: str = "submission.c",
) -> dict[str, Any]:
    """Generate, independently verify, freeze, and bind assignment test cases.

    No reference C program is generated, loaded, compiled, or executed. The
    first LLM call proposes input/output pairs; a second LLM call independently
    solves the proposed inputs from the assignment specification. The sandbox
    is reserved for grading the student's original submission later.
    """
    testing_cfg = rules.get("test_generation", {})
    testing_cfg = testing_cfg if isinstance(testing_cfg, dict) else {}
    if not testing_cfg.get("enabled", True):
        return {"status": "disabled", "reason": "test_generation.enabled=false"}

    execution_metrics = [
        metric for metric in rules.get("metrics", [])
        if metric.get("enabled", True) and metric.get("analyzer") in {
            "io_tests", "edge_case_runner", "memory_safety", "runtime_profile"
        }
    ]
    if not execution_metrics:
        return {"status": "not_required", "reason": "No enabled execution-based metrics."}

    project_root = Path(project_root).resolve()
    instructor_by_input: dict[str, dict[str, Any]] = {}
    for metric in execution_metrics:
        params = metric.get("params", {})
        path_value = params.get("resolved_testset_path") if params.get("testset_path") not in (None, "") else None
        if path_value:
            try:
                teacher_data = json.loads(Path(path_value).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise TestGenerationError("instructor_tests_unreadable", f"Could not read instructor tests for {metric.get('name')}: {exc}") from exc
            if not isinstance(teacher_data, list):
                raise TestGenerationError("instructor_tests_invalid", f"Instructor tests for {metric.get('name')} must be a JSON array.")
            for index, item in enumerate(teacher_data, 1):
                if not isinstance(item, dict) or not isinstance(item.get("input"), str) or not isinstance(item.get("expected"), str):
                    raise TestGenerationError("instructor_tests_invalid", f"Instructor case {index} for {metric.get('name')} is malformed.")
                key = item["input"]
                if key not in instructor_by_input:
                    instructor_by_input[key] = {**item, "origin_metrics": [str(metric.get("name"))]}
                else:
                    prior = instructor_by_input[key]
                    if not _outputs_match(prior["expected"], item["expected"]):
                        raise TestGenerationError("conflicting_instructor_tests", f"Instructor metrics disagree on expected output for identical input ({metric.get('name')}, case {item.get('id', index)}).")
                    prior.setdefault("origin_metrics", []).append(str(metric.get("name")))

    instructor_cases = list(instructor_by_input.values())
    seed = int(generation_seed)
    max_cases = testing_cfg.get("max_cases", 12)
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or not 1 <= max_cases <= MAX_GENERATED_CASES:
        raise TestGenerationError("invalid_test_generation_config", f"test_generation.max_cases must be from 1 to {MAX_GENERATED_CASES}.")
    additional_focus = testing_cfg.get("additional_focus", "")
    if not isinstance(additional_focus, str) or len(additional_focus) > 500:
        raise TestGenerationError("invalid_test_generation_config", "test_generation.additional_focus must be text of at most 500 characters.")
    submission_source_sha256 = (
        hashlib.sha256(submission_source.encode("utf-8")).hexdigest()
        if isinstance(submission_source, str) and submission_source.strip() else ""
    )
    fingerprint = assignment_fingerprint(
        problem=str(rules.get("problem", "")),
        problem_contract=rules.get("problem_contract", {}),
        model_id=model_id,
        input_format=rules.get("problem_contract", {}).get("input_format") if isinstance(rules.get("problem_contract"), dict) else None,
        output_format=rules.get("problem_contract", {}).get("output_format") if isinstance(rules.get("problem_contract"), dict) else None,
        generation_seed=int(seed),
        generator_version=str(testing_cfg.get("generator_version", "1")) + ":two-pass-llm-verified-outputs-v14",
        rsi_passes=1,
        max_cases=int(max_cases),
        temperature=float(temperature),
        additional_focus=additional_focus,
        submission_source_sha256=submission_source_sha256,
    )
    supplemental = {
        "base": fingerprint,
        "instructor_cases": instructor_cases,
        "expected_output_verification": "independent_second_llm_call",
    }
    try:
        fingerprint = hashlib.sha256(
            json.dumps(supplemental, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise TestGenerationError("invalid_fingerprint_data", f"Assignment fingerprint data is not valid JSON: {exc}") from exc

    cache_dir_value = testing_cfg.get("cache_dir", ".generated_tests")
    cache_dir = _project_relative_path(project_root, cache_dir_value, "test_generation.cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = cache_dir / f"{fingerprint}.json"
    require_frozen = testing_cfg.get("require_frozen_suite", False)
    if not isinstance(require_frozen, bool):
        raise TestGenerationError("invalid_test_generation_config", "test_generation.require_frozen_suite must be boolean.")
    configured_seed = testing_cfg.get("seed", generation_seed)
    if isinstance(configured_seed, bool) or not isinstance(configured_seed, int) or configured_seed != int(generation_seed):
        raise TestGenerationError("generation_seed_mismatch", "test_generation.seed must match the llama-server seed configured for this run.")
    configured_temperature = testing_cfg.get("temperature", temperature)
    if isinstance(configured_temperature, bool) or not isinstance(configured_temperature, (int, float)) or float(configured_temperature) != float(temperature):
        raise TestGenerationError("generation_temperature_mismatch", "test_generation.temperature must match the llama-server temperature configured for this run.")

    if not frozen_path.is_file() and require_frozen:
        raise TestGenerationError(
            "frozen_suite_required_missing",
            f"Strict reproducibility requires the pre-generated assignment suite {frozen_path.name}; distribute the canonical .generated_tests artifact instead of regenerating it on this machine.",
        )

    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen.get("assignment_fingerprint") != fingerprint:
            raise TestGenerationError("suite_fingerprint_mismatch", "Frozen suite fingerprint does not match the current assignment.")
        if frozen.get("schema_version") != 3:
            raise TestGenerationError("suite_schema_version_mismatch", "Frozen suite uses an obsolete reference-program schema; regenerate it with the two-pass expected-output verifier.")
        all_cases = frozen.get("test_cases")
        if (
            not isinstance(all_cases, list)
            or not all_cases
            or any(
                not isinstance(case, dict)
                or not isinstance(case.get("input"), str)
                or not isinstance(case.get("expected"), str)
                for case in all_cases
            )
        ):
            raise TestGenerationError("suite_empty", "Frozen assignment suite has missing or malformed input/expected-output pairs.")
        generation_history = frozen.get("generation_passes", [])
        suite_source = "frozen_cache"
    else:
        submission_analysis = None
        if submission_source_sha256:
            submission_analysis = analyze_submission_for_test_generation(
                problem=str(rules.get("problem", "")),
                problem_contract=rules.get("problem_contract", {}),
                submission_source=submission_source,
                submission_filename=submission_filename,
                request_json=request_json,
            )
        spec, generation_history = generate_assignment_test_spec(
            problem=str(rules.get("problem", "")),
            problem_contract=rules.get("problem_contract", {}),
            request_json=request_json,
            passes=1,
            generation_seed=int(seed),
            max_cases=int(max_cases),
            additional_focus=additional_focus,
            instructor_examples=instructor_cases,
            submission_analysis=submission_analysis,
        )
        if submission_analysis is not None:
            spec["submission_analysis"] = submission_analysis
        rejected_generated_cases: list[dict[str, str]] = list(spec.get("rejected_generated_cases", []))
        all_cases = materialize_test_cases(
            spec,
            instructor_cases=instructor_cases,
            rejected_generated_cases=rejected_generated_cases,
        )
        instructor_suite = [case for case in all_cases if case.get("origin") == "instructor"]
        generated_suite = [
            case for case in all_cases
            if case.get("origin") == "llm_generated_independently_verified"
        ]
        if not generated_suite:
            preview = "; ".join(
                f"{item.get('id', 'candidate')}: {item.get('reason', 'rejected')}"
                for item in rejected_generated_cases[:5]
            )
            raise TestGenerationError(
                "no_generated_cases_verified",
                "Requested additional cases, but no generated pair survived independent expected-output verification. "
                + (preview or "Review the assignment specification and instructor examples."),
            )
        all_cases = instructor_suite + generated_suite[:int(max_cases)]
        spec["rejected_generated_cases"] = rejected_generated_cases
        oracle_kind = (
            "instructor-examples-plus-two-pass-llm-verification"
            if instructor_cases else
            "two-pass-llm-verified-expected-outputs"
        )
        suite_path = write_frozen_suite(
            cache_dir,
            fingerprint=fingerprint,
            cases=all_cases,
            generated_spec=spec,
            model_id=model_id,
            assignment_statement=str(rules.get("problem", "")),
            problem_contract=rules.get("problem_contract", {}),
            oracle_kind=oracle_kind,
        )
        frozen = json.loads(suite_path.read_text(encoding="utf-8"))
        suite_source = "generated_and_frozen"

    edge_markers = {
        "edge", "boundary", "corner", "overflow", "extreme", "empty", "singleton",
        "duplicate", "adversarial", "minimum", "maximum", "negative", "zero", "large",
        "small", "max", "min", "limit", "numeric", "integer",
    }

    def is_edge_case(case: dict[str, Any]) -> bool:
        tags = case.get("tags", [])
        origins = case.get("origin_metrics", [])
        if isinstance(origins, list) and {str(x).casefold() for x in origins}.intersection(
            {"logical", "edge_cases", "edge_case_runner"}
        ):
            return True
        text_parts = [str(tag) for tag in tags] if isinstance(tags, list) else []
        text_parts.extend(str(case.get(key, "")) for key in ("description", "rationale"))
        words = set(re.findall(r"[a-z0-9]+", " ".join(text_parts).casefold()))
        return bool(words & edge_markers) or bool({"edge", "boundary", "corner"} & words)

    edge_cases = [case for case in all_cases if isinstance(case, dict) and is_edge_case(case)]
    if not edge_cases and any(metric.get("analyzer") == "edge_case_runner" for metric in execution_metrics):
        raise TestGenerationError("edge_coverage_missing", "Generated test suite contains no identifiable edge/boundary cases; logical metric cannot be measured.")

    all_path = cache_dir / f"{fingerprint}-all.json"
    edge_path = cache_dir / f"{fingerprint}-edges.json"
    for target, content in ((all_path, all_cases), (edge_path, edge_cases)):
        encoded = json.dumps(content, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        if target.exists() and target.read_text(encoding="utf-8") != encoded:
            raise TestGenerationError("testset_cache_conflict", f"Refusing to overwrite conflicting generated test file {target.name}.")
        if not target.exists():
            target.write_text(encoded, encoding="utf-8")

    metric_bindings: dict[str, str] = {}
    for metric in execution_metrics:
        analyzer = metric.get("analyzer")
        target = edge_path if analyzer == "edge_case_runner" else all_path
        metric.setdefault("params", {})["resolved_testset_path"] = str(target)
        metric_bindings[str(metric.get("name"))] = str(target)
    generated_count = sum(
        case.get("origin") == "llm_generated_independently_verified"
        for case in all_cases if isinstance(case, dict)
    )
    verification_summary = frozen.get("expected_output_verification", {})
    return {
        "status": "completed",
        "source": suite_source,
        "assignment_fingerprint": fingerprint,
        "suite_path": str(frozen_path),
        "instructor_test_count": len(instructor_cases),
        "total_test_count": len(all_cases),
        "generated_test_count": generated_count,
        "requested_generated_test_count": int(max_cases),
        "generated_test_count_shortfall": max(0, int(max_cases) - generated_count),
        "rejected_generated_test_count": len(frozen.get("rejected_generated_cases", [])),
        "edge_test_count": len(edge_cases),
        "generation_passes": generation_history,
        "metric_testset_paths": metric_bindings,
        "all_testset_path": str(all_path),
        "edge_testset_path": str(edge_path),
        "submission_analysis": frozen.get("submission_analysis", {}),
        "inferred_contract": frozen.get("inferred_contract", {}),
        "spec_summary": frozen.get("spec_summary", ""),
        "oracle_kind": frozen.get("oracle_kind", "two-pass-llm-verified-expected-outputs"),
        "oracle_caveat": frozen.get("oracle_caveat", "Expected outputs were independently checked by a second LLM call against the assignment specification."),
        "expected_output_verification": verification_summary,
        "unresolved_ambiguities": frozen.get("generation_ambiguities", []),
    }
