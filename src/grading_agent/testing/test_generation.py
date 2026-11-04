"""Single-pass, source-aware test generation with deterministic candidate validation.

The LLM first studies the submitted implementation once, then proposes test
inputs once. Invalid or duplicate candidates are discarded; accepted inputs
must also execute successfully against a trusted or independent reference
solution in the restricted sandbox. The resulting suite is frozen by a
source-aware fingerprint.
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
    fallback_reference_solution: str = "",
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TestGenerationError("invalid_response", f"Generation response {pass_number} did not return an object.")
    summary = value.get("spec_summary", value.get("summary"))
    ambiguity = value.get("ambiguities", value.get("unresolved_ambiguities", []))
    reference = value.get("reference_solution_c", value.get("reference_solution"))
    cases = value.get("test_inputs", value.get("test_cases", value.get("tests")))
    reference_source = "model"
    if fallback_reference_solution.strip():
        reference = fallback_reference_solution
        reference_source = "trusted_instructor_reference"
    if not isinstance(summary, str) or not summary.strip():
        # The assignment text is instructor-authored and already passed the
        # actionable-contract check. Reuse it verbatim rather than discarding
        # otherwise valid tests because the model omitted optional prose.
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
    if not isinstance(reference, str) or not reference.strip() or len(reference) > MAX_REFERENCE_SOURCE_CHARS:
        raise TestGenerationError("missing_reference", f"Generation response {pass_number} omitted a bounded C reference solution.")
    if not isinstance(cases, list) or not cases or len(cases) > MAX_GENERATED_CASES:
        raise TestGenerationError("invalid_test_count", f"Generation response {pass_number} must return 1..{MAX_GENERATED_CASES} test inputs.")

    cleaned, seen_ids, seen_inputs = [], set(), set()
    for index, item in enumerate(cases, 1):
        if isinstance(item, str):
            item = {"input": item}
        if not isinstance(item, dict):
            raise TestGenerationError("invalid_test", f"Generated test {index} must contain an input string.")
        case_id = item.get("id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", case_id):
            case_id = f"generated-{index}"
        if case_id in seen_ids:
            raise TestGenerationError("duplicate_test_id", f"Duplicate generated test ID: {case_id}.")
        test_input = item.get("input", item.get("stdin"))
        if not isinstance(test_input, str) or len(test_input) > MAX_INPUT_CHARS:
            raise TestGenerationError("invalid_test_input", f"Generated test {case_id} has invalid input.")
        tags = item.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(x, str) for x in tags):
            tags = []
        if not tags:
            tags = ["generated"]
        rationale = item.get("rationale", item.get("description", ""))
        if not isinstance(rationale, str) or not rationale.strip():
            rationale = "Generated input; expected output is computed by the configured reference oracle."
        seen_ids.add(case_id)
        if test_input in seen_inputs:
            continue
        seen_inputs.add(test_input)
        cleaned.append({"id": case_id, "input": test_input, "tags": sorted(set(tags)), "description": rationale.strip()[:240]})
    if not cleaned:
        raise TestGenerationError("empty_after_deduplication", "All generated inputs were duplicates.")
    claims = value.get("coverage_claims", [])
    if not isinstance(claims, list) or not all(isinstance(x, str) for x in claims):
        claims = []
    return {
        "spec_summary": summary.strip()[:2_000],
        "spec_summary_source": "assignment_statement" if value.get("spec_summary") != summary else "model",
        "inferred_contract_source": contract_source,
        "reference_solution_source": reference_source,
        "ambiguities": [item.strip()[:500] for item in ambiguity if item.strip()][:10],
        "inferred_contract": {
            "input_format": inferred["input_format"].strip()[:1_000],
            "output_format": inferred["output_format"].strip()[:1_000],
            "constraints": [item.strip()[:400] for item in inferred["constraints"][:25]],
            "semantics": [item.strip()[:400] for item in inferred["semantics"][:25]],
            "formatting": [item.strip()[:400] for item in inferred["formatting"][:25]],
        },
        "reference_solution_c": reference, "test_inputs": cleaned,
        "coverage_claims": claims[:30],
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


def generate_assignment_test_spec(
    *, problem: str, problem_contract: dict[str, Any],
    request_json: Callable[[list[dict[str, str]], dict[str, Any], str], dict[str, Any] | None],
    passes: int = 3, generation_seed: int = 42, max_cases: int = 60,
    additional_focus: str = "",
    instructor_examples: list[dict[str, Any]] | None = None,
    trusted_reference_solution: str | None = None,
    submission_analysis: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate a reference solution and test inputs in one LLM call.

    The passes parameter is retained for API compatibility; testcase generation
    now always uses one pass. Candidate validation and reference execution are
    deterministic and do not trigger additional LLM generation calls.
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

    schema = {
        "type": "object", "properties": {
            "spec_summary": {"type": "string"},
            "ambiguities": {"type": "array", "items": {"type": "string"}},
            "inferred_contract": {"type": "object", "properties": {
                "input_format": {"type": "string"},
                "output_format": {"type": "string"},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "semantics": {"type": "array", "items": {"type": "string"}},
                "formatting": {"type": "array", "items": {"type": "string"}},
            }, "required": ["input_format", "output_format", "constraints", "semantics", "formatting"]},
            "reference_solution_c": {"type": "string"},
            "test_inputs": {"type": "array", "minItems": max_cases, "maxItems": max_cases, "items": {
                "type": "object", "properties": {
                    "id": {"type": "string"}, "input": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                }, "required": ["id", "input", "tags", "rationale"],
            }},
            "coverage_claims": {"type": "array", "items": {"type": "string"}},
        }, "required": ["spec_summary", "ambiguities", "inferred_contract", "reference_solution_c", "test_inputs", "coverage_claims"],
    }
    if trusted_reference_solution:
        # When the instructor supplies the oracle, keep the model response
        # focused on test inputs instead of asking a small model to reproduce
        # the contract, summary, and reference implementation as well.
        for optional_field in ("spec_summary", "inferred_contract", "reference_solution_c", "coverage_claims"):
            schema["properties"].pop(optional_field, None)
            if optional_field in schema["required"]:
                schema["required"].remove(optional_field)
    base = {
        "assignment": problem, "problem_contract": problem_contract,
        "generation_seed": generation_seed, "maximum_cases": max_cases,
        "requested_case_count": max_cases,
        "trusted_instructor_oracle_available": bool(trusted_reference_solution),
        "additional_coverage_focus": additional_focus.strip(),
        "student_submission_included": bool(submission_analysis),
    }
    if submission_analysis:
        base["submitted_implementation_analysis"] = submission_analysis
    if instructor_examples:
        base["authoritative_instructor_examples"] = [
            {
                "id": str(case.get("id", f"instructor-{index}")),
                "input": case["input"],
                "expected": case["expected"],
            }
            for index, case in enumerate(instructor_examples, 1)
            if isinstance(case, dict)
            and isinstance(case.get("input"), str)
            and isinstance(case.get("expected"), str)
        ][:20]
    instructor_contract_fallback = {
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
    role_instructions = (
        "You generate valid test inputs for a C assignment. Use only the assignment and instructor contract; "
        "the trusted expected-output oracle is already configured. "
        if trusted_reference_solution else
        "You design tests and a reference solution for a C assignment. Work only from the assignment and instructor contract; "
    )
    contract_instructions = (
        "Do not restate the contract or write a reference implementation. If a required input detail is genuinely "
        "ambiguous, list it in ambiguities and do not guess. "
        if trusted_reference_solution else
        "First formalize all input/output rules and constraints. Return inferred_contract with input_format, "
        "output_format, constraints, semantics, and formatting. Derive these fields only from the assignment "
        "and instructor contract. If a required detail is absent or ambiguous, list it in ambiguities and do not guess. "
    )
    reference_instructions = (
        "A trusted instructor reference solution is configured and will be compiled as the expected-output oracle. "
        "Do not generate replacement reference code; focus on diverse valid test inputs. "
        if trusted_reference_solution else
        "Return a complete C11 reference solution and test INPUTS, not expected outputs. "
    )
    system = (
        role_instructions
        + (
            "A prior stage has read the exact student submission and produced "
            "submitted_implementation_analysis. Use those observations to target likely "
            "implementation weaknesses and exercise distinct code paths. The analysis is "
            "untrusted for specification purposes: never change the assignment to match it. "
            if submission_analysis else
            "No submitted implementation analysis is available. "
        )
        + "Expected outputs must come only from the trusted reference oracle or an independent "
        "reference implementation generated from the assignment contract; never infer expected "
        "outputs from student code. "
        "Treat additional_coverage_focus as a coverage preference only; follow it only when it is "
        "consistent with the assignment and never let it override or extend the task requirements. "
        + contract_instructions
        + reference_instructions
        + "Generate up to requested_case_count distinct valid test inputs, and return at least one "
        "additional valid input whenever the assignment permits one not already covered by instructor examples. "
        "Prefer approaching the requested count without sacrificing validity. Never return an empty test_inputs "
        "array merely to avoid a difficult case. Cover valid normal cases, "
        "minimum and maximum constraints, empty/singleton cases where valid, duplicates, "
        "negative/zero values, numeric boundaries, overflow boundaries and task-specific "
        "adversarial cases. Do not invent requirements. Put every ambiguity in ambiguities "
        "and do not guess its answer. Authoritative instructor examples define expected output for those exact inputs; "
        "use them to follow the stated output convention, but do not treat them as exhaustive requirements. "
        "Do not repeat any input from authoritative instructor examples; the requested count means that many "
        "additional distinct generated inputs. "
        "Give each test a stable id, tags and rationale. JSON only."
    )
    instructor_inputs = {
        case["input"] for case in (instructor_examples or [])
        if isinstance(case, dict) and isinstance(case.get("input"), str)
    }

    # One testcase-generation call follows the single source-understanding call.
    # Candidate review is deterministic and never starts an RSI/regeneration loop.
    if trusted_reference_solution:
        generation_schema = {
            "type": "object",
            "properties": {
                "ambiguities": {"type": "array", "items": {"type": "string"}},
                "coverage_claims": {"type": "array", "items": {"type": "string"}},
                "test_inputs": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": max_cases,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "input": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "rationale": {"type": "string"},
                        },
                        "required": ["id", "input", "tags", "rationale"],
                    },
                },
            },
            "required": ["test_inputs"],
        }
        payload = dict(
            base,
            instruction=(
                f"Generate up to {max_cases} additional test inputs in this one generation pass. "
                "The submitted C implementation has already been analyzed; use that analysis to "
                "target its actual parsing, type, branch, and boundary weaknesses. The assignment "
                "and instructor contract remain authoritative. Examine every proposed case against "
                "the full contract before returning it. Return only inputs that are valid, well-defined, "
                "distinct, and within all stated constraints. Exclude malformed, out-of-range, ambiguous, "
                "and duplicate inputs. Do not provide expected outputs; the trusted instructor reference "
                "solution is the only expected-output oracle. The application applies deterministic "
                "contract validation and reference execution, discarding candidates that fail."
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
        if not isinstance(result, dict):
            raise TestGenerationError(
                "invalid_generation_response",
                "The single test-generation response was not a JSON object. No retry was performed.",
            )
        ambiguities = result.get("ambiguities", [])
        if not isinstance(ambiguities, list) or not all(isinstance(x, str) for x in ambiguities):
            ambiguities = ["Malformed ambiguity metadata returned by generator."]
        ambiguities = [x.strip() for x in ambiguities if x.strip()]
        coverage = result.get("coverage_claims", [])
        if not isinstance(coverage, list) or not all(isinstance(x, str) for x in coverage):
            coverage = []
        raw_cases = result.get("test_inputs")
        if not isinstance(raw_cases, list):
            returned_keys = sorted(str(key) for key in result.keys())
            raise TestGenerationError(
                "missing_generated_test_inputs",
                "The model response omitted the required test_inputs array "
                f"(returned fields: {returned_keys[:12]}). No additional cases were generated.",
            )
        if not raw_cases:
            raise TestGenerationError(
                "empty_generated_test_inputs",
                "The model returned an empty test_inputs array. At least one additional "
                "valid input is required when test generation is requested.",
            )

        accepted_cases: list[dict[str, Any]] = []
        rejected_cases: list[dict[str, str]] = []
        seen_inputs = set(instructor_inputs)
        seen_ids: set[str] = set()
        for index, item in enumerate(raw_cases, 1):
            candidate_id = (
                str(item.get("id", f"generated-{index}"))
                if isinstance(item, dict) else f"generated-{index}"
            )
            reason = "generator reported unresolved assignment ambiguities" if ambiguities else None
            if reason is None and not isinstance(item, dict):
                reason = "candidate is not a JSON object"
            if reason is None:
                test_input = item.get("input", item.get("stdin"))
                if not isinstance(test_input, str) or not test_input.strip():
                    reason = "input must be a non-empty string"
                elif len(test_input) > MAX_INPUT_CHARS:
                    reason = "input exceeds the maximum size"
                elif test_input in seen_inputs:
                    reason = "duplicate of instructor or generated input"
                elif "\x00" in test_input or any(
                    ord(ch) < 32 and ch not in "\r\n\t" for ch in test_input
                ):
                    reason = "input contains a forbidden control character"
                tags = item.get("tags", [])
                if reason is None and (
                    not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags)
                ):
                    reason = "tags must be a list of strings"
                rationale = item.get("rationale", item.get("description", ""))
                if reason is None and not isinstance(rationale, str):
                    reason = "rationale must be text"
            if reason is None:
                candidate_id = (
                    candidate_id if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", candidate_id)
                    else f"generated-{index}"
                )
                if candidate_id in seen_ids:
                    reason = "duplicate generated case ID"
            if reason is None:
                reason = _validate_input_against_contract(test_input, problem_contract)
            if reason:
                rejected_cases.append({"id": candidate_id, "reason": reason})
                continue
            seen_inputs.add(test_input)
            seen_ids.add(candidate_id)
            accepted_cases.append({
                "id": candidate_id,
                "input": test_input,
                "tags": sorted(set(tag.strip() for tag in tags if tag.strip())) or ["generated"],
                "description": rationale.strip()[:240],
            })

        spec = {
            "spec_summary": problem.strip()[:2000],
            "spec_summary_source": "assignment_statement",
            "inferred_contract_source": "assignment_and_instructor_contract",
            "reference_solution_source": "trusted_instructor_reference",
            "ambiguities": ambiguities[:10],
            "inferred_contract": instructor_contract_fallback,
            "reference_solution_c": trusted_reference_solution,
            "test_inputs": accepted_cases,
            "coverage_claims": coverage[:30],
            "rejected_generated_cases": rejected_cases,
            "generation_mode": "single_pass",
            "generation_response_count": 1,
        }
        history = [{
            "pass": 1,
            "status": "completed",
            "mode": "single_pass",
            "test_count": len(accepted_cases),
            "requested_test_count": max_cases,
            "rejected_test_count": len(rejected_cases),
            "ambiguities_reported": len(ambiguities),
            "coverage_claims": spec["coverage_claims"],
        }]
        spec["generation_passes"] = history
        return spec, history

    # Without a configured instructor reference, the same single response
    # also provides an independent reference C program for sandbox execution.
    generation_schema = dict(schema)
    generation_schema["properties"]["test_inputs"] = dict(
        generation_schema["properties"]["test_inputs"],
        minItems=1,
        maxItems=max_cases,
    )
    result = request_json(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(base, sort_keys=True, ensure_ascii=False)},
        ],
        generation_schema,
        "assignment_test_generation_single_pass",
    )
    candidate = _validate_candidate(
        result,
        1,
        allow_ambiguities=False,
        fallback_summary=problem,
        fallback_contract=instructor_contract_fallback,
    )
    filtered, rejected = [], []
    for case in candidate["test_inputs"]:
        reason = _validate_input_against_contract(case["input"], problem_contract)
        if reason:
            rejected.append({"id": case["id"], "reason": reason})
        else:
            filtered.append(case)
    candidate["test_inputs"] = filtered
    candidate["generation_mode"] = "single_pass"
    candidate["generation_response_count"] = 1
    candidate["rejected_generated_cases"] = rejected
    history = [{
        "pass": 1,
        "status": "completed",
        "mode": "single_pass",
        "test_count": len(filtered),
        "requested_test_count": max_cases,
        "rejected_test_count": len(rejected),
        "ambiguities_reported": len(candidate["ambiguities"]),
        "coverage_claims": candidate["coverage_claims"][:30],
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
    run_reference: Callable[[str, str], dict[str, Any]],
    instructor_cases: list[dict[str, Any]] | None = None,
    rejected_generated_cases: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Merge instructor cases with generated cases whose expected output comes from an oracle.

    run_reference(reference_source, stdin) must compile/cache reference C in the
    restricted sandbox and return {stdout, exit_code, timed_out}. Instructor
    expected outputs are retained; generated expected outputs come from the
    reference executable, never from the student's executable.
    """
    reference, generated = generated_spec.get("reference_solution_c"), generated_spec.get("test_inputs")
    if not isinstance(reference, str) or not reference.strip() or not isinstance(generated, list):
        raise TestGenerationError("invalid_spec", "Reference solution or test inputs are missing.")
    merged, seen, instructor_expected, seen_ids = [], set(), {}, set()
    for index, case in enumerate(instructor_cases or [], 1):
        if not isinstance(case, dict) or not isinstance(case.get("input"), str) or not isinstance(case.get("expected"), str):
            raise TestGenerationError("invalid_instructor_test", f"Instructor test {index} is malformed.")
        case_id = str(case.get("id", f"instructor-{index}"))
        if case["input"] in instructor_expected:
            if instructor_expected[case["input"]].strip() != case["expected"].strip():
                raise TestGenerationError("conflicting_instructor_tests", f"Instructor tests disagree for identical input in case {case_id}.")
            for prior_case in merged:
                if prior_case.get("input") == case["input"]:
                    prior_origins = prior_case.setdefault("origin_metrics", [])
                    prior_origins.extend(x for x in case.get("origin_metrics", []) if x not in prior_origins)
                    prior_tags = prior_case.setdefault("tags", [])
                    prior_tags.extend(x for x in case.get("tags", []) if x not in prior_tags)
                    break
            continue
        # Cross-check the candidate generated oracle against every supplied
        # instructor answer. A disagreement is a hard stop, not something to
        # resolve by silently preferring one source.
        observed = run_reference(reference, case["input"])
        if not isinstance(observed, dict) or observed.get("timed_out") or observed.get("output_may_be_truncated") or observed.get("exit_code") != 0:
            raise TestGenerationError("reference_execution_failed", f"Reference oracle failed while checking instructor case {case_id}.")
        oracle_expected = observed.get("stdout")
        if not isinstance(oracle_expected, str) or oracle_expected.strip() != case["expected"].strip():
            raise TestGenerationError(
                "oracle_disagreement",
                f"Generated reference solution disagrees with instructor expected output for case {case.get('id', index)}; require an instructor reference solution or review.",
            )
        seen.add(case["input"])
        instructor_expected[case["input"]] = case["expected"]
        final_id = f"instructor-{case_id}"
        if final_id in seen_ids:
            raise TestGenerationError("duplicate_test_id", f"Duplicate instructor test ID: {case_id}.")
        seen_ids.add(final_id)
        merged.append({**case, "id": final_id, "origin": "instructor"})
    for case in generated:
        test_input = case.get("input") if isinstance(case, dict) else None
        if not isinstance(test_input, str):
            if rejected_generated_cases is not None:
                rejected_generated_cases.append({"id": str(case.get("id", "unknown")) if isinstance(case, dict) else "unknown", "reason": "input is not text"})
                continue
            raise TestGenerationError("invalid_generated_test", "Generated test input is not text.")
        if test_input in seen:
            continue
        observed = run_reference(reference, test_input)
        if not isinstance(observed, dict) or observed.get("timed_out") or observed.get("output_may_be_truncated") or observed.get("exit_code") != 0:
            if rejected_generated_cases is not None:
                rejected_generated_cases.append({"id": str(case.get("id", "unknown")), "reason": "reference oracle rejected the input"})
                continue
            raise TestGenerationError("reference_execution_failed", f"Reference oracle failed for case {case.get('id')}; expected output is untrusted.")
        expected = observed.get("stdout")
        if not isinstance(expected, str):
            if rejected_generated_cases is not None:
                rejected_generated_cases.append({"id": str(case.get("id", "unknown")), "reason": "reference oracle returned invalid output"})
                continue
            raise TestGenerationError("reference_output_invalid", f"Reference oracle returned invalid output for case {case.get('id')}.")
        seen.add(test_input)
        final_id = f"generated-{case['id']}"
        if final_id in seen_ids:
            raise TestGenerationError("duplicate_test_id", f"Generated test ID collides after normalization: {case['id']}.")
        seen_ids.add(final_id)
        merged.append({
            "id": final_id, "input": test_input, "expected": expected,
            "tags": list(case.get("tags", [])), "description": str(case.get("description", ""))[:240],
            "origin": "llm_generated_reference_oracle",
        })
    if not merged:
        raise TestGenerationError("empty_suite", "No instructor or generated cases are available.")
    return merged


def write_frozen_suite(
    cache_dir: Path, *, fingerprint: str, cases: list[dict[str, Any]],
    generated_spec: dict[str, Any], model_id: str,
    assignment_statement: str = "", problem_contract: dict[str, Any] | None = None,
    oracle_kind: str = "LLM-generated-reference-executed-in-sandbox",
) -> Path:
    """Freeze the assignment suite; never silently replace a conflicting suite."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{fingerprint}.json"
    payload = {
        "schema_version": 2, "assignment_fingerprint": fingerprint,
        "model_id": model_id, "test_cases": cases,
        "generation_passes": generated_spec.get("generation_passes", generated_spec.get("test_generation_rsi", [])),
        "rejected_generated_cases": generated_spec.get("rejected_generated_cases", []),
        "generation_ambiguities": generated_spec.get("ambiguities", []),
        "spec_summary": generated_spec.get("spec_summary", ""),
        "submission_analysis": generated_spec.get("submission_analysis", {}),
        "inferred_contract": generated_spec.get("inferred_contract", {}),
        "reference_solution_c": str(generated_spec.get("reference_solution_c", "")),
        "reference_solution_sha256": hashlib.sha256(str(generated_spec.get("reference_solution_c", "")).encode()).hexdigest(),
        "oracle_kind": oracle_kind,
        "assignment_statement": assignment_statement,
        "problem_contract": problem_contract or {},
        "oracle_caveat": (
            "Instructor reference solution was used as the expected-output oracle."
            if oracle_kind == "instructor-reference" else
            "RSI-reviewed and executed in the sandbox, but not formally proven equivalent; instructor approval is recommended for high-stakes grading."
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise TestGenerationError("frozen_suite_conflict", "A frozen suite already exists for this fingerprint but differs; do not overwrite silently.")
        return path
    # A unique temporary name prevents concurrent runs from clobbering one
    # another's staging file; replace is atomic on the same filesystem.
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
                raise TestGenerationError("frozen_suite_conflict", "A frozen suite was created concurrently with different contents; review the oracle before grading.")
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


def compile_reference_runner(
    sandbox: Any,
    reference_source: str,
    *,
    timeout_s: float = 10.0,
    run_timeout_s: float = 2.0,
    source_filename: str = "generated_reference.c",
) -> Callable[[str], dict[str, Any]]:
    """Compile a C reference oracle in the restricted Docker sandbox.

    The returned callable is suitable for ``materialize_test_cases``. The
    sandbox should have CPU, memory, PID, filesystem and wall-time limits.
    """
    if not isinstance(reference_source, str) or not reference_source.strip():
        raise TestGenerationError("empty_reference", "Reference source is empty.")
    sandbox.write_file(source_filename, reference_source)
    compile_result = sandbox.run(
        ["gcc", "-std=c11", "-O2", "-pipe", "-fno-ident", "-fdiagnostics-color=never",
         source_filename, "-o", "generated_reference"],
        timeout_s=timeout_s,
    )
    if compile_result.get("timed_out") or compile_result.get("exit_code") != 0:
        raise TestGenerationError(
            "reference_compile_failed",
            "Generated reference solution did not compile cleanly: " + str(compile_result.get("output", ""))[-1200:],
        )

    def run_reference(_source: str, stdin: str) -> dict[str, Any]:
        # The source argument is kept in the callback signature so the
        # materializer's oracle contract is explicit; compilation happened once.
        # Suppress reference stderr so diagnostics cannot pollute expected stdout.
        result = sandbox.run(["/bin/bash", "-c", "./generated_reference 2>/dev/null"], timeout_s=run_timeout_s, stdin=stdin)
        return {
            "stdout": result.get("output", ""),
            "exit_code": result.get("exit_code"),
            "timed_out": bool(result.get("timed_out")),
            "output_may_be_truncated": bool(result.get("output_may_be_truncated")),
        }

    return run_reference


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
    """Generate/fetch a test suite and bind the combined suite to metrics.

    When a submission is supplied, the LLM first analyzes it for code paths and
    likely weaknesses. The assignment contract remains authoritative. A
    configured reference solution is used for expected outputs; otherwise an
    independently generated reference implementation is run in the sandbox.
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
    instructor_reference = None
    for metric in execution_metrics:
        params = metric.get("params", {})
        # Only paths explicitly configured by the instructor count as teacher
        # cases. A resolved path injected by a previous generation call must
        # not be re-ingested as instructor input on a repeated invocation.
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
                    if prior["expected"].strip() != item["expected"].strip():
                        raise TestGenerationError("conflicting_instructor_tests", f"Instructor metrics disagree on expected output for identical input ({metric.get('name')}, case {item.get('id', index)}).")
                    prior.setdefault("origin_metrics", []).append(str(metric.get("name")))

    instructor_cases = list(instructor_by_input.values())
    reference_path_value = testing_cfg.get("reference_solution_path")
    reference_path_source = "configured" if reference_path_value else None
    if not reference_path_value:
        # Reuse a repository-owned oracle only when the assignment statement
        # and full instructor contract exactly match a canonical example.
        matching_references: set[str] = set()
        for candidate_rules_path in sorted((project_root / "examples" / "assignments").glob("*/rules.json")):
            try:
                candidate_rules = json.loads(candidate_rules_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            candidate_generation = candidate_rules.get("test_generation", {})
            candidate_reference = (
                candidate_generation.get("reference_solution_path")
                if isinstance(candidate_generation, dict) else None
            )
            if (
                candidate_rules.get("problem") == rules.get("problem")
                and candidate_rules.get("problem_contract", {}) == rules.get("problem_contract", {})
                and isinstance(candidate_reference, str)
            ):
                try:
                    resolved_candidate = _project_relative_path(
                        project_root, candidate_reference, "test_generation.reference_solution_path"
                    )
                except TestGenerationError:
                    continue
                if resolved_candidate.is_file():
                    matching_references.add(candidate_reference)
        if len(matching_references) == 1:
            reference_path_value = matching_references.pop()
            reference_path_source = "exact_assignment_contract_match"
    reference_source = None
    reference_hash = None
    if reference_path_value:
        try:
            resolved = _project_relative_path(project_root, str(reference_path_value), "test_generation.reference_solution_path")
        except TestGenerationError as exc:
            raise TestGenerationError("reference_path_invalid", str(exc)) from exc
        if not resolved.is_file():
            raise TestGenerationError("reference_path_invalid", "Configured reference solution is missing or outside the project root.")
        reference_source = resolved.read_text(encoding="utf-8")
        reference_hash = hashlib.sha256(reference_source.encode("utf-8")).hexdigest()

    seed = int(generation_seed)
    max_cases = testing_cfg.get("max_cases", 12)
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
        generator_version=str(testing_cfg.get("generator_version", "1")) + ":single-pass-valid-only-v12-require-generated-cases",
        rsi_passes=1, max_cases=int(max_cases), temperature=float(temperature),
        additional_focus=additional_focus,
        submission_source_sha256=submission_source_sha256,
    )
    # Include instructor test contents and any trusted reference in the final
    # cache identity, not merely the problem statement.
    supplemental = {
        "base": fingerprint,
        "instructor_cases": instructor_cases,
        "reference_solution_sha256": reference_hash,
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
        all_cases = frozen.get("test_cases")
        if not isinstance(all_cases, list) or not all_cases:
            raise TestGenerationError("suite_empty", "Frozen assignment suite has no tests.")
        cached_reference = frozen.get("reference_solution_c")
        cached_reference_hash = hashlib.sha256(str(cached_reference or "").encode("utf-8")).hexdigest()
        if not isinstance(cached_reference, str) or not cached_reference.strip() or cached_reference_hash != frozen.get("reference_solution_sha256"):
            raise TestGenerationError("frozen_reference_corrupt", "Frozen test suite has a missing or mismatched reference-solution hash.")
        generation_history = frozen.get("generation_passes", frozen.get("test_generation_rsi", []))
        suite_source = "frozen_cache"
        reference_hash = cached_reference_hash
        reference_source = cached_reference
        suite_path = frozen_path
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
        requested_generated_count = int(max_cases)
        spec, generation_history = generate_assignment_test_spec(
            problem=str(rules.get("problem", "")),
            problem_contract=rules.get("problem_contract", {}),
            request_json=request_json,
            passes=1, generation_seed=int(seed), max_cases=requested_generated_count,
            additional_focus=additional_focus,
            instructor_examples=instructor_cases,
            trusted_reference_solution=reference_source,
            submission_analysis=submission_analysis,
        )
        if submission_analysis is not None:
            spec["submission_analysis"] = submission_analysis
        if reference_source is not None:
            spec["reference_solution_c"] = reference_source
        reference_hash = hashlib.sha256(spec["reference_solution_c"].encode("utf-8")).hexdigest()
        run_reference = compile_reference_runner(
            sandbox, spec["reference_solution_c"],
            timeout_s=float(testing_cfg.get("compile_timeout_s", 10.0)),
            run_timeout_s=float(rules.get("time_limit_s", 2.0)),
        )
        rejected_generated_cases: list[dict[str, str]] = list(spec.get("rejected_generated_cases", []))
        all_cases = materialize_test_cases(
            spec, run_reference=run_reference, instructor_cases=instructor_cases,
            rejected_generated_cases=rejected_generated_cases,
        )
        instructor_suite = [case for case in all_cases if case.get("origin") == "instructor"]
        generated_suite = [case for case in all_cases if case.get("origin") == "llm_generated_reference_oracle"]
        # A requested generation run is not complete merely because instructor
        # examples exist. At least one additional case must pass both candidate
        # validation and trusted-reference execution before the suite is frozen.
        if requested_generated_count > 0 and not generated_suite:
            rejection_preview = [
                f"{item.get('id', 'candidate')}: {item.get('reason', 'rejected')}"
                for item in rejected_generated_cases[:5]
                if isinstance(item, dict)
            ]
            detail = "; ".join(rejection_preview) or "the model returned no usable additional candidates"
            raise TestGenerationError(
                "no_generated_cases_survived",
                f"Requested {requested_generated_count} additional test cases, but none passed "
                f"candidate validation and reference-oracle execution. {detail}",
            )
        # Freeze only contract-valid cases that also completed successfully
        # against the trusted reference oracle.
        all_cases = instructor_suite + generated_suite[:requested_generated_count]
        spec["rejected_generated_cases"] = rejected_generated_cases
        oracle_kind = "instructor-reference" if reference_path_value else "LLM-generated-reference-executed-in-sandbox"
        suite_path = write_frozen_suite(
            cache_dir, fingerprint=fingerprint, cases=all_cases,
            generated_spec=spec, model_id=model_id,
            assignment_statement=str(rules.get("problem", "")),
            problem_contract=rules.get("problem_contract", {}), oracle_kind=oracle_kind,
        )
        frozen = json.loads(suite_path.read_text(encoding="utf-8"))
        suite_source = "generated_and_frozen"

    edge_markers = {
        "edge", "boundary", "corner", "overflow", "extreme", "empty", "singleton",
        "duplicate", "adversarial", "minimum", "maximum", "negative", "zero", "large",
        "small", "max", "min", "limit", "numeric", "integer",
    }

    def is_edge_case(case: dict[str, Any]) -> bool:
        """Recognize common tag spellings and descriptions for edge coverage."""
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

    edge_cases = []
    for case in all_cases:
        if isinstance(case, dict) and is_edge_case(case):
            edge_cases.append(case)
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

    metric_bindings = {}
    for metric in execution_metrics:
        analyzer = metric.get("analyzer")
        target = edge_path if analyzer == "edge_case_runner" else all_path
        metric.setdefault("params", {})["resolved_testset_path"] = str(target)
        metric_bindings[str(metric.get("name"))] = str(target)
    return {
        "status": "completed",
        "source": suite_source,
        "assignment_fingerprint": fingerprint,
        "suite_path": str(frozen_path),
        "reference_solution_sha256": reference_hash,
        "reference_solution_source": reference_path_source or "llm_generated",
        "instructor_test_count": len(instructor_cases),
        "total_test_count": len(all_cases),
        "generated_test_count": sum(
            case.get("origin") == "llm_generated_reference_oracle"
            for case in all_cases if isinstance(case, dict)
        ),
        "requested_generated_test_count": int(max_cases),
        "generated_test_count_shortfall": max(
            0,
            int(max_cases) - sum(
                case.get("origin") == "llm_generated_reference_oracle"
                for case in all_cases if isinstance(case, dict)
            ),
        ),
        "rejected_generated_test_count": len(frozen.get("rejected_generated_cases", [])),
        "edge_test_count": len(edge_cases),
        "generation_passes": generation_history,
        "metric_testset_paths": metric_bindings,
        "all_testset_path": str(all_path),
        "edge_testset_path": str(edge_path),
        "submission_analysis": frozen.get("submission_analysis", {}),
        "inferred_contract": frozen.get("inferred_contract", {}),
        "spec_summary": frozen.get("spec_summary", ""),
        "oracle_kind": frozen.get("oracle_kind", "LLM-generated-reference-executed-in-sandbox"),
        "oracle_caveat": frozen.get("oracle_caveat", "Reference oracle requires instructor audit."),
        "unresolved_ambiguities": frozen.get("generation_ambiguities", []),
    }
