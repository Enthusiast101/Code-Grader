"""Source-aware test generation with a sequential RSI review.

The LLM first studies the submitted implementation to identify its input paths,
assumptions, and likely weak points. It then generates contract-compliant test
inputs. Expected outputs are never taken from the submission: they are obtained
by running a trusted or independently generated reference solution in the
restricted sandbox. The resulting suite is frozen by a source-aware fingerprint.
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
MAX_TEST_INPUTS_PER_LLM_BATCH = 4
MAX_TEST_INPUT_BATCH_ATTEMPTS = 3


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
        raise TestGenerationError("invalid_response", f"RSI pass {pass_number} did not return an object.")
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
            raise TestGenerationError("missing_spec", f"RSI pass {pass_number} omitted the formal task summary.")
    if not isinstance(ambiguity, list) or not all(isinstance(x, str) for x in ambiguity):
        raise TestGenerationError("invalid_ambiguities", f"RSI pass {pass_number} returned malformed ambiguity details.")
    if ambiguity and not allow_ambiguities:
        raise TestGenerationError("ambiguous_assignment", "Unresolved assignment ambiguities: " + "; ".join(ambiguity[:5]))
    inferred = value.get("inferred_contract", value.get("io_contract"))
    fallback_contract = fallback_contract if isinstance(fallback_contract, dict) else {}
    contract_source = "model"
    if not isinstance(inferred, dict):
        inferred = dict(fallback_contract)
        contract_source = "assignment_and_instructor_contract"
        if not inferred:
            raise TestGenerationError("missing_inferred_contract", f"RSI pass {pass_number} omitted the assignment-derived I/O contract.")
    else:
        inferred = dict(inferred)
        for field in ("input_format", "output_format"):
            if not isinstance(inferred.get(field), str) or len(inferred[field].strip()) < 8:
                fallback_value = fallback_contract.get(field)
                if not isinstance(fallback_value, str) or len(fallback_value.strip()) < 8:
                    raise TestGenerationError("invalid_inferred_contract", f"RSI pass {pass_number} must infer a meaningful {field}.")
                inferred[field] = fallback_value
                contract_source = "assignment_and_instructor_contract"
        for field in ("constraints", "semantics", "formatting"):
            entries = inferred.get(field)
            if not isinstance(entries, list) or not all(isinstance(item, str) and item.strip() for item in entries):
                fallback_entries = fallback_contract.get(field)
                if not isinstance(fallback_entries, list):
                    raise TestGenerationError("invalid_inferred_contract", f"RSI pass {pass_number} returned invalid inferred_contract.{field}.")
                inferred[field] = fallback_entries
                contract_source = "assignment_and_instructor_contract"
        if not inferred["semantics"]:
            inferred["semantics"] = fallback_contract.get("semantics", [])
            contract_source = "assignment_and_instructor_contract"
            if not inferred["semantics"]:
                raise TestGenerationError("invalid_inferred_contract", f"RSI pass {pass_number} must identify at least one task semantic.")
    if not isinstance(reference, str) or not reference.strip() or len(reference) > MAX_REFERENCE_SOURCE_CHARS:
        raise TestGenerationError("missing_reference", f"RSI pass {pass_number} omitted a bounded C reference solution.")
    if not isinstance(cases, list) or not cases or len(cases) > MAX_GENERATED_CASES:
        raise TestGenerationError("invalid_test_count", f"RSI pass {pass_number} must return 1..{MAX_GENERATED_CASES} test inputs.")

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
    """Have the LLM understand the submitted C implementation before test design.

    This stage identifies code paths and likely weaknesses only. The assignment
    remains authoritative, and expected outputs are always computed by the
    reference oracle in the sandbox, never from the student submission.
    """
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
            "implementation_summary": {"type": "string"},
            "input_parsing_and_validation": {"type": "array", "items": {"type": "string"}},
            "control_flow_and_data_types": {"type": "array", "items": {"type": "string"}},
            "likely_weak_points": {"type": "array", "items": {"type": "string"}},
            "recommended_test_dimensions": {"type": "array", "items": {"type": "string"}},
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
                "The assignment and instructor contract are authoritative: do not reinterpret "
                "requirements to match the student's implementation. Do not execute the source, "
                "produce expected outputs, or treat the source's behavior as correct. Return JSON only."
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
                        "Understand this exact implementation first. Identify likely edge cases "
                        "and missing checks that a separate test-generation stage should probe. "
                        "Do not supply test inputs or expected outputs in this stage."
                    ),
                },
                sort_keys=True,
                ensure_ascii=False,
            ),
        },
    ]
    last_error = "The model did not return a valid implementation analysis."
    for attempt in range(1, 4):
        result = request_json(messages, schema, f"assignment_submission_understanding_attempt_{attempt}")
        if isinstance(result, dict):
            summary = result.get("implementation_summary")
            list_fields = (
                "input_parsing_and_validation", "control_flow_and_data_types",
                "likely_weak_points", "recommended_test_dimensions",
            )
            if isinstance(summary, str) and summary.strip() and all(
                isinstance(result.get(field), list)
                and all(isinstance(item, str) for item in result[field])
                for field in list_fields
            ):
                return {
                    "implementation_summary": summary.strip()[:4000],
                    **{
                        field: [item.strip()[:500] for item in result[field] if item.strip()][:20]
                        for field in list_fields
                    },
                }
            last_error = "The model's implementation analysis omitted required fields or returned malformed lists."
        else:
            last_error = "The model did not return an object for implementation analysis."
        if attempt < 3:
            messages = [
                *messages,
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False) if result is not None else "null"},
                {"role": "user", "content": f"Repair the response. Return all required JSON fields. {last_error}"},
            ]
    raise TestGenerationError("submission_understanding_failed", last_error)


def generate_assignment_test_spec(
    *, problem: str, problem_contract: dict[str, Any],
    request_json: Callable[[list[dict[str, str]], dict[str, Any], str], dict[str, Any] | None],
    passes: int = 3, generation_seed: int = 42, max_cases: int = 60,
    additional_focus: str = "",
    instructor_examples: list[dict[str, Any]] | None = None,
    trusted_reference_solution: str | None = None,
    submission_analysis: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate a reference solution and test inputs, then sequentially RSI-recheck.

    request_json(messages, schema, stage_name) performs one structured model
    call. Without a trusted oracle, each RSI pass returns one complete
    replacement object. With a trusted instructor oracle, each pass is assembled
    from small batches so a large suite cannot be lost to one truncated response.
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
        + "Generate exactly requested_case_count "
        "distinct valid test inputs unless the assignment has fewer valid inputs. Cover valid normal cases, "
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

    # When an instructor reference solution is available, the model only has
    # to produce stdin cases. Asking a small local model for 12-80 complete
    # cases in one JSON response is prone to truncation / invalid structured
    # output; the grader then silently falls back to the instructor cases.
    # Generate and RSI-review the suite in small, schema-constrained batches
    # instead. Every pass still replaces the complete candidate suite, while
    # each individual response contains at most four inputs.
    if trusted_reference_solution:
        accepted: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        prior: dict[str, Any] | None = None

        batch_schema = {
            "type": "object",
            "properties": {
                "ambiguities": {"type": "array", "items": {"type": "string"}},
                "coverage_claims": {"type": "array", "items": {"type": "string"}},
                "test_inputs": {
                    "type": "array",
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

        for pass_number in range(1, passes + 1):
            if pass_number == 1:
                pass_instruction = (
                    "Generate a complete diverse set of additional valid test inputs for the assignment. "
                    "The trusted instructor reference solution supplies expected outputs."
                )
            else:
                pass_instruction = (
                    f"RSI pass {pass_number}: independently review the preceding complete candidate suite "
                    "for invalid inputs, duplicate cases, missing boundary coverage, and unsupported assumptions. "
                    "Replace it with a complete improved suite. Generate the replacement suite in chunks; "
                    "this request is for only the next chunk. Preserve the stated assignment contract."
                )

            pass_cases: list[dict[str, Any]] = []
            pass_ambiguities: list[str] = []
            pass_coverage: list[str] = []
            batch_number = 0

            while len(pass_cases) < max_cases:
                batch_number += 1
                batch_size = min(MAX_TEST_INPUTS_PER_LLM_BATCH, max_cases - len(pass_cases))
                accepted_batch = False
                last_error = "The model did not return enough distinct valid inputs."

                for attempt in range(1, MAX_TEST_INPUT_BATCH_ATTEMPTS + 1):
                    payload = dict(
                        base,
                        maximum_cases=max_cases,
                        requested_case_count=batch_size,
                        instruction=(
                            f"{pass_instruction} Return exactly {batch_size} new, distinct valid test inputs "
                            "for this batch. Do not include expected outputs or reference code. Do not repeat "
                            "instructor examples or any input in excluded_inputs. If a case's expected behavior "
                            "is ambiguous under the assignment contract, do not generate it. "
                            f"Batch {batch_number}; attempt {attempt}."
                        ),
                        excluded_inputs=[case["input"] for case in pass_cases[-20:]],
                    )
                    if prior is not None:
                        preceding = dict(prior)
                        preceding.pop("reference_solution_c", None)
                        payload["preceding_complete_response"] = preceding

                    result = request_json(
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": json.dumps(payload, sort_keys=True, ensure_ascii=False)},
                        ],
                        {
                            **batch_schema,
                            "properties": {
                                **batch_schema["properties"],
                                "test_inputs": {
                                    **batch_schema["properties"]["test_inputs"],
                                    "minItems": batch_size,
                                    "maxItems": batch_size,
                                },
                            },
                        },
                        f"assignment_test_generation_rsi_{pass_number}_batch_{batch_number}_attempt_{attempt}",
                    )
                    try:
                        candidate = _validate_candidate(
                            result,
                            pass_number,
                            allow_ambiguities=True,
                            fallback_summary=problem,
                            fallback_contract=instructor_contract_fallback,
                            fallback_reference_solution=trusted_reference_solution,
                        )
                    except TestGenerationError as exc:
                        last_error = exc.message
                        continue

                    existing_inputs = {case["input"] for case in pass_cases}
                    new_cases = [
                        case for case in candidate["test_inputs"]
                        if case["input"] not in instructor_inputs
                        and case["input"] not in existing_inputs
                    ]
                    if not new_cases:
                        last_error = "The batch contained only instructor duplicates or previously generated inputs."
                        continue

                    for case in new_cases:
                        # Generated IDs from separate calls often restart at
                        # generated-1; namespace them by pass and batch.
                        case["id"] = f"p{pass_number}b{batch_number}-{case['id']}"[:64]
                        pass_cases.append(case)
                        if len(pass_cases) == max_cases:
                            break
                    pass_ambiguities.extend(candidate.get("ambiguities", []))
                    pass_coverage.extend(candidate.get("coverage_claims", []))
                    accepted_batch = True
                    break

                if not accepted_batch:
                    raise TestGenerationError(
                        "test_input_batch_failed",
                        f"Could not generate batch {batch_number} in RSI pass {pass_number}: {last_error}",
                    )

            accepted = {
                "spec_summary": problem.strip()[:2_000],
                "spec_summary_source": "assignment_statement",
                "inferred_contract_source": "assignment_and_instructor_contract",
                "reference_solution_source": "trusted_instructor_reference",
                "ambiguities": list(dict.fromkeys(pass_ambiguities))[:10],
                "inferred_contract": {
                    "input_format": instructor_contract_fallback["input_format"],
                    "output_format": instructor_contract_fallback["output_format"],
                    "constraints": instructor_contract_fallback["constraints"],
                    "semantics": instructor_contract_fallback["semantics"],
                    "formatting": instructor_contract_fallback["formatting"],
                },
                "reference_solution_c": trusted_reference_solution,
                "test_inputs": pass_cases,
                "coverage_claims": list(dict.fromkeys(pass_coverage))[:30],
                "case_count_shortfall": max(0, max_cases - len(pass_cases)),
            }
            history.append({
                "pass": pass_number,
                "status": "completed",
                "test_count": len(pass_cases),
                "requested_test_count": max_cases,
                "test_count_shortfall": accepted["case_count_shortfall"],
                "batch_count": batch_number,
                "ambiguities_reported": len(accepted["ambiguities"]),
                "reference_source_sha256": hashlib.sha256(trusted_reference_solution.encode()).hexdigest(),
                "coverage_claims": accepted["coverage_claims"],
            })
            prior = accepted

        if accepted is None:
            raise TestGenerationError("no_generation", "No valid generated test suite was produced.")
        if accepted["case_count_shortfall"]:
            raise TestGenerationError(
                "generated_test_count_shortfall",
                f"The LLM produced {len(accepted['test_inputs'])} distinct additional cases; {max_cases} were requested.",
            )
        accepted["test_generation_rsi"] = history
        return accepted, history

    accepted, prior, history = None, None, []
    for index in range(1, passes + 1):
        if index == 1 and trusted_reference_solution:
            instruction = (
                "PASS 1: generate the requested number of diverse valid test inputs for the stated contract. "
                "The trusted instructor oracle supplies all expected outputs."
            )
        elif index == 1:
            instruction = (
                "PASS 1: formalize the contract, independently derive a correct reference C11 "
                "solution, and generate broad test inputs."
            )
        elif trusted_reference_solution:
            instruction = (
                f"PASS {index}: recheck the complete previous test-input set for validity, coverage, "
                "duplicates, and unsupported assumptions. The trusted instructor oracle supplies "
                "expected outputs; do not generate or modify a reference solution. Resolve details "
                "only from the assignment, contract, or instructor examples. Return a complete "
                "replacement test-input set."
            )
        else:
            instruction = (
                f"PASS {index}: RSI recheck the complete previous specification, C solution and test set. "
                "Actively find misread requirements, omitted boundary cases, undefined behavior, parsing "
                "errors, reference implementation bugs, unsupported assumptions, duplicate/invalid tests, "
                "and mismatched expected semantics. Resolve each prior ambiguity only from the assignment, "
                "contract, or authoritative instructor examples. If a detail remains unresolved, avoid "
                "cases whose expected behavior depends on it and keep the ambiguity explicit. Return a "
                "complete corrected replacement object, not a patch."
            )
        if prior is not None and prior.get("case_count_shortfall", 0):
            instruction += (
                f" The previous response supplied only {len(prior['test_inputs'])} distinct cases; "
                f"add valid, non-duplicate cases to reach the requested count of {max_cases}."
            )
        payload = dict(base, instruction=instruction)
        if prior is not None:
            preceding = dict(prior)
            if trusted_reference_solution:
                preceding.pop("reference_solution_c", None)
            payload["preceding_complete_response"] = preceding
        result = request_json(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, sort_keys=True, ensure_ascii=False)}],
            schema, f"assignment_test_generation_rsi_{index}",
        )
        accepted = _validate_candidate(
            result,
            index,
            allow_ambiguities=index < passes or bool(trusted_reference_solution),
            fallback_summary=problem,
            fallback_contract=instructor_contract_fallback,
            fallback_reference_solution=trusted_reference_solution or "",
        )
        # Instructor cases are already retained in the final suite. Exclude
        # them from the generated target so materialization cannot silently
        # reduce the number of additional cases requested by the UI.
        accepted["test_inputs"] = [
            case for case in accepted["test_inputs"]
            if case["input"] not in instructor_inputs
        ]
        accepted["case_count_shortfall"] = max(0, max_cases - len(accepted["test_inputs"]))
        history.append({
            "pass": index, "status": "completed", "test_count": len(accepted["test_inputs"]),
            "ambiguities_reported": len(accepted["ambiguities"]),
            "requested_test_count": max_cases,
            "test_count_shortfall": accepted["case_count_shortfall"],
            "reference_source_sha256": hashlib.sha256(accepted["reference_solution_c"].encode()).hexdigest(),
            "coverage_claims": accepted["coverage_claims"],
        })
        prior = accepted
    if accepted is None:
        raise TestGenerationError("no_generation", "No valid generated test suite was produced.")
    if accepted["case_count_shortfall"]:
        raise TestGenerationError(
            "generated_test_count_shortfall",
            f"The LLM produced {len(accepted['test_inputs'])} distinct additional cases; {max_cases} were requested.",
        )
    accepted["test_generation_rsi"] = history
    return accepted, history


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
        "test_generation_rsi": generated_spec.get("test_generation_rsi", []),
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
    passes = testing_cfg.get("rsi_passes", 3)
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
        generator_version=str(testing_cfg.get("generator_version", "1")) + ":source-aware-contract-v10",
        rsi_passes=int(passes), max_cases=int(max_cases), temperature=float(temperature),
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
        generation_history = frozen.get("test_generation_rsi", [])
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
        replacement_reserve = min(
            MAX_GENERATED_CASES - requested_generated_count,
            max(3, math.ceil(requested_generated_count * 0.25)),
        )
        candidate_generated_count = requested_generated_count + replacement_reserve
        spec, generation_history = generate_assignment_test_spec(
            problem=str(rules.get("problem", "")),
            problem_contract=rules.get("problem_contract", {}),
            request_json=request_json,
            passes=int(passes), generation_seed=int(seed), max_cases=candidate_generated_count,
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
        rejected_generated_cases: list[dict[str, str]] = []
        all_cases = materialize_test_cases(
            spec, run_reference=run_reference, instructor_cases=instructor_cases,
            rejected_generated_cases=rejected_generated_cases,
        )
        instructor_suite = [case for case in all_cases if case.get("origin") == "instructor"]
        generated_suite = [case for case in all_cases if case.get("origin") == "llm_generated_reference_oracle"]
        if len(generated_suite) < requested_generated_count:
            rejected_ids = ", ".join(item["id"] for item in rejected_generated_cases[:5])
            detail = f" Rejected candidates: {rejected_ids}." if rejected_ids else ""
            raise TestGenerationError(
                "generated_test_count_shortfall",
                f"Only {len(generated_suite)} valid generated cases remained after oracle checks; "
                f"{requested_generated_count} were requested.{detail}",
            )
        # The reserve is only for replacing malformed inputs. Keep the final
        # grading suite at the exact count selected in the UI.
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
        "rsi_passes": generation_history,
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
