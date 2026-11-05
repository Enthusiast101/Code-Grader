"""Assignment-driven test generation with independent expected-output verification.

A source-understanding call may identify useful test dimensions, but it never
determines correctness. A generation call proposes exact stdin/expected-stdout
pairs; a second, independent LLM call solves the inputs from the assignment
specification. One bounded refill request can replace malformed, invalid, or
unverified candidates. No generated C reference program is compiled or run.
The independently verified suite is frozen by an assignment/source fingerprint.
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
# Limits on what the generator may write. The schema is enforced while the
# model decodes, so a string must close at its limit: a small model that
# starts repeating digits (e.g. a 30x30 grid) cannot run past the token budget.
GENERATED_INPUT_MAX_CHARS = 1_200
GENERATED_OUTPUT_MAX_CHARS = 600
CONTRACT_ITEM_MAX_CHARS = 160
MAX_SUBMISSION_SOURCE_BYTES = 100_000


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


MAX_HINT_FIELD_CHARS = 400
# A small model copies the examples it is shown, so only a few are sent as
# format hints; duplicate detection still uses every instructor case.
MAX_HINT_EXAMPLES = 3


def _shorten_for_prompt(text: str, limit: int) -> str:
    """Keep the head and tail of a long value with an explicit omission marker."""
    if len(text) <= limit:
        return text
    head = text[: limit * 2 // 3]
    tail = text[-(limit // 4):]
    return f"{head}\n... [{len(text) - len(head) - len(tail)} characters omitted] ...\n{tail}"


def _instructor_hint_examples(instructor_examples: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Instructor cases as compact format hints for the model.

    Hints only illustrate input/output conventions. A performance case can be
    huge (one LIS runtime input is ~97k characters, ~98k tokens), which alone
    overflows the model context, so long fields are shortened and flagged.
    Duplicate detection elsewhere still compares the full, untruncated inputs.
    """
    hints: list[dict[str, Any]] = []
    for index, case in enumerate(instructor_examples or [], 1):
        if not (isinstance(case, dict) and isinstance(case.get("input"), str) and isinstance(case.get("expected"), str)):
            continue
        hint = {
            "id": str(case.get("id", f"instructor-{index}")),
            "input": _shorten_for_prompt(case["input"], MAX_HINT_FIELD_CHARS),
            "expected_output": _shorten_for_prompt(case["expected"], MAX_HINT_FIELD_CHARS),
        }
        if hint["input"] != case["input"] or hint["expected_output"] != case["expected"]:
            hint["shortened_for_prompt"] = True
        hints.append(hint)
        if len(hints) >= MAX_HINT_EXAMPLES:
            break
    return hints


_COUNT_THEN_VALUES = re.compile(
    r"\bread\s+(?:an?\s+integer\s+)?(?P<var>[A-Za-z])\b[^.]*?\bfollowed\s+by\s+(?P=var)\s+"
    r"(?:[\w-]+\s+){0,3}?(?:integers?|numbers?|values?)\b",
    re.IGNORECASE,
)


def _declared_count_mismatch(problem: str, test_input: str) -> str | None:
    """Reject "n, then n values" inputs whose value count differs from n.

    Applies only when the assignment states exactly that shape (for example
    "Read n (...), followed by n signed 32-bit integers"). A generated case that
    declares n=20000 but lists 100 values has no defined expected output, and
    both model stages were observed to accept one.
    """
    if not isinstance(problem, str) or not _COUNT_THEN_VALUES.search(problem):
        return None
    tokens = test_input.split()
    if not tokens:
        return None
    try:
        declared = int(tokens[0])
    except ValueError:
        return None
    supplied = len(tokens) - 1
    if declared >= 0 and supplied != declared:
        return f"input declares {declared} values but supplies {supplied}"
    return None


def _input_identity(test_input: str, *, exact: bool = False) -> str:
    """Key used to decide whether two stdin inputs are the same test.

    Programs that read stdin with scanf-style parsing do not distinguish
    trailing spaces or a final newline, so "5\\n1 2 3" and "5\\n1 2 3\\n" are one
    test. For assignments that read no stdin, whitespace-only inputs are
    deliberate robustness cases, so identity stays exact there.
    """
    if exact:
        return test_input
    lines = test_input.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip("\n")


def _validate_candidate(
    value: Any,
    pass_number: int,
    *,
    allow_ambiguities: bool = False,
    fallback_summary: str = "",
    fallback_contract: dict[str, Any] | None = None,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Validate generated input/output pairs before independent verification.

    Empty stdin is preserved as a possible test input. The assignment contract
    and independent verifier—not a blanket non-empty-string rule—decide whether
    a particular task permits it. allow_empty permits a wholly unusable batch
    to reach the caller's bounded replacement path.
    """
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

    if (
        not isinstance(cases, list)
        or (not cases and not allow_empty)
        or len(cases) > MAX_GENERATED_CASES
    ):
        raise TestGenerationError("invalid_test_count", f"Generation response {pass_number} must return 1..{MAX_GENERATED_CASES} input/output pairs.")
    cleaned, seen_ids, seen_inputs = [], set(), set()
    rejected_cases: list[dict[str, str]] = []
    for index, item in enumerate(cases, 1):
        if not isinstance(item, dict):
            rejected_cases.append({
                "id": f"generated-{index}",
                "reason": "candidate was not an object containing input and expected_output",
            })
            continue
        case_id = item.get("id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", case_id):
            case_id = f"generated-{index}"
        if case_id in seen_ids:
            rejected_cases.append({"id": case_id, "reason": "duplicate generated test ID"})
            continue
        # Reserve IDs before payload validation so malformed candidates cannot
        # let a later candidate silently reuse the same identity.
        seen_ids.add(case_id)
        test_input = item.get("input", item.get("stdin"))
        expected_output = item.get("expected_output", item.get("expected"))
        if not isinstance(test_input, str):
            rejected_cases.append({"id": case_id, "reason": "input is missing or is not text"})
            continue
        if len(test_input) > MAX_INPUT_CHARS:
            rejected_cases.append({"id": case_id, "reason": f"input exceeds {MAX_INPUT_CHARS} characters"})
            continue
        if not isinstance(expected_output, str):
            rejected_cases.append({"id": case_id, "reason": "expected_output is missing or is not text"})
            continue
        if len(expected_output) > MAX_INPUT_CHARS:
            rejected_cases.append({"id": case_id, "reason": f"expected_output exceeds {MAX_INPUT_CHARS} characters"})
            continue
        tags = item.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(x, str) for x in tags):
            tags = []
        tags = sorted(set(tag.strip()[:24] for tag in tags if tag.strip()))[:3] or ["generated"]
        rationale = item.get("rationale", item.get("description", ""))
        if not isinstance(rationale, str):
            rationale = ""
        if test_input in seen_inputs:
            rejected_cases.append({"id": case_id, "reason": "duplicate generated input"})
            continue
        seen_inputs.add(test_input)
        cleaned.append({
            "id": case_id,
            "input": test_input,
            "expected_output": expected_output,
            "tags": tags,
            "description": rationale.strip()[:240],
        })
    if not cleaned and not allow_empty:
        preview = "; ".join(
            f"{item['id']}: {item['reason']}" for item in rejected_cases[:5]
        )
        raise TestGenerationError(
            "no_valid_generated_candidates",
            "The LLM returned no usable input/output pairs. "
            + (preview or "All generated inputs were duplicates."),
        )
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
        "rejected_generated_cases": rejected_cases,
    }

_WORD_PATTERN = re.compile(r"[a-z_][a-z0-9_]*|\d+")
_NUMBER_PATTERN = re.compile(r"^\d+$")
# Function words and generic I/O vocabulary that any restatement may use
# without having copied it from the submitted implementation.
_GROUNDING_STOPWORDS = frozenset("""
a an the and or but nor not no of to in on at by for from with without into onto over under
after before during until while when where which who whom whose that this these those it its
is are was were be been being has have had do does did done can could shall should will would
may might must each every all any some both either neither one once twice only exactly same
than then there their them they as if so such per via also more most less least very
input inputs output outputs stdin stdout value values number numbers string strings line lines
text program result results case cases test tests format formatting rule rules type types
print printed printing return returns returned valid invalid correct correctly use used using
end start first last fixed required require requires follow followed following newline
""".split())
_NEGATION_WORDS = frozenset(
    "missing no not lacks lack lacking without unchecked never fails fail absent unvalidated "
    "unguarded ignores ignore omit omits omitted incorrect wrong overflow leak leaks".split()
)


def _grounding_stem(word: str) -> str:
    for suffix in ("izing", "ized", "izes", "ize", "ions", "ion", "ing", "ed", "es", "s"):
        if suffix == "s" and word.endswith("ss"):
            continue
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def _grounding_terms(text: str) -> set[str]:
    terms = set()
    for token in _WORD_PATTERN.findall(text.casefold()):
        if _NUMBER_PATTERN.match(token):
            terms.add(token)
        elif token not in _GROUNDING_STOPWORDS and len(token) > 2:
            terms.add(_grounding_stem(token))
    return terms


def _submission_derived_terms(text: str, *, spec_terms: set[str], submission_terms: set[str]) -> list[str]:
    """Terms in text that came from the submission analysis but not the specification."""
    return sorted((_grounding_terms(text) & submission_terms) - spec_terms)


def _remove_submission_derived_contract_entries(
    inferred: dict[str, Any],
    *,
    spec_text: str,
    submission_analysis: dict[str, Any] | None,
    fallback_contract: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Strip contract statements that restate the student's code, not the assignment.

    The generator sees an implementation analysis so it can choose useful test
    dimensions. The inferred contract, however, is later presented to the
    grader as assignment-derived, so any entry that uses vocabulary or numeric
    literals found only in the implementation analysis (loop bounds, buffer
    sizes, keywords, library calls) is removed deterministically and recorded.
    """
    if not isinstance(submission_analysis, dict) or not submission_analysis:
        return inferred, []
    submission_text = json.dumps(submission_analysis, ensure_ascii=False)
    spec_terms = _grounding_terms(spec_text)
    submission_terms = _grounding_terms(submission_text)
    cleaned = dict(inferred)
    removed: list[dict[str, str]] = []
    for field in ("input_format", "output_format"):
        value = cleaned.get(field, "")
        leaked = _submission_derived_terms(str(value), spec_terms=spec_terms, submission_terms=submission_terms)
        if leaked:
            removed.append({"field": field, "entry": str(value)[:400], "submission_terms": ", ".join(leaked[:8])})
            cleaned[field] = str(fallback_contract.get(field, ""))
    for field in ("constraints", "semantics", "formatting"):
        kept = []
        for entry in cleaned.get(field, []):
            leaked = _submission_derived_terms(entry, spec_terms=spec_terms, submission_terms=submission_terms)
            if leaked:
                removed.append({"field": field, "entry": entry[:400], "submission_terms": ", ".join(leaked[:8])})
            else:
                kept.append(entry)
        cleaned[field] = kept
    if not cleaned.get("semantics"):
        cleaned["semantics"] = list(fallback_contract.get("semantics", []))
    return cleaned, removed


def _drop_self_contradicting_weak_points(
    weak_points: list[str], present_behaviour: list[str],
) -> tuple[list[str], list[str]]:
    """Drop a 'weakness' that merely restates a check the analysis says is present.

    A weak point that adds no negation (missing, unchecked, leak, ...) and shares
    nearly all its terms with an observed-behaviour bullet describes code that
    exists, not a defect, so it must not be shown to the user as a weakness.
    """
    observed = [(_grounding_terms(item), set(item.casefold().split())) for item in present_behaviour]
    kept, dropped = [], []
    for point in weak_points:
        terms = _grounding_terms(point)
        words = set(point.casefold().split())
        contradicted = False
        for item_terms, item_words in observed:
            if not terms or not item_terms:
                continue
            overlap = len(terms & item_terms) / len(terms | item_terms)
            adds_negation = bool((words & _NEGATION_WORDS) - item_words)
            if overlap >= 0.75 and not adds_negation:
                contradicted = True
                break
        (dropped if contradicted else kept).append(point)
    return kept, dropped


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
    if len(submission_source.encode("utf-8")) > MAX_SUBMISSION_SOURCE_BYTES:
        raise TestGenerationError(
            "submission_source_too_large",
            f"The C source is too large for the test-design stage (limit: {MAX_SUBMISSION_SOURCE_BYTES} bytes).",
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
    analysis = {
        "implementation_summary": summary.strip()[:700],
        **{
            field: [item.strip()[:180] for item in result[field] if item.strip()][:8]
            for field in list_fields
        },
    }
    kept, dropped = _drop_self_contradicting_weak_points(
        analysis["likely_weak_points"],
        analysis["input_parsing_and_validation"] + analysis["control_flow_and_data_types"],
    )
    analysis["likely_weak_points"] = kept
    if dropped:
        analysis["dropped_contradictory_weak_points"] = dropped
    return analysis


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
    stdin_unused: bool = False,
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
                        # Results are linked to tests by id; the enum pins the ids.
                        "id": {"type": "string", "enum": [str(case["id"]) for case in candidate_cases]},
                        "input": {"type": "string", "maxLength": MAX_INPUT_CHARS, "description": "Exact stdin text; empty is valid when the assignment reads no stdin."},
                        "input_is_valid": {"type": "boolean"},
                        "expected_output": {"type": "string"},
                        "reason": {"type": "string", "maxLength": 240},
                    },
                    "required": ["id", "input", "input_is_valid", "expected_output", "reason"],
                },
            },
        },
        "required": ["verifications"],
    }
    examples = _instructor_hint_examples(instructor_examples)
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
            "set input_is_valid=false and explain briefly. Also set input_is_valid=false when a count "
            "declared in the input (for example n) does not match the number of values supplied. "
            "Return exactly one verification for every id."
            + (
                " This assignment reads no stdin: every supplied stdin is a valid robustness test that a "
                "correct program ignores, so input_is_valid is true and expected_output is the same fixed "
                "output for every id."
                if stdin_unused else ""
            )
        ),
    }
    verifier_messages = [
        {
            "role": "system",
            "content": (
                "You are an independent expected-output verifier for programming tests. "
                "You are not reviewing or executing the student's code. Derive each answer from "
                "the assignment specification yourself, using instructor examples only to understand "
                "the intended conventions. Return exactly one JSON object with a top-level "
                "'verifications' array. Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(payload, sort_keys=True, ensure_ascii=False)},
    ]
    required_item_fields = {
        "id", "input", "input_is_valid", "expected_output", "reason",
    }
    alternate_array_keys = (
        "results", "checks", "test_results", "verification_results",
    )

    def parse_verifier_response(response: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(response, dict):
            raise TestGenerationError(
                "expected_output_verification_failed",
                f"Verifier returned {type(response).__name__}, not a JSON object.",
            )
        verifications = response.get("verifications")
        if not isinstance(verifications, list):
            # Accept a compatible envelope alias only when it is unambiguous
            # and every entry still satisfies the complete verification contract.
            alternatives = [
                key for key in alternate_array_keys
                if isinstance(response.get(key), list)
                and all(
                    isinstance(item, dict) and required_item_fields.issubset(item)
                    for item in response[key]
                )
            ]
            if len(alternatives) == 1:
                verifications = response[alternatives[0]]
            else:
                keys = sorted(str(key) for key in response.keys())[:12]
                raise TestGenerationError(
                    "expected_output_verification_failed",
                    "Verifier response is missing the required 'verifications' array. "
                    f"Top-level keys returned: {keys}; valid alternate arrays found: {alternatives}.",
                )

        if len(verifications) != len(candidate_cases):
            raise TestGenerationError(
                "expected_output_verification_failed",
                f"Verifier returned {len(verifications)} items for {len(candidate_cases)} supplied inputs.",
            )
        checks: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(verifications, 1):
            if not isinstance(item, dict):
                raise TestGenerationError(
                    "expected_output_verification_failed",
                    f"Verifier item {index} is {type(item).__name__}, not an object.",
                )
            missing_fields = sorted(required_item_fields - set(item))
            if missing_fields:
                raise TestGenerationError(
                    "expected_output_verification_failed",
                    f"Verifier item {index} is missing fields: {missing_fields}.",
                )
            case_id = item.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise TestGenerationError(
                    "expected_output_verification_failed",
                    f"Verifier item {index} has a missing or invalid id.",
                )
            if case_id in checks:
                raise TestGenerationError(
                    "expected_output_verification_failed",
                    f"Verifier returned duplicate results for test {case_id}.",
                )
            if not isinstance(item.get("input"), str):
                raise TestGenerationError(
                    "expected_output_verification_failed",
                    f"Verifier input echo for test {case_id} is not text.",
                )
            if type(item.get("input_is_valid")) is not bool:
                raise TestGenerationError(
                    "expected_output_verification_failed",
                    f"Verifier input_is_valid for test {case_id} is not boolean.",
                )
            if not isinstance(item.get("expected_output"), str):
                raise TestGenerationError(
                    "expected_output_verification_failed",
                    f"Verifier expected_output for test {case_id} is not text.",
                )
            if len(item["expected_output"]) > MAX_INPUT_CHARS:
                raise TestGenerationError(
                    "expected_output_verification_failed",
                    f"Verifier expected_output for test {case_id} exceeds {MAX_INPUT_CHARS} characters.",
                )
            if not isinstance(item.get("reason"), str):
                raise TestGenerationError(
                    "expected_output_verification_failed",
                    f"Verifier reason for test {case_id} is not text.",
                )
            checks[case_id] = item

        expected_ids = {str(case["id"]) for case in candidate_cases}
        missing_ids = sorted(expected_ids - set(checks))
        unexpected_ids = sorted(set(checks) - expected_ids)
        if missing_ids:
            raise TestGenerationError(
                "expected_output_verification_failed",
                f"Verifier omitted test {missing_ids[0]}.",
            )
        if unexpected_ids:
            raise TestGenerationError(
                "expected_output_verification_failed",
                f"Verifier returned unexpected test IDs {unexpected_ids[:5]}.",
            )
        # A differing input echo is handled per case below: one oversized or
        # mis-copied input must not invalidate every other verification.
        return checks

    verifier_errors: list[str] = []
    checks: dict[str, dict[str, Any]] | None = None
    for attempt in range(2):
        stage_name = (
            "assignment_expected_output_verification_single_pass"
            if attempt == 0
            else "assignment_expected_output_verification_repair"
        )
        messages = verifier_messages
        if attempt == 1:
            repair_payload = dict(
                payload,
                corrective_instruction=(
                    "Your previous response did not satisfy the required verification response structure. "
                    "Re-solve the same supplied inputs from the assignment specification. Return exactly "
                    "one JSON object whose only top-level field is 'verifications'. Its value must be an "
                    "array with exactly one object per supplied input. Every object must contain all five "
                    "fields: id (copy exactly), input (copy the exact stdin string, including empty strings "
                    "and newlines), input_is_valid (JSON boolean), expected_output (exact stdout for a "
                    "correct solution), and reason (short text). Do not omit fields, rename keys, group "
                    "results under another key, add prose, or use student code or proposed outputs."
                ),
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are correcting a malformed independent test verifier response. "
                        "Re-derive the expected output from the assignment. Output exactly one JSON "
                        "object with a top-level verifications array and no other top-level fields. "
                        "Return JSON only."
                    ),
                },
                {"role": "user", "content": json.dumps(repair_payload, sort_keys=True, ensure_ascii=False)},
            ]
        try:
            response = request_json(messages, schema, stage_name)
            checks = parse_verifier_response(response)
            break
        except Exception as exc:
            detail = " ".join(str(exc).split())[:500] or type(exc).__name__
            verifier_errors.append(f"{stage_name}: {detail}")
            checks = None

    if checks is None:
        raise TestGenerationError(
            "expected_output_verification_failed",
            "The independent verifier did not return a structurally valid response after one corrective retry. "
            + " | ".join(verifier_errors),
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
        if _input_identity(str(check.get("input", "")), exact=stdin_unused) != _input_identity(case["input"], exact=stdin_unused):
            # The verifier solved a different text than the real test input
            # (typically a long input shortened to fit the model context), so
            # its expected output cannot be trusted for this case.
            rejected.append({
                "id": case_id,
                "reason": (
                    "independent verifier's copy of the input differs from the test input "
                    f"({len(str(check.get('input', '')))} vs {len(case['input'])} characters; long inputs may be "
                    "shortened to fit the model context), so its expected output was not trusted"
                ),
            })
            continue
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
    oracle: Any = None,
    oracle_trusted: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate input/output pairs and independently verify them.

    With an execution ``oracle`` (see testing.oracle), expected outputs come
    from running a reference program on each generated input. A trusted oracle
    (the instructor's reference, or a model-written program that reproduced
    every instructor case) is final; an untrusted one must agree with the
    independent model check. Without an oracle, the generator and the
    independent check must agree.

    A single bounded refill may replace rejected or missing candidates. The
    trusted_reference_solution argument remains ignored for compatibility:
    no generated or configured C reference source is compiled or executed.
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
    examples = _instructor_hint_examples(instructor_examples)
    stdin_unused = _assignment_reads_no_stdin(problem, problem_contract)
    instructor_inputs = {
        case["input"] for case in (instructor_examples or [])
        if isinstance(case, dict) and isinstance(case.get("input"), str)
    }
    base = {
        "assignment": problem,
        "assignment_reads_stdin": not stdin_unused,
        "problem_contract": problem_contract,
        "instructor_examples_as_hints": examples,
        "instructor_case_count": len(instructor_inputs),
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
            "spec_summary": {"type": "string", "maxLength": 600},
            "ambiguities": {"type": "array", "items": {"type": "string", "maxLength": CONTRACT_ITEM_MAX_CHARS}, "maxItems": 5},
            "inferred_contract": {
                "type": "object",
                "properties": {
                    "input_format": {"type": "string", "maxLength": 300},
                    "output_format": {"type": "string", "maxLength": 300},
                    "constraints": {"type": "array", "items": {"type": "string", "maxLength": CONTRACT_ITEM_MAX_CHARS}, "maxItems": 8},
                    "semantics": {"type": "array", "items": {"type": "string", "maxLength": CONTRACT_ITEM_MAX_CHARS}, "maxItems": 8},
                    "formatting": {"type": "array", "items": {"type": "string", "maxLength": CONTRACT_ITEM_MAX_CHARS}, "maxItems": 6},
                },
                "required": ["input_format", "output_format", "constraints", "semantics", "formatting"],
            },
            "test_inputs": {
                "type": "array",
                "minItems": 0,
                "maxItems": max_cases,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "maxLength": 48},
                        "input": {"type": "string", "maxLength": GENERATED_INPUT_MAX_CHARS, "description": "Exact stdin text; empty is valid when the assignment reads no stdin."},
                        "expected_output": {"type": "string", "maxLength": GENERATED_OUTPUT_MAX_CHARS},
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
        "First formalize the I/O contract from the assignment and problem_contract alone: "
        "inferred_contract must never restate details of the submitted implementation (loop bounds, "
        "buffer or allocation sizes, variable names, keywords, library calls, or initial values) unless "
        "the assignment itself states them. Generate diverse valid inputs and independently solve each "
        "input from the assignment to provide expected_output. Do not write, propose, or return C code. "
        "Do not invent requirements or guess ambiguous behaviour. If a required rule is genuinely "
        "ambiguous, report it in ambiguities. Include normal, boundary, empty/singleton, negative/zero, "
        "and task-specific cases only when valid. Prefer concise inputs and outputs. Return JSON only."
    )
    stdin_instruction = (
        "This assignment reads no stdin, so a correct program must ignore stdin entirely and its "
        "expected_output is the same fixed output for every test. Generate distinct stdin-robustness "
        "edge cases that check the program neither reads nor is affected by stdin: for example "
        "whitespace or newlines only, a single number, several numbers, plain text, a very long line, "
        "and data that looks like input for a different task. Tag each case 'edge' and 'stdin-ignored'. "
        if stdin_unused else ""
    )
    payload = dict(
        base,
        instruction=stdin_instruction + (
            f"Generate up to {max_cases} additional distinct test cases. Each test case must contain "
            "id, input, expected_output, and up to three short tags. The input is the exact stdin text for one "
            "test (not a label such as 'normal' or 'edge'). When the input declares a count (for example "
            f"n), supply exactly that many values. Keep every input under {GENERATED_INPUT_MAX_CHARS} "
            "characters (use small sizes such as grids up to 6 x 6 or up to 30 values); large performance "
            "cases already come from the instructor. "
            "The empty string is valid only when the "
            "assignment requires no stdin; whitespace-only text must never stand in for missing required "
            "input. Derive expected_output from the assignment contract, not from the submitted code. "
            "Follow the style shown by instructor "
            "examples while exploring new inputs. The instructor suite already has instructor_case_count cases; "
            "every input you return must be new: different values and sizes from the hints, not a copy "
            "with different whitespace, because copies are discarded. Do not include "
            "C source, coverage narration, or long rationales."
        ),
    )
    generation_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, sort_keys=True, ensure_ascii=False)},
    ]
    try:
        result = request_json(
            generation_messages,
            generation_schema,
            "assignment_test_generation_single_pass",
        )
    except Exception as first_error:
        # Usually a response cut off at the token limit (finish_reason=length).
        # Retry once asking for half as many, shorter cases instead of failing.
        fewer = max(1, max_cases // 2)
        retry_schema = json.loads(json.dumps(generation_schema))
        retry_schema["properties"]["test_inputs"]["maxItems"] = fewer
        retry_payload = dict(
            payload,
            maximum_cases=fewer,
            requested_case_count=fewer,
            previous_attempt_problem=(
                "The previous answer was cut off or invalid. Return at most "
                f"{fewer} cases, keep each input under 300 characters, and keep every contract "
                "entry to one short sentence."
            ),
        )
        retry_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(retry_payload, sort_keys=True, ensure_ascii=False)},
        ]
        try:
            result = request_json(retry_messages, retry_schema, "assignment_test_generation_single_pass_retry")
        except Exception as retry_error:
            raise TestGenerationError(
                "generation_failed",
                f"Test generation failed twice. First attempt: {' '.join(str(first_error).split())[:400]} | "
                f"Shorter retry: {' '.join(str(retry_error).split())[:400]}",
            ) from retry_error
    candidate = _validate_candidate(
        result,
        1,
        allow_ambiguities=False,
        fallback_summary=problem,
        fallback_contract=fallback_contract,
        allow_empty=True,
    )
    # The inferred contract is later shown to the grader as assignment-derived,
    # so remove anything the generator copied from the implementation analysis.
    spec_text = "\n".join([
        problem,
        json.dumps(problem_contract, ensure_ascii=False, default=str),
        json.dumps(examples, ensure_ascii=False),
        additional_focus,
    ])
    candidate["inferred_contract"], removed_entries = _remove_submission_derived_contract_entries(
        candidate["inferred_contract"],
        spec_text=spec_text,
        submission_analysis=submission_analysis,
        fallback_contract=fallback_contract,
    )
    candidate["submission_derived_contract_entries_removed"] = removed_entries

    rejected: list[dict[str, str]] = list(candidate.get("rejected_generated_cases", []))
    used_inputs = {_input_identity(item, exact=stdin_unused) for item in instructor_inputs}

    def retain_contract_valid_cases(
        cases: list[dict[str, Any]],
        *,
        id_prefix: str = "",
    ) -> list[dict[str, Any]]:
        retained: list[dict[str, Any]] = []
        for index, case in enumerate(cases, 1):
            item = dict(case)
            if id_prefix:
                item["id"] = f"{id_prefix}{index}-{str(item.get('id', 'case'))[:48]}"
            identity = _input_identity(item["input"], exact=stdin_unused)
            already_seen = identity in used_inputs
            used_inputs.add(identity)
            reason = (
                _validate_input_against_contract(item["input"], problem_contract)
                or _declared_count_mismatch(problem, item["input"])
            )
            if reason:
                rejected.append({"id": str(item["id"]), "reason": reason})
            elif already_seen:
                rejected.append({
                    "id": str(item["id"]),
                    "reason": "duplicate of an instructor or earlier generated input",
                })
            else:
                retained.append(item)
        return retained

    def check_batch(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any], int]:
        """Return (accepted, rejected, summary, model_calls) for one candidate batch."""
        if oracle is None:
            accepted, rejected_now, summary = _verify_expected_outputs(
                problem=problem, problem_contract=problem_contract,
                inferred_contract=candidate["inferred_contract"], candidate_cases=batch,
                instructor_examples=instructor_examples, request_json=request_json, stdin_unused=stdin_unused,
            )
            for case in accepted:
                case["expected_output_source"] = "generator_and_independent_check_agreed"
            return accepted, rejected_now, summary, 1
        executed: list[dict[str, Any]] = []
        rejected_now = []
        for case in batch:
            outcome = oracle.run(case["input"])
            if not outcome["ok"]:
                rejected_now.append({"id": str(case["id"]), "reason": f"reference program rejected this input: {outcome['reason']}"[:300]})
                continue
            executed.append({**case, "expected_output": outcome["stdout"], "generator_expected_output": case["expected_output"]})
        summary = {
            "method": oracle.kind if oracle_trusted else f"{oracle.kind}_plus_independent_check",
            "requested": len(batch), "verified": len(executed), "rejected": len(rejected_now),
            "comparison": "expected output computed by executing the reference program",
        }
        if oracle_trusted or not executed:
            for case in executed:
                case["expected_output_source"] = oracle.kind
            return executed, rejected_now, summary, 0
        accepted, disagreed, check_summary = _verify_expected_outputs(
            problem=problem, problem_contract=problem_contract,
            inferred_contract=candidate["inferred_contract"], candidate_cases=executed,
            instructor_examples=instructor_examples, request_json=request_json, stdin_unused=stdin_unused,
        )
        for item in disagreed:
            item["reason"] = item["reason"].replace(
                "generated expected_output disagreed with the independent verifier",
                "reference program output disagreed with the independent check",
            )
        for case in accepted:
            case["expected_output_source"] = f"{oracle.kind}_agreed_with_independent_check"
        summary.update({"verified": len(accepted), "rejected": len(rejected_now) + len(disagreed),
                        "comparison": "reference program output must equal the independent check (trimmed, normalized newlines)"})
        return accepted, rejected_now + disagreed, summary, 1

    first_batch = retain_contract_valid_cases(candidate["test_inputs"])
    verified_cases: list[dict[str, Any]] = []
    verification_calls = 0
    verification_requested = 0
    verification_rejected = 0
    first_batch_verified: list[dict[str, Any]] = []
    first_verification_summary: dict[str, Any] = {
        "method": "independent_second_llm_call",
        "requested": 0,
        "verified": 0,
        "rejected": 0,
        "comparison": "trimmed exact text with normalized line endings",
    }

    if first_batch:
        first_batch_verified, first_batch_rejected, first_verification_summary, calls_used = check_batch(first_batch)
        verification_calls += calls_used
        verification_requested += len(first_batch)
        verification_rejected += len(first_batch_rejected)
        verified_cases.extend(first_batch_verified)
        rejected.extend(first_batch_rejected)

    initial_rejected_count = len(rejected)
    repair_count = 0
    repair_response_count = 0
    repair_failure: str | None = None
    repair_verified: list[dict[str, Any]] = []
    repair_verification_summary: dict[str, Any] = {
        "method": "independent_second_llm_call",
        "requested": 0,
        "verified": 0,
        "rejected": 0,
        "comparison": "trimmed exact text with normalized line endings",
    }

    # A single bounded refill prevents one malformed/invalid batch from
    # aborting generation. Every replacement goes through the same strict
    # structural checks and independent expected-output verification.
    # max_cases is a ceiling. A partial but independently verified batch is
    # usable; refill only when the entire first batch yielded no verified pair.
    missing_count = max_cases if not verified_cases else 0
    if missing_count:
        repair_count = 1
        repair_payload = dict(
            base,
            requested_replacement_count=missing_count,
            previously_used_inputs=[
                _shorten_for_prompt(item, MAX_HINT_FIELD_CHARS) for item in sorted(used_inputs)[:max_cases]
            ],
            previously_rejected_candidates=[
                {"id": item.get("id", "candidate"), "reason": item.get("reason", "rejected")}
                for item in rejected[:40]
            ],
            provisional_inferred_contract=candidate["inferred_contract"],
            instruction=stdin_instruction + (
                f"The first batch did not produce {max_cases} independently verified distinct cases. "
                f"Generate up to {missing_count} NEW replacement cases. Each must contain an id, exact "
                "stdin text in input, expected_output, and up to three concise tags. Do not reuse any "
                "previously_used_inputs. Follow the assignment and problem_contract exactly. Empty stdin "
                "is valid only when the assignment requires no stdin; never use spaces or blank text as "
                "a placeholder for missing required values. For input-required tasks, include all required "
                "tokens and lines. Independently derive expected_output from the assignment, not from the "
                "student implementation. Do not return C code."
            ),
        )
        repair_schema = json.loads(json.dumps(generation_schema))
        repair_schema["properties"]["test_inputs"]["minItems"] = 1
        repair_schema["properties"]["test_inputs"]["maxItems"] = missing_count
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You repair an incomplete test set for a C programming assignment. "
                    "The assignment specification is authoritative. Return only new, distinct, valid "
                    "input/expected-output pairs. Empty stdin is valid only for a task that does not "
                    "read stdin. Never use blank or whitespace-only placeholders for required input. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(repair_payload, sort_keys=True, ensure_ascii=False),
            },
        ]
        try:
            repair_result = request_json(
                repair_messages,
                repair_schema,
                "assignment_test_generation_repair",
            )
            repair_response_count = 1
            repair_candidate = _validate_candidate(
                repair_result,
                2,
                allow_ambiguities=True,
                fallback_summary=problem,
                fallback_contract=fallback_contract,
                allow_empty=True,
            )
            if repair_candidate["ambiguities"]:
                repair_failure = (
                    "replacement generation reported unresolved ambiguities: "
                    + "; ".join(repair_candidate["ambiguities"][:3])
                )
                rejected.append({"id": "refill-batch", "reason": repair_failure[:300]})
            else:
                rejected.extend(repair_candidate.get("rejected_generated_cases", []))
                repair_batch = retain_contract_valid_cases(
                    repair_candidate["test_inputs"],
                    id_prefix="refill-",
                )
                if repair_batch:
                    repair_verified, repair_rejected, repair_verification_summary, calls_used = check_batch(repair_batch)
                    verification_calls += calls_used
                    verification_requested += len(repair_batch)
                    verification_rejected += len(repair_rejected)
                    verified_cases.extend(repair_verified)
                    rejected.extend(repair_rejected)
        except Exception as exc:
            repair_failure = " ".join(str(exc).split())[:240] or type(exc).__name__
            rejected.append({
                "id": "refill-batch",
                "reason": ("replacement generation or verification failed: " + repair_failure)[:300],
            })

    if not verified_cases:
        preview = "; ".join(
            f"{item['id']}: {item['reason']}" for item in rejected[:8]
        )
        raise TestGenerationError(
            "no_generated_cases_verified",
            "No generated input/output pair passed deterministic validation and independent "
            "expected-output verification after one bounded refill. "
            + (preview or "Review the assignment input/output contract and instructor examples."),
        )

    candidate["test_inputs"] = verified_cases[:max_cases]
    candidate["stdin_unused"] = stdin_unused
    candidate["generation_mode"] = "generate_then_independently_verify_with_bounded_refill"
    candidate["generation_response_count"] = 1 + repair_response_count
    candidate["verification_response_count"] = verification_calls
    candidate["verification_summary"] = {
        "method": "independent_second_llm_call",
        "requested": verification_requested,
        "verified": len(candidate["test_inputs"]),
        "rejected": verification_rejected,
        "verification_calls": verification_calls,
        "comparison": "trimmed exact text with normalized line endings",
    }
    candidate["rejected_generated_cases"] = rejected

    history = [{
        "pass": 1,
        "status": "completed" if first_batch_verified else "recovered_or_refilled",
        "mode": "generate_then_independently_verify",
        "test_count": len(first_batch_verified),
        "requested_test_count": max_cases,
        "rejected_test_count": initial_rejected_count,
        "verification": first_verification_summary,
    }]
    if repair_count:
        history.append({
            "pass": 2,
            "status": "failed" if repair_failure else ("completed" if repair_verified else "no_additional_cases_verified"),
            "mode": "bounded_replacement_generation",
            "test_count": len(repair_verified),
            "requested_test_count": missing_count,
            "rejected_test_count": len(rejected) - initial_rejected_count,
            "verification": repair_verification_summary,
            **({"error": repair_failure} if repair_failure else {}),
        })
    candidate["generation_passes"] = history
    return candidate, history

_NO_STDIN_FIELD = re.compile(
    r"^\s*(?:none|n/?a|no\s+stdin|no\s+input(?:\s+(?:is\s+)?(?:read|required|needed))?)\s*\.?\s*$",
    re.IGNORECASE,
)
_NO_STDIN_STATEMENT = re.compile(
    r"\b(?:without\s+reading\s+(?:any\s+)?(?:input|stdin)"
    r"|(?:reads|takes|requires|needs)\s+no\s+(?:input|stdin)"
    r"|does\s+not\s+(?:read|take|require|need)\s+(?:any\s+)?(?:input|stdin)"
    r"|no\s+(?:input|stdin)\s+is\s+(?:read|required|needed))\b",
    re.IGNORECASE,
)


def _assignment_reads_no_stdin(problem: str, problem_contract: dict[str, Any] | None) -> bool:
    """Return True only when the assignment explicitly states that stdin is unused.

    For such a task every non-empty stdin is ignored by a correct program, so
    empty input is the only meaningful test input. Explicit contract flags win
    over prose; prose detection uses narrow phrases such as "without reading
    input" so that statements like "if there is no input, print 0" do not match.
    """
    contract = problem_contract if isinstance(problem_contract, dict) else {}
    for key in ("reads_stdin", "stdin_required", "requires_input"):
        if isinstance(contract.get(key), bool):
            return not contract[key]
    for key in ("input_token_count", "expected_token_count", "max_input_tokens"):
        value = contract.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value == 0:
            return True
    for key in ("input", "input_format", "input_range", "stdin"):
        value = contract.get(key)
        if isinstance(value, str) and (
            _NO_STDIN_FIELD.match(value) or _NO_STDIN_STATEMENT.search(value)
        ):
            return True
    return bool(_NO_STDIN_STATEMENT.search(problem or ""))


def _validate_input_against_contract(
    test_input: str,
    problem_contract: dict[str, Any],
) -> str | None:
    """Validate stdin against explicit, machine-readable instructor constraints.

    Free-form prose cannot be interpreted as a formal grammar. Strict structural
    validation is available through input_schema/input_fields token descriptors.
    """
    if not isinstance(test_input, str):
        return "input must be text"
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
    if isinstance(min_count, int) and not isinstance(min_count, bool) and len(tokens) < min_count:
        return f"expected at least {min_count} whitespace-delimited input tokens"
    if isinstance(max_count, int) and not isinstance(max_count, bool) and len(tokens) > max_count:
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
    exact_inputs = bool(generated_spec.get("stdin_unused"))
    seen_ids: set[str] = set()

    for index, case in enumerate(instructor_cases or [], 1):
        if not isinstance(case, dict) or not isinstance(case.get("input"), str) or not isinstance(case.get("expected"), str):
            raise TestGenerationError("invalid_instructor_test", f"Instructor test {index} is malformed.")
        test_input = case["input"]
        expected = case["expected"].replace("\r\n", "\n").replace("\r", "\n")
        if _input_identity(test_input, exact=exact_inputs) in seen_inputs:
            if not _outputs_match(seen_inputs[_input_identity(test_input, exact=exact_inputs)], expected):
                raise TestGenerationError("conflicting_instructor_tests", f"Instructor examples disagree about expected output for case {case.get('id', index)}.")
            for prior in merged:
                if _input_identity(str(prior.get("input")), exact=exact_inputs) == _input_identity(test_input, exact=exact_inputs):
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
        seen_inputs[_input_identity(test_input, exact=exact_inputs)] = expected
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
        if _input_identity(test_input, exact=exact_inputs) in seen_inputs:
            if not _outputs_match(seen_inputs[_input_identity(test_input, exact=exact_inputs)], expected):
                raise TestGenerationError("generated_instructor_disagreement", f"Generated expected output disagrees with an instructor example for test {case_id}.")
            continue
        final_id = f"generated-{case_id}"
        if final_id in seen_ids:
            raise TestGenerationError("duplicate_test_id", f"Generated test ID collides after normalization: {case_id}.")
        seen_inputs[_input_identity(test_input, exact=exact_inputs)] = expected
        seen_ids.add(final_id)
        merged.append({
            "id": final_id,
            "input": test_input,
            "expected": expected,
            "tags": list(case.get("tags", [])),
            "description": str(case.get("description", ""))[:240],
            "origin": "llm_generated_independently_verified",
            **({"expected_output_source": case["expected_output_source"]} if case.get("expected_output_source") else {}),
        })
    if not merged:
        raise TestGenerationError("empty_suite", "No instructor or independently verified generated cases are available.")
    return merged

def oracle_caveat(record: dict[str, Any] | None) -> str:
    """One sentence describing where generated expected outputs came from."""
    record = record if isinstance(record, dict) else {}
    kind = record.get("kind")
    if kind == "instructor_reference":
        text = f"Generated expected outputs were computed by running the instructor's reference solution ({record.get('file', 'reference')}) in the sandbox."
        return text + (" " + record["warning"] if record.get("warning") else "")
    if kind == "llm_reference" and record.get("trusted"):
        return (
            "Generated expected outputs were computed by running a model-written reference program that first "
            f"reproduced all {record.get('validated_against_instructor_cases', 0)} instructor test case(s)."
        )
    if kind == "llm_reference":
        return (
            "Generated expected outputs were computed by running a model-written reference program and kept only "
            "where an independent model check gave the same answer."
        )
    return (
        "Generated expected outputs were NOT confirmed by executing a reference program; they rely on agreement "
        "between two model calls. " + str(record.get("reason", "")).strip()
    ).strip()


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
        "expected_output_oracle": generated_spec.get("expected_output_oracle", {}),
        "submission_derived_contract_entries_removed": generated_spec.get(
            "submission_derived_contract_entries_removed", []
        ),
        "expected_output_verification": generated_spec.get("verification_summary", {}),
        "oracle_kind": oracle_kind,
        "assignment_statement": assignment_statement,
        "problem_contract": problem_contract or {},
        "oracle_caveat": oracle_caveat(generated_spec.get("expected_output_oracle", {})),
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




def _build_expected_output_oracle(
    *,
    sandbox: Any,
    rules: dict[str, Any],
    reference_source: str,
    reference_path: str,
    instructor_cases: list[dict[str, Any]],
    request_json: Callable[[list[dict[str, str]], dict[str, Any], str], dict[str, Any] | None],
) -> tuple[Any, dict[str, Any]]:
    """Choose the source of truth for generated expected outputs."""
    from grading_agent.testing.oracle import build_llm_oracle, instructor_oracle

    if sandbox is None:
        return None, {
            "kind": "none", "trusted": False,
            "reason": "No sandbox is available (Docker not running), so no reference program could be executed; "
                      "expected outputs rely on agreement between two model calls and are not execution-checked.",
        }
    if reference_source:
        oracle = instructor_oracle(sandbox, reference_source, reference_path)
        build_error = oracle.prepare()
        if build_error:
            raise TestGenerationError(
                "reference_solution_build_failed",
                f"The instructor reference solution {Path(reference_path).name} did not build: {build_error[:600]}",
            )
        check = oracle.check_against(instructor_cases)
        record = {
            "kind": "instructor_reference", "trusted": True, "file": Path(reference_path).name,
            "instructor_cases_passed": len(check["passed"]),
            "instructor_cases_mismatched": [item["id"] for item in check["mismatched"]],
            "instructor_cases_unrunnable": [item["id"] for item in check["unrunnable"]],
        }
        if check["mismatched"]:
            record["warning"] = (
                "The instructor reference disagrees with these instructor test cases: "
                + ", ".join(item["id"] for item in check["mismatched"][:5])
                + ". Generated expected outputs follow the reference; check the test files."
            )
        return oracle, record
    oracle, record = build_llm_oracle(
        sandbox=sandbox,
        problem=str(rules.get("problem", "")),
        problem_contract=rules.get("problem_contract", {}) if isinstance(rules.get("problem_contract"), dict) else {},
        instructor_cases=instructor_cases,
        hint_examples=_instructor_hint_examples(instructor_cases),
        request_json=request_json,
    )
    if oracle is None:
        record.setdefault(
            "reason",
            "No usable model-written reference program; expected outputs rely on agreement between two "
            "model calls and are not execution-checked.",
        )
    return oracle, record


def prepare_assignment_test_suite(
    rules: dict[str, Any],
    *,
    sandbox: Any = None,
    request_json: Callable[[list[dict[str, str]], dict[str, Any], str], dict[str, Any] | None],
    project_root: Path,
    model_id: str,
    generation_seed: int = 42,
    temperature: float = 0.0,
    submission_source: str = "",
    submission_filename: str = "submission.c",
) -> dict[str, Any]:
    """Generate, verify, freeze, and bind assignment test cases.

    Expected outputs of generated inputs come from, in order of authority:
    the instructor's reference solution (``reference_solution`` in rules.json),
    run in the sandbox; otherwise a Python reference program the model writes
    from the assignment (never from the student's code), trusted once it
    reproduces every instructor case and otherwise required to agree with an
    independent model check; otherwise, without a sandbox, agreement between
    the generator and an independent model check (flagged as unexecuted).
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
    stdin_unused = _assignment_reads_no_stdin(str(rules.get("problem", "")), rules.get("problem_contract", {}))
    effective_requested = int(max_cases)
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
        generator_version=str(testing_cfg.get("generator_version", "1")) + ":execution-oracle-v23",
        rsi_passes=1,
        max_cases=int(max_cases),
        temperature=float(temperature),
        additional_focus=additional_focus,
        submission_source_sha256=submission_source_sha256,
    )
    reference_path_value = rules.get("resolved_reference_solution")
    reference_source = ""
    if reference_path_value:
        try:
            reference_source = Path(reference_path_value).read_text(encoding="utf-8")
        except OSError as exc:
            raise TestGenerationError("reference_solution_unreadable", f"Could not read the instructor reference solution: {exc}") from exc
    supplemental = {
        "base": fingerprint,
        "instructor_cases": instructor_cases,
        "expected_output_verification": "execution_oracle",
        # Changing the instructor reference must regenerate the suite.
        "reference_solution_sha256": hashlib.sha256(reference_source.encode("utf-8")).hexdigest() if reference_source else "",
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
            raise TestGenerationError("suite_schema_version_mismatch", "Frozen suite uses an obsolete schema; regenerate it with the two-pass expected-output verifier.")
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
        oracle, oracle_record = _build_expected_output_oracle(
            sandbox=sandbox,
            rules=rules,
            reference_source=reference_source,
            reference_path=str(reference_path_value or ""),
            instructor_cases=instructor_cases,
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
            oracle=oracle,
            oracle_trusted=bool(oracle_record.get("trusted")),
        )
        spec["expected_output_oracle"] = {key: value for key, value in oracle_record.items() if key != "program"}
        if oracle_record.get("program"):
            spec["expected_output_oracle"]["program_sha256"] = hashlib.sha256(oracle_record["program"].encode("utf-8")).hexdigest()
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
        oracle_record = spec.get("expected_output_oracle", {})
        oracle_kind = {
            "instructor_reference": "instructor-reference-solution-executed",
            "llm_reference": (
                "llm-reference-program-validated-on-instructor-cases" if oracle_record.get("trusted")
                else "llm-reference-program-agreed-with-independent-check"
            ),
        }.get(str(oracle_record.get("kind")), "two-model-agreement-unexecuted")
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
        text_parts = [str(case.get("id", ""))]
        if isinstance(tags, list):
            text_parts.extend(str(tag) for tag in tags)
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
        "requested_generated_test_count": effective_requested,
        "configured_max_generated_cases": int(max_cases),
        "generated_test_count_shortfall": max(0, effective_requested - generated_count),
        "stdin_unused": stdin_unused,
        "rejected_generated_test_count": len(frozen.get("rejected_generated_cases", [])),
        "edge_test_count": len(edge_cases),
        "generation_passes": generation_history,
        "metric_testset_paths": metric_bindings,
        "all_testset_path": str(all_path),
        "edge_testset_path": str(edge_path),
        "submission_analysis": frozen.get("submission_analysis", {}),
        "inferred_contract": frozen.get("inferred_contract", {}),
        "spec_summary": frozen.get("spec_summary", ""),
        "oracle_kind": frozen.get("oracle_kind", "two-model-agreement-unexecuted"),
        "oracle_caveat": frozen.get("oracle_caveat", oracle_caveat(frozen.get("expected_output_oracle", {}))),
        "expected_output_oracle": frozen.get("expected_output_oracle", {}),
        "expected_output_verification": verification_summary,
        "unresolved_ambiguities": frozen.get("generation_ambiguities", []),
    }
