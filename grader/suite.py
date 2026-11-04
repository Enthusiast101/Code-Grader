from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .llm import LLMClient, LLMError, extract_c_code
from .sandbox import DockerSandbox, SandboxError

GENERATOR_VERSION = "5.0"
EDGE_WORDS = {"edge", "boundary", "adversarial", "corner", "overflow", "negative", "zero", "minimum", "maximum", "large", "limit", "extreme"}


class SuiteError(RuntimeError):
    pass


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_edge(case: dict[str, Any]) -> bool:
    tags = {str(tag).casefold() for tag in case.get("tags", [])}
    return bool(tags & EDGE_WORDS) or "edge" in tags


def _clean_tests(value: Any, *, max_cases: int, pass_number: int) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        raw = value.get("tests", value.get("test_cases", value.get("test_inputs", value.get("additional_tests"))))
    else:
        raw = value
    if not isinstance(raw, list):
        raise SuiteError(f"Test-generation response {pass_number} did not return a tests array.")

    cleaned: list[dict[str, Any]] = []
    seen_inputs: set[str] = set()
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        test_input = item.get("input", item.get("stdin"))
        if not isinstance(test_input, str) or not test_input.strip() or len(test_input) > 20000:
            continue
        if test_input in seen_inputs:
            continue
        tags = item.get("tags", ["core"])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            tags = ["core"]
        tags = sorted({re.sub(r"[^a-z0-9_-]", "", tag.casefold()) for tag in tags if tag.strip()}) or ["core"]
        rationale = item.get("rationale", item.get("description", "Generated from the assignment contract."))
        if not isinstance(rationale, str):
            rationale = "Generated from the assignment contract."
        signal_text = " ".join([*tags, str(item.get("id", "")), rationale]).casefold()
        if any(word in signal_text for word in EDGE_WORDS):
            tags = sorted(set(tags) | {"edge"})
        case_id = item.get("id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", case_id):
            case_id = f"generated{pass_number}_{index:02d}"
        seen_inputs.add(test_input)
        cleaned.append({"id": case_id, "input": test_input, "tags": tags, "description": rationale.strip()[:240]})
        if len(cleaned) >= max_cases:
            break
    return cleaned


def _normalise_case_tags(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(tag, str) for tag in value):
        value = ["core"]
    tags = sorted({re.sub(r"[^a-z0-9_-]", "", tag.casefold()) for tag in value if tag.strip()})
    return tags or ["core"]


def _load_instructor_cases(rules: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, item in enumerate(rules.get("examples", []), 1):
        cases.append({
            "id": str(item.get("id", f"sample_{index:02d}")),
            "input": item["input"],
            "expected": item["expected"],
            "tags": _normalise_case_tags(item.get("tags", ["sample", "core"])),
            "description": str(item.get("description", "Instructor-provided sample."))[:240],
            "origin": "instructor_example",
        })
    path_value = rules["raw"].get("testset_path")
    if path_value is None:
        path_value = rules["test_generation"].get("testset_path")
    if path_value:
        candidate = Path(str(path_value))
        if candidate.is_absolute():
            raise SuiteError("testset_path must be relative to the rules.json directory.")
        resolved = (rules["root"] / candidate).resolve()
        if not resolved.is_relative_to(rules["root"].resolve()) or not resolved.is_file():
            raise SuiteError(f"Configured testset_path is missing or outside the project: {path_value}")
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SuiteError(f"Invalid test JSON in {resolved.name}: {exc}") from exc
        if not isinstance(data, list) or len(data) > 500:
            raise SuiteError("testset_path must contain a JSON array of at most 500 test cases.")
        for index, item in enumerate(data, 1):
            if not isinstance(item, dict) or not isinstance(item.get("input"), str) or not isinstance(item.get("expected"), str):
                raise SuiteError(f"Test case {index} in {resolved.name} must contain input and expected strings.")
            if not item["input"].strip() or len(item["input"]) > 20000 or len(item["expected"]) > 20000:
                raise SuiteError(f"Test case {index} in {resolved.name} exceeds the 20000-character input/output limit or is empty.")
            cases.append({"id": str(item.get("id", f"instructor_{index:02d}"))[:64], "input": item["input"], "expected": item["expected"],
                          "tags": _normalise_case_tags(item.get("tags", ["core"])), "description": str(item.get("description", "Instructor test."))[:240], "origin": "instructor_test"})
    return cases


def _fingerprint(rules: dict[str, Any], model: str, sandbox_image_id: str | None = None, llm_seed: int | None = None) -> str:
    cfg = rules["test_generation"]
    reference_path = cfg.get("reference_solution_path")
    reference_hash = None
    if reference_path:
        candidate = Path(str(reference_path))
        if candidate.is_absolute():
            raise SuiteError("reference_solution_path must be relative to rules.json.")
        resolved = (rules["root"] / candidate).resolve()
        if not resolved.is_relative_to(rules["root"].resolve()) or not resolved.is_file():
            raise SuiteError("Configured reference_solution_path is missing or outside the project.")
        reference_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()
    payload = {
        "generator_version": GENERATOR_VERSION,
        "problem": rules["problem"],
        "contract": rules["problem_contract"],
        "examples": rules["examples"],
        "testset_path_content": _load_instructor_cases(rules),
        "model": model,
        "llm_seed": int(llm_seed if llm_seed is not None else rules["llm"].get("seed", cfg["seed"])),
        "seed": cfg["seed"],
        "temperature": cfg["temperature"],
        "rsi_passes": 1,
        "max_cases": cfg["max_cases"],
        "generation_enabled": cfg["enabled"],
        "comparison_mode": rules["output_comparison"],
        "docker_image": rules["docker_image"],
        "docker_image_id": sandbox_image_id,
        "reference_sha256": reference_hash,
    }
    return _digest(payload)


def _validate_test_input_contract(test_input: str, contract: dict[str, Any]) -> str | None:
    """Reject candidates that violate explicit structured assignment constraints."""
    if not isinstance(test_input, str) or not test_input.strip():
        return "input must be a non-empty string"
    if len(test_input) > 20000:
        return "input exceeds the 20000-character limit"
    if "\x00" in test_input or any(ord(ch) < 32 and ch not in "\r\n\t" for ch in test_input):
        return "input contains a forbidden control character"
    if not isinstance(contract, dict):
        return None

    tokens = test_input.split()
    exact = contract.get("input_token_count", contract.get("expected_token_count"))
    min_tokens, max_tokens = contract.get("min_input_tokens"), contract.get("max_input_tokens")
    if isinstance(exact, int) and not isinstance(exact, bool) and len(tokens) != exact:
        return f"expected exactly {exact} input tokens"
    if isinstance(min_tokens, int) and len(tokens) < min_tokens:
        return f"expected at least {min_tokens} input tokens"
    if isinstance(max_tokens, int) and len(tokens) > max_tokens:
        return f"expected at most {max_tokens} input tokens"

    schema = contract.get("input_schema", contract.get("input_fields"))
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
                if not __import__("math").isfinite(number):
                    return f"field {name} must be finite"
            if number is not None:
                lower = field.get("min", field.get("minimum", 0 if kind in {"uint", "unsigned", "unsigned_integer", "natural"} else None))
                upper = field.get("max", field.get("maximum"))
                if lower is not None and number < lower:
                    return f"field {name} is below its minimum"
                if upper is not None and number > upper:
                    return f"field {name} exceeds its maximum"
            allowed = field.get("enum", field.get("allowed_values"))
            if isinstance(allowed, list) and token not in {str(value) for value in allowed}:
                return f"field {name} is not one of its allowed values"
    if not allow_extra and cursor != len(tokens):
        return f"input contains {len(tokens) - cursor} unexpected token(s)"
    return None


def _generate_test_inputs(llm: LLMClient, rules: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Generate one candidate suite; validation never re-enters the LLM."""
    cfg = rules["test_generation"]
    max_cases = int(cfg["max_cases"])
    system = (
        "You are a careful test designer for an automated C-program grader. "
        "Use only the instructor assignment. Do not inspect or infer from a student submission. "
        "Infer the complete input grammar, output grammar, constraints, valid-domain semantics, and formatting rules "
        "directly from the natural-language assignment statement. The instructor will NOT provide separate input/output "
        "format fields. Do not ask for them. If wording is ambiguous, record the ambiguity and choose the narrowest "
        "reasonable interpretation supported by the statement; do not invent constraints. "
        "Generate valid stdin inputs only; invalid input is included only if the statement explicitly defines it. "
        "Return JSON only. Every input must be self-contained. Tag each case as core or edge/boundary/adversarial. "
        "Examine every proposed input against the full assignment contract before returning it. Do not output expected answers."
    )
    user = {
        "task": rules["problem"],
        "instructor_contract": rules["problem_contract"],
        "instruction": (
            f"In this single test-generation pass, infer the complete input/output contract and create up to "
            f"{max_cases} high-value valid stdin cases. Cover ordinary behavior, explicit constraint boundaries, "
            "whitespace variations, numeric extremes, minimal structures and task-specific adversarial cases where valid. "
            "Return JSON with inferred_contract (input_format, output_format, constraints, semantics, formatting, "
            "ambiguities), tests, and review_summary. Each test has id, input, tags, rationale. "
            "Return only distinct cases that satisfy all stated constraints. Do not include expected outputs."
        ),
        "seed": cfg["seed"],
    }
    try:
        answer = llm.json(
            system,
            json.dumps(user, ensure_ascii=False, sort_keys=True),
            max_tokens=min(4096, llm.max_tokens),
        )
    except (LLMError, ValueError, TypeError) as exc:
        raise SuiteError(f"Single test-generation request failed: {exc}") from exc
    if not isinstance(answer, dict):
        raise SuiteError("The single test-generation response was not a JSON object.")
    inferred = answer.get("inferred_contract")
    required_contract_fields = ("input_format", "output_format", "semantics")
    if not isinstance(inferred, dict) or any(
        not isinstance(inferred.get(key), (str, list)) or not inferred.get(key)
        for key in required_contract_fields
    ):
        raise SuiteError("The single test-generation response omitted required inferred_contract fields.")
    inferred_contract = {
        key: value for key, value in inferred.items()
        if key in {"input_format", "output_format", "constraints", "semantics", "formatting", "ambiguities"}
    }
    proposed_cases = _clean_tests(answer, max_cases=max_cases, pass_number=1)
    accepted_cases, rejected_cases = [], []
    for case in proposed_cases:
        reason = _validate_test_input_contract(case["input"], rules.get("problem_contract", {}))
        if reason:
            rejected_cases.append({"id": case["id"], "input": case["input"], "reason": reason})
        else:
            accepted_cases.append(case)
    if not accepted_cases:
        detail = "; ".join(item["reason"] for item in rejected_cases[:5])
        raise SuiteError("The single test-generation response contained no valid test inputs after contract validation. " + detail)
    issues = answer.get("issues_found", [])
    history = [{
        "pass": 1,
        "status": "completed",
        "mode": "single_pass",
        "tests_added": len(accepted_cases),
        "rejected_tests": len(rejected_cases),
        "total_tests": len(accepted_cases),
        "review_summary": answer.get("review_summary", "")[:500] if isinstance(answer.get("review_summary"), str) else "",
        "issues_found": issues[:12] if isinstance(issues, list) else [],
    }]
    return accepted_cases, history, {
        "warnings": [f"Rejected candidate {item['id']}: {item['reason']}" for item in rejected_cases],
        "inferred_contract": inferred_contract,
        "rejected_generated_cases": rejected_cases,
    }


def _reference_source(llm: LLMClient, rules: dict[str, Any]) -> tuple[str, str]:
    configured = rules["test_generation"].get("reference_solution_path")
    if configured:
        candidate = Path(str(configured))
        if candidate.is_absolute():
            raise SuiteError("reference_solution_path must be relative to rules.json.")
        resolved = (rules["root"] / candidate).resolve()
        if not resolved.is_relative_to(rules["root"].resolve()) or not resolved.is_file():
            raise SuiteError("Configured reference_solution_path is missing or outside the project.")
        return resolved.read_text(encoding="utf-8"), "instructor_reference"

    system = (
        "Write a correct, minimal, complete C11 reference program for the instructor assignment. "
        "The reference is an oracle for test expected outputs. Infer the complete input/output contract from the "
        "natural-language assignment itself; separate format fields may be absent. Follow stated constraints, "
        "numeric semantics, whitespace tolerance and exact output formatting. Do not add unsupported validation behavior. "
        "Return only compilable C source code, preferably in one fenced c code block. No explanations."
    )
    instructor_examples = _load_instructor_cases(rules)[:10]
    bounded_examples = [{
        "id": item["id"], "input": item["input"][:1200], "expected": item["expected"][:1200]
    } for item in instructor_examples]
    user = json.dumps({
        "assignment": rules["problem"],
        "instructor_contract_if_any": rules["problem_contract"],
        "inferred_contract_from_test_design": rules.get("inferred_contract", {}),
        "instructor_examples": bounded_examples,
        "instruction": "Infer input parsing, output formatting, constraints, edge behavior and exact semantics from the assignment text. No external input/output format document exists. Implement only behavior required by the assignment."
    }, ensure_ascii=False, sort_keys=True)
    last_error = None
    for attempt in range(1, 3):
        try:
            prompt = user if attempt == 1 else user + "\nThe previous response did not compile. Correct this compile failure and return the complete source only."
            source = extract_c_code(llm.text(system, prompt, max_tokens=min(6000, llm.max_tokens + 1500)))
            return source, "llm_reference"
        except LLMError as exc:
            last_error = exc
    raise SuiteError(f"Could not obtain a complete C reference solution: {last_error}")


def _compile_reference(sandbox: DockerSandbox, source: str, rules: dict[str, Any]) -> None:
    sandbox.put_text("/work/reference.c", source)
    sandbox.lock_source("reference.c", "reference.c")
    result = sandbox.compile("reference.c", "reference", flags=["-std=c11", "-O2", "-pipe", "-fno-ident", "-fdiagnostics-color=never"], timeout=15)
    if result["timed_out"] or result["exit_code"] != 0:
        raise SuiteError("Reference C solution did not compile: " + (result["stderr"] + result["stdout"])[-2000:])
    sandbox.lock_binary("reference", "reference")


def _materialize_cases(sandbox: DockerSandbox, suite: list[dict[str, Any]], instructor_cases: list[dict[str, Any]], rules: dict[str, Any]) -> tuple[list[dict[str, Any]], bool, list[dict[str, Any]]]:
    all_inputs: dict[str, dict[str, Any]] = {}
    for case in instructor_cases:
        prior = all_inputs.get(case["input"])
        if prior and not _compare(prior.get("expected", ""), case["expected"], rules["output_comparison"]):
            raise SuiteError(f"Instructor cases disagree on expected output for input {case['input']!r}.")
        if prior:
            case = dict(prior, tags=sorted(set(prior.get("tags", [])) | set(case.get("tags", []))))
        all_inputs[case["input"]] = dict(case)

    for case in suite:
        if case["input"] not in all_inputs:
            all_inputs[case["input"]] = dict(case, origin="generated")

    results: list[dict[str, Any]] = []
    rejected_generated: list[dict[str, Any]] = []
    verified_examples = bool(instructor_cases)
    for index, case in enumerate(all_inputs.values(), 1):
        oracle = sandbox.run_program("reference", case["input"], timeout_s=rules["time_limit_s"])
        if oracle["timed_out"] or oracle["exit_code"] != 0 or oracle["output_truncated"]:
            failure = {
                "id": case.get("id", f"case_{index:02d}"),
                "reason": f"Reference oracle failed (exit={oracle['exit_code']}, timeout={oracle['timed_out']}, truncated={oracle['output_truncated']}).",
                "input": case["input"],
                "stderr": str(oracle.get("stderr", ""))[:1000],
            }
            if case.get("origin", "generated") == "generated":
                rejected_generated.append(failure)
                continue
            raise SuiteError(f"Instructor reference example {failure['id']} could not be evaluated by the reference program: {failure['reason']}")
        expected = oracle["stdout"]
        if "expected" in case:
            if _compare(expected, case["expected"], rules["output_comparison"]) is False:
                raise SuiteError(
                    f"Reference solution disagrees with instructor expected output for {case.get('id', index)}. "
                    f"Instructor expected={case['expected']!r}; reference produced={expected!r}. "
                    "Do not grade until the assignment/oracle is corrected."
                )
        results.append({
            "id": str(case.get("id", f"case_{index:02d}")),
            "input": case["input"],
            "expected": case.get("expected", expected),
            "tags": sorted(set(case.get("tags", ["core"]))),
            "description": case.get("description", "Generated from assignment contract."),
            "origin": case.get("origin", "generated"),
        })
    if not results:
        raise SuiteError("No test cases could be evaluated by the reference program.")
    return results, verified_examples, rejected_generated


def _compare(actual: str, expected: str, mode: str) -> bool:
    actual = actual.replace("\r\n", "\n").replace("\r", "\n")
    expected = expected.replace("\r\n", "\n").replace("\r", "\n")
    if mode == "tokens":
        return actual.split() == expected.split()
    if mode == "trimmed":
        return actual.strip() == expected.strip()
    return actual == expected


def prepare_suite(rules: dict[str, Any], sandbox: DockerSandbox, llm: LLMClient, *, refresh: bool = False) -> dict[str, Any]:
    cfg = rules["test_generation"]
    model_id = llm.model
    fingerprint = _fingerprint(rules, model_id, sandbox.image_id, getattr(llm, "seed", rules["llm"].get("seed", cfg["seed"])))
    cache_dir = (rules["root"] / cfg["cache_dir"]).resolve()
    if not cache_dir.is_relative_to(rules["root"].resolve()):
        raise SuiteError("test_generation.cache_dir must remain inside the project directory.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{fingerprint}.json"
    if cache_path.is_file() and not refresh:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                cached_source = cached.get("reference_source", "")
                cached_source = cached_source if isinstance(cached_source, str) else None
                source_hash_ok = cached_source is not None and hashlib.sha256(cached_source.encode("utf-8")).hexdigest() == cached.get("reference_sha256")
                expected_digest = _digest({"test_cases": cached.get("test_cases"), "reference_sha256": cached.get("reference_sha256"), "fingerprint": fingerprint})
                test_cases = cached.get("test_cases")
                test_cases_ok = isinstance(test_cases, list) and bool(test_cases) and all(
                    isinstance(item, dict) and isinstance(item.get("input"), str) and isinstance(item.get("expected"), str)
                    and isinstance(item.get("tags"), list) and all(isinstance(tag, str) for tag in item.get("tags", []))
                    for item in test_cases
                )
                if cached.get("fingerprint") == fingerprint and source_hash_ok and test_cases_ok and cached.get("suite_sha256") == expected_digest:
                    return cached
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            pass

    instructor_cases = _load_instructor_cases(rules)
    if cfg.get("enabled", True):
        try:
            generated, history, generation_meta = _generate_test_inputs(llm, rules)
        except SuiteError as exc:
            if not instructor_cases:
                raise
            generated, history = [], [{"pass": 1, "status": "failed_fallback_to_instructor", "error": str(exc), "total_tests": 0}]
            generation_meta = {"warnings": [f"Automatic test generation failed; grading will use instructor-provided I/O cases only: {exc}"]}
    else:
        generated, history, generation_meta = [], [], {"warnings": [], "contract_used": "Test generation disabled."}
        if not instructor_cases:
            raise SuiteError("Test generation is disabled and no instructor test cases were supplied.")

    reference_source, reference_origin = "", "instructor_expected_outputs"
    oracle_verified = False
    rejected_generated: list[dict[str, Any]] = list(generation_meta.get("rejected_generated_cases", []))
    if generated:
        try:
            oracle_rules = dict(rules)
            oracle_rules["inferred_contract"] = generation_meta.get("inferred_contract", {})
            reference_source, reference_origin = _reference_source(llm, oracle_rules)
            try:
                _compile_reference(sandbox, reference_source, rules)
            except SuiteError as compile_error:
                if reference_origin != "llm_reference":
                    raise
                repair_system = (
                    "You are repairing a C11 reference solution. Return only the complete corrected C source code. "
                    "The program must implement the assignment exactly; do not refer to the student submission."
                )
                repair_user = json.dumps({
                    "assignment": rules["problem"],
                    "contract": rules["problem_contract"],
                    "previous_reference_source": reference_source,
                    "compiler_error": str(compile_error),
                    "instruction": "Fix the compile error and return the entire translation unit, not a patch."
                }, ensure_ascii=False, sort_keys=True)
                try:
                    reference_source = extract_c_code(llm.text(repair_system, repair_user, max_tokens=min(6000, llm.max_tokens + 1500)))
                except LLMError as exc:
                    raise SuiteError(f"Reference repair request failed after compile error: {exc}") from exc
                _compile_reference(sandbox, reference_source, rules)
            cases, oracle_verified, oracle_rejected = _materialize_cases(sandbox, generated, instructor_cases, rules)
            rejected_generated.extend(oracle_rejected)
        except SuiteError as exc:
            if not instructor_cases:
                raise
            # Trusted instructor I/O cases remain usable even when the LLM test
            # suite/reference is malformed or contradicts the examples. Never
            # use unverified LLM-generated expected outputs as a fallback.
            generation_meta.setdefault("warnings", []).append(
                f"Generated cases were discarded because their reference oracle could not be validated; "
                f"using instructor-provided cases only: {exc}"
            )
            rejected_generated = [{
                "id": item.get("id", "generated"),
                "input": item["input"],
                "reason": "Discarded because no validated reference oracle was available.",
            } for item in generated]
            cases = [dict(item) for item in instructor_cases]
            reference_source, reference_origin, oracle_verified = "", "instructor_expected_outputs", False
        finally:
            # The oracle is deliberately destroyed before any student process
            # is compiled or executed. It must never be callable by a submission.
            sandbox.clear_workspaces()
    else:
        cases = [dict(item) for item in instructor_cases]
    suite = {
        "status": "completed",
        "fingerprint": fingerprint,
        "test_cases": cases,
        "test_count": len(cases),
        "generated_count": sum(item.get("origin") == "generated" for item in cases),
        "rejected_generated_count": len(rejected_generated),
        "rejected_generated_cases": rejected_generated,
        "instructor_count": len(instructor_cases),
        "rsi_history": history,
        "inferred_contract": generation_meta.get("inferred_contract", {}),
        "generation_warnings": generation_meta["warnings"] + [item["reason"] for item in rejected_generated],
        "reference_origin": reference_origin,
        "reference_source": reference_source,
        "reference_sha256": hashlib.sha256(reference_source.encode("utf-8")).hexdigest(),
        "oracle_crosschecked_with_instructor_examples": oracle_verified,
        "oracle_caveat": "A generated reference program is an executable interpretation of the statement, not a formal proof. Instructor examples are cross-checked when supplied.",
        "comparison_mode": rules["output_comparison"],
    }
    suite["suite_sha256"] = _digest({"test_cases": cases, "reference_sha256": suite["reference_sha256"], "fingerprint": fingerprint})
    # Atomic replacement prevents concurrent grading runs from leaving a
    # truncated JSON cache file if they finish at nearly the same time.
    fd, temp_name = tempfile.mkstemp(prefix=cache_path.name + ".", suffix=".tmp", dir=cache_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(suite, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, cache_path)
    finally:
        Path(temp_name).unlink(missing_ok=True)
    return suite
