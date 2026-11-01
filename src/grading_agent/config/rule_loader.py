"""Load and validate instructor-provided grading rules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from grading_agent.config import settings


SUPPORTED_ANALYZERS = {
    "io_tests",
    "edge_case_runner",
    "runtime_profile",
    "compiler_diagnostics",
    "static_analysis",
    "modularity",
    "complexity_estimator",
    "readability",
    "similarity_flag",
    "memory_safety",
    # Legacy alias: now normalized to the composite memory-safety metric.
    "valgrind_memcheck",
}


class RuleValidationError(ValueError):
    """Raised when an instructor rules file is invalid."""


def resolve_project_path(
    project_root: Path,
    relative_path: str,
    *,
    field_name: str,
    must_exist: bool = True,
) -> Path:
    """Resolve a project-relative path and prevent escaping project_root."""
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise RuleValidationError(f"'{field_name}' must be a non-empty path.")

    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise RuleValidationError(
            f"'{field_name}' must be relative to the project root."
        )

    root = project_root.resolve()
    resolved = (root / candidate).resolve()

    if not resolved.is_relative_to(root):
        raise RuleValidationError(
            f"'{field_name}' must not point outside the project root."
        )

    if must_exist and not resolved.is_file():
        raise RuleValidationError(
            f"File configured by '{field_name}' does not exist: "
            f"{relative_path}"
        )

    return resolved


def _positive_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RuleValidationError(
            f"'{field_name}' must be a finite positive number."
        )
    return float(value)


def _unit_interval_weight(value: Any, field_name: str) -> float:
    """Validate a finite weight/ratio in the inclusive [0, 1] interval."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise RuleValidationError(
            f"'{field_name}' must be a finite number between 0 and 1."
        )
    return float(value)


def _percentage_penalty(value: Any, field_name: str) -> float:
    """Validate a severity deduction in the inclusive [0, 100] interval."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 100
    ):
        raise RuleValidationError(
            f"'{field_name}' must be a finite number between 0 and 100."
        )
    return float(value)


def load_test_cases(path: Path) -> list[dict[str, Any]]:
    """Load and validate JSON input/expected-output test cases."""
    try:
        with path.open("r", encoding="utf-8") as file:
            cases = json.load(file)
    except json.JSONDecodeError as exc:
        raise RuleValidationError(
            f"Invalid test-case JSON in {path}: {exc}"
        ) from exc

    if not isinstance(cases, list) or not cases:
        raise RuleValidationError(
            f"Test-case file must contain a non-empty JSON array: {path}"
        )

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, case in enumerate(cases, start=1):
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("input"), str)
            or not isinstance(case.get("expected"), str)
        ):
            raise RuleValidationError(
                f"Test case {index} in {path} must contain string "
                "'input' and 'expected' fields."
            )
        case_id = case.get("id")
        if case_id is not None:
            if not isinstance(case_id, str) or not case_id.strip():
                raise RuleValidationError(f"Test case {index} in {path} has invalid 'id'; expected a non-empty string.")
            if case_id in seen_ids:
                raise RuleValidationError(f"Duplicate test case id '{case_id}' in {path}.")
            seen_ids.add(case_id)
        tags = case.get("tags", [])
        description = case.get("description", "")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise RuleValidationError(f"Test case {index} in {path} has invalid 'tags'; expected a list of strings.")
        if not isinstance(description, str):
            raise RuleValidationError(f"Test case {index} in {path} has invalid 'description'; expected a string.")
        validated_case: dict[str, Any] = {
            "input": case["input"],
            "expected": case["expected"],
        }
        if case_id is not None:
            validated_case["id"] = case_id
        if description:
            validated_case["description"] = description
        if tags:
            validated_case["tags"] = tags
        validated.append(validated_case)

    return validated


def load_rules(
    rules_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Load, normalize, and validate a grading rules JSON file.

    Supported identifiers cover I/O correctness, edge cases, runtime profiling,
    compiler diagnostics, cppcheck static analysis, the composite Valgrind +
    LLM memory-safety metric, modularity, complexity, readability, and source
    similarity. Legacy ``valgrind_memcheck`` is normalized to ``memory_safety``.
    """
    rules_path = rules_path.resolve()
    project_root = project_root.resolve()

    if not rules_path.is_file():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")

    try:
        with rules_path.open("r", encoding="utf-8") as file:
            rules = json.load(file)
    except json.JSONDecodeError as exc:
        raise RuleValidationError(
            f"Invalid rules JSON in {rules_path}: {exc}"
        ) from exc

    if not isinstance(rules, dict):
        raise RuleValidationError("The rules file must contain a JSON object.")

    problem = rules.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        raise RuleValidationError("'problem' must be a non-empty string.")

    time_limit_s = _positive_number(
        rules.get("time_limit_s"),
        "time_limit_s",
    )
    if time_limit_s > settings.DOCKER_MAX_TIMEOUT_S:
        raise RuleValidationError(
            f"'time_limit_s' cannot exceed the sandbox maximum "
            f"of {settings.DOCKER_MAX_TIMEOUT_S} seconds."
        )

    memory_limit_mb = rules.get("memory_limit_mb")
    if (
        isinstance(memory_limit_mb, bool)
        or not isinstance(memory_limit_mb, int)
        or memory_limit_mb < 1
    ):
        raise RuleValidationError(
            "'memory_limit_mb' must be a positive integer."
        )

    submission = rules.get("submission")
    if not isinstance(submission, dict):
        raise RuleValidationError("'submission' must be a JSON object.")

    language = submission.get("language")
    if language != "c":
        raise RuleValidationError(
            "This version supports only submission language 'c'."
        )

    source_path = resolve_project_path(
        project_root,
        submission.get("source_file"),
        field_name="submission.source_file",
    )
    if source_path.suffix.lower() != ".c":
        raise RuleValidationError(
            "'submission.source_file' must point to a .c file."
        )

    metrics = rules.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise RuleValidationError(
            "'metrics' must be a non-empty JSON array."
        )

    seen_names: set[str] = set()
    enabled_weights: list[float] = []
    normalized_metrics: list[dict[str, Any]] = []

    for index, metric in enumerate(metrics, start=1):
        if not isinstance(metric, dict):
            raise RuleValidationError(f"Metric {index} must be an object.")

        name = metric.get("name")
        analyzer = metric.get("analyzer")
        if not isinstance(name, str) or not name.strip():
            raise RuleValidationError(
                f"Metric {index} requires a non-empty 'name'."
            )
        if name in seen_names:
            raise RuleValidationError(f"Duplicate metric name: {name}")
        seen_names.add(name)

        if analyzer not in SUPPORTED_ANALYZERS:
            supported = ", ".join(sorted(SUPPORTED_ANALYZERS))
            raise RuleValidationError(
                f"Unsupported analyzer '{analyzer}' in metric '{name}'. "
                f"Currently supported: {supported}."
            )

        # Old configurations named the runtime-only check valgrind_memcheck.
        # Normalize it to the composite metric so both audits are now run.
        if analyzer == "valgrind_memcheck":
            analyzer = "memory_safety"

        enabled = metric.get("enabled", True)
        if not isinstance(enabled, bool):
            raise RuleValidationError(
                f"'enabled' for metric '{name}' must be boolean."
            )

        weight = metric.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or not 0 <= weight <= 1
        ):
            raise RuleValidationError(
                f"Weight for metric '{name}' must be between 0 and 1."
            )

        params = metric.get("params", {})
        if not isinstance(params, dict):
            raise RuleValidationError(
                f"'params' for metric '{name}' must be a JSON object."
            )

        normalized = {
            "name": name,
            "analyzer": analyzer,
            "weight": float(weight),
            "enabled": enabled,
            "params": dict(params),
        }

        if enabled:
            for timeout_key in ("timeout_s", "compile_timeout_s"):
                if timeout_key in params:
                    timeout_value = _positive_number(
                        params[timeout_key], f"{name}.params.{timeout_key}"
                    )
                    if timeout_value > settings.DOCKER_MAX_TIMEOUT_S:
                        raise RuleValidationError(
                            f"'{name}.params.{timeout_key}' cannot exceed "
                            f"{settings.DOCKER_MAX_TIMEOUT_S} seconds."
                        )
                    normalized["params"][timeout_key] = timeout_value

        if enabled:
            enabled_weights.append(float(weight))

        if enabled and analyzer in {"io_tests", "memory_safety", "edge_case_runner", "runtime_profile"}:
            testset_path = resolve_project_path(
                project_root,
                params.get("testset_path"),
                field_name=f"metrics[{index}].params.testset_path",
            )
            # Validate the JSON structure at configuration-loading time.
            load_test_cases(testset_path)
            # Keep the resolved path in params because the coordinator reads
            # it from there. Also retain the metric-level field for clarity.
            normalized["params"]["resolved_testset_path"] = str(testset_path)
            normalized["resolved_testset_path"] = str(testset_path)

        if enabled and analyzer == "similarity_flag":
            corpus_dir = params.get("corpus_dir")
            if not isinstance(corpus_dir, str) or not corpus_dir.strip():
                raise RuleValidationError(
                    f"'{name}.params.corpus_dir' must be a non-empty directory path."
                )
            candidate = Path(corpus_dir)
            if candidate.is_absolute():
                raise RuleValidationError("similarity corpus_dir must be project-relative.")
            resolved_dir = (project_root / candidate).resolve()
            if not resolved_dir.is_relative_to(project_root):
                raise RuleValidationError("similarity corpus_dir must not escape the project root.")
            if not resolved_dir.is_dir():
                raise RuleValidationError(f"Similarity corpus directory does not exist: {corpus_dir}")
            normalized["params"]["resolved_corpus_dir"] = str(resolved_dir)

        if enabled and analyzer == "runtime_profile":
            target_ms = params.get("target_runtime_ms", settings.ANALYZER_RUNTIME_TARGET_MS)
            if isinstance(target_ms, bool) or not isinstance(target_ms, (int, float)) or not math.isfinite(target_ms) or target_ms <= 0:
                raise RuleValidationError(f"'{name}.params.target_runtime_ms' must be positive.")
            repetitions = params.get("repetitions", settings.ANALYZER_RUNTIME_REPETITIONS)
            if isinstance(repetitions, bool) or not isinstance(repetitions, int) or not 1 <= repetitions <= settings.ANALYZER_RUNTIME_MAX_REPETITIONS:
                raise RuleValidationError(f"'{name}.params.repetitions' must be an integer from 1 to {settings.ANALYZER_RUNTIME_MAX_REPETITIONS}.")
            normalized["params"].update({"target_runtime_ms": float(target_ms), "repetitions": repetitions})
            normalized["params"].setdefault("compile_timeout_s", settings.ANALYZER_COMPILE_TIMEOUT_S)

        if enabled and analyzer == "edge_case_runner":
            normalized["params"].setdefault("compile_timeout_s", settings.ANALYZER_COMPILE_TIMEOUT_S)

        if enabled and analyzer == "modularity":
            min_functions = params.get("min_functions", settings.ANALYZER_MIN_FUNCTIONS)
            expect_helpers = params.get("expect_helpers", settings.ANALYZER_EXPECT_HELPERS)
            if isinstance(min_functions, bool) or not isinstance(min_functions, int) or min_functions < 1:
                raise RuleValidationError(f"'{name}.params.min_functions' must be a positive integer.")
            if not isinstance(expect_helpers, bool):
                raise RuleValidationError(f"'{name}.params.expect_helpers' must be boolean.")
            normalized["params"].update({"min_functions": min_functions, "expect_helpers": expect_helpers})

        if enabled and analyzer == "complexity_estimator":
            for key, default in (("max_cyclomatic", settings.ANALYZER_MAX_CYCLOMATIC), ("max_function_length", settings.ANALYZER_MAX_FUNCTION_LENGTH)):
                value = params.get(key, default)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise RuleValidationError(f"'{name}.params.{key}' must be a positive integer.")
                normalized["params"][key] = value
            for key, default in (("complexity_penalty", settings.ANALYZER_COMPLEXITY_PENALTY), ("length_penalty_per_line", settings.ANALYZER_LENGTH_PENALTY_PER_LINE)):
                normalized["params"][key] = _percentage_penalty(
                    params.get(key, default), f"{name}.params.{key}"
                )

        if enabled and analyzer == "readability":
            max_line_length = params.get("max_line_length", settings.ANALYZER_MAX_LINE_LENGTH)
            if isinstance(max_line_length, bool) or not isinstance(max_line_length, int) or max_line_length < 20:
                raise RuleValidationError(f"'{name}.params.max_line_length' must be an integer >= 20.")
            normalized["params"]["max_line_length"] = max_line_length
            for key, default in (
                ("long_line_penalty", settings.ANALYZER_LONG_LINE_PENALTY),
                ("single_char_identifier_penalty", settings.ANALYZER_SINGLE_CHAR_IDENTIFIER_PENALTY),
                ("missing_comment_penalty", settings.ANALYZER_MISSING_COMMENT_PENALTY),
            ):
                normalized["params"][key] = _percentage_penalty(
                    params.get(key, default), f"{name}.params.{key}"
                )

        if enabled and analyzer == "compiler_diagnostics":
            flags = params.get("flags", list(settings.ANALYZER_COMPILER_FLAGS))
            if not isinstance(flags, list) or not all(isinstance(item, str) for item in flags):
                raise RuleValidationError(f"'{name}.params.flags' must be a list of strings.")
            normalized["params"]["flags"] = flags
            normalized["params"]["warning_penalty"] = _percentage_penalty(
                params.get("warning_penalty", settings.ANALYZER_COMPILER_WARNING_PENALTY), f"{name}.params.warning_penalty"
            )
            normalized["params"].setdefault("timeout_s", settings.ANALYZER_COMPILER_TIMEOUT_S)

        if enabled and analyzer == "static_analysis":
            supplied = params.get("severity_penalties", {})
            allowed_cppcheck_severities = {"error", "warning", "style", "performance", "portability", "information"}
            if not isinstance(supplied, dict):
                raise RuleValidationError(f"'{name}.params.severity_penalties' must be an object.")
            unknown = set(supplied) - allowed_cppcheck_severities
            if unknown:
                raise RuleValidationError(f"Unknown cppcheck severities: {', '.join(sorted(unknown))}.")
            normalized["params"]["severity_penalties"] = {
                severity: _percentage_penalty(value, f"{name}.params.severity_penalties.{severity}")
                for severity, value in supplied.items()
            }
            normalized["params"].setdefault("timeout_s", settings.ANALYZER_STATIC_TIMEOUT_S)

        if enabled and analyzer == "similarity_flag":
            threshold = _unit_interval_weight(params.get("flag_threshold", settings.ANALYZER_SIMILARITY_THRESHOLD), f"{name}.params.flag_threshold")
            max_files = params.get("max_files", settings.ANALYZER_SIMILARITY_MAX_FILES)
            shingle_size = params.get("shingle_size", settings.ANALYZER_SIMILARITY_SHINGLE_SIZE)
            if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files < 1:
                raise RuleValidationError(f"'{name}.params.max_files' must be positive.")
            if isinstance(shingle_size, bool) or not isinstance(shingle_size, int) or shingle_size < settings.ANALYZER_SIMILARITY_MIN_SHINGLE_SIZE:
                raise RuleValidationError(f"'{name}.params.shingle_size' must be >= {settings.ANALYZER_SIMILARITY_MIN_SHINGLE_SIZE}.")
            normalized["params"].update({"flag_threshold": threshold, "max_files": max_files, "shingle_size": shingle_size})


        normalized_metrics.append(normalized)

    if not enabled_weights:
        raise RuleValidationError("At least one metric must be enabled.")

    if not math.isclose(sum(enabled_weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuleValidationError(
            "Weights of enabled metrics must sum to 1.0; "
            f"received {sum(enabled_weights):.6f}."
        )

    problem_contract = rules.get("problem_contract", {})
    if not isinstance(problem_contract, dict):
        raise RuleValidationError("'problem_contract' must be a JSON object when provided.")
    for key in ("description", "operand_type", "sum_type", "input_range", "output_rule"):
        value = problem_contract.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise RuleValidationError(f"'problem_contract.{key}' must be a non-empty string when provided.")

    normalized_rules = dict(rules)
    normalized_rules["problem_contract"] = dict(problem_contract)
    normalized_rules["time_limit_s"] = time_limit_s
    normalized_rules["memory_limit_mb"] = memory_limit_mb
    normalized_rules["submission"] = {
        **submission,
        "resolved_source_path": str(source_path),
    }
    for metric in normalized_metrics:
        if metric["analyzer"] == "memory_safety":
            run_sanitizers = metric["params"].get("run_sanitizers", True)
            if not isinstance(run_sanitizers, bool):
                raise RuleValidationError(
                    f"'{metric['name']}.params.run_sanitizers' must be boolean."
                )
            metric["params"]["run_sanitizers"] = run_sanitizers

    normalized_rules["metrics"] = normalized_metrics
    # Ignore the old experimental analyzer/LLM blend setting. Metric scores
    # now come from RSI, with analyzer scores used only as fallback.
    normalized_rules.pop("scoring", None)
    normalized_rules["rules_path"] = str(rules_path)
    normalized_rules["project_root"] = str(project_root)

    return normalized_rules
