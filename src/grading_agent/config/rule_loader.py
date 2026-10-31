"""Load and validate instructor-provided grading rules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from grading_agent.config import settings


SUPPORTED_ANALYZERS = {
    "io_tests",
    "memory_safety",
    "static_analysis",
    # Backward-compatible alias. It is normalized to the composite metric so
    # enabling memory safety always runs both Valgrind and the LLM source audit.
    "valgrind_memcheck",
}

MEMORY_SAFETY_SEVERITIES = {"critical", "high", "medium", "low"}


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


def load_test_cases(path: Path) -> list[dict[str, str]]:
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

    validated: list[dict[str, str]] = []
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
        validated.append({
            "input": case["input"],
            "expected": case["expected"],
        })

    return validated


def load_rules(
    rules_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Load, normalize, and validate a grading rules JSON file.

    Supported analyzer identifiers currently map to implemented modules:
    ``io_tests``, ``valgrind_memcheck``, and ``static_analysis``.
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
            enabled_weights.append(float(weight))

        if enabled and analyzer in {"io_tests", "memory_safety"}:
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

        if enabled and analyzer == "memory_safety":
            # Both component weights have defaults. If only one is supplied,
            # the other is its complement so their sum remains exactly 1.
            valgrind_weight = params.get("valgrind_weight")
            llm_weight = params.get("llm_audit_weight")

            if valgrind_weight is None and llm_weight is None:
                valgrind_weight = settings.MEMORY_SAFETY_DEFAULT_VALGRIND_WEIGHT
                llm_weight = settings.MEMORY_SAFETY_DEFAULT_LLM_WEIGHT
            elif valgrind_weight is None:
                llm_weight = _unit_interval_weight(
                    llm_weight, f"{name}.params.llm_audit_weight"
                )
                valgrind_weight = 1.0 - llm_weight
            elif llm_weight is None:
                valgrind_weight = _unit_interval_weight(
                    valgrind_weight, f"{name}.params.valgrind_weight"
                )
                llm_weight = 1.0 - valgrind_weight
            else:
                valgrind_weight = _unit_interval_weight(
                    valgrind_weight, f"{name}.params.valgrind_weight"
                )
                llm_weight = _unit_interval_weight(
                    llm_weight, f"{name}.params.llm_audit_weight"
                )

            if valgrind_weight <= 0 or llm_weight <= 0:
                raise RuleValidationError(
                    f"Both Valgrind and LLM audit weights for '{name}' "
                    "must be greater than zero so both checks contribute."
                )
            if not math.isclose(
                valgrind_weight + llm_weight,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise RuleValidationError(
                    f"'{name}' component weights must sum to 1.0; received "
                    f"{valgrind_weight + llm_weight:.6f}."
                )

            penalties = dict(
                settings.MEMORY_SAFETY_DEFAULT_SEVERITY_PENALTIES
            )
            supplied_penalties = params.get("severity_penalties", {})
            if not isinstance(supplied_penalties, dict):
                raise RuleValidationError(
                    f"'{name}.params.severity_penalties' must be an object."
                )
            unknown_severities = set(supplied_penalties) - MEMORY_SAFETY_SEVERITIES
            if unknown_severities:
                raise RuleValidationError(
                    f"Unknown severity penalty keys for '{name}': "
                    f"{', '.join(sorted(unknown_severities))}."
                )
            for severity, penalty in supplied_penalties.items():
                penalties[severity] = _percentage_penalty(
                    penalty,
                    f"{name}.params.severity_penalties.{severity}",
                )

            max_findings = params.get(
                "llm_audit_max_findings",
                settings.MEMORY_SAFETY_LLM_MAX_FINDINGS,
            )
            if (
                isinstance(max_findings, bool)
                or not isinstance(max_findings, int)
                or max_findings < 1
            ):
                raise RuleValidationError(
                    f"'{name}.params.llm_audit_max_findings' must be a positive integer."
                )

            normalized["params"].update({
                "valgrind_weight": float(valgrind_weight),
                "llm_audit_weight": float(llm_weight),
                "severity_penalties": penalties,
                "llm_audit_max_findings": max_findings,
            })

        normalized_metrics.append(normalized)

    if not enabled_weights:
        raise RuleValidationError("At least one metric must be enabled.")

    if not math.isclose(sum(enabled_weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuleValidationError(
            "Weights of enabled metrics must sum to 1.0; "
            f"received {sum(enabled_weights):.6f}."
        )

    normalized_rules = dict(rules)
    normalized_rules["time_limit_s"] = time_limit_s
    normalized_rules["memory_limit_mb"] = memory_limit_mb
    normalized_rules["submission"] = {
        **submission,
        "resolved_source_path": str(source_path),
    }
    normalized_rules["metrics"] = normalized_metrics
    normalized_rules["rules_path"] = str(rules_path)
    normalized_rules["project_root"] = str(project_root)

    return normalized_rules
