"""Registry that dispatches instructor-selected analyzers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from grading_agent.config.rule_loader import load_test_cases


@dataclass
class AnalyzerContext:
    """Shared inputs passed to an analyzer implementation."""

    sandbox: Any
    metric: dict[str, Any]
    rules: dict[str, Any]
    source_code: str
    source_filename: str
    test_case_cache: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    @property
    def params(self) -> dict[str, Any]:
        return self.metric["params"]

    def get_test_cases(self) -> list[dict[str, str]]:
        """Load a configured test set once and reuse it across analyzers."""
        resolved_path = self.params.get("resolved_testset_path")
        if not resolved_path:
            raise ValueError(
                f"Analyzer '{self.metric['analyzer']}' requires a resolved testset_path."
            )

        path = Path(resolved_path)
        cache_key = str(path)
        if cache_key not in self.test_case_cache:
            self.test_case_cache[cache_key] = load_test_cases(path)
        return self.test_case_cache[cache_key]


def _run_io_tests(context: AnalyzerContext) -> dict[str, Any]:
    from grading_agent.analyzers.io_tests import run_io_tests

    return run_io_tests(
        sandbox=context.sandbox,
        source_code=context.source_code,
        test_cases=context.get_test_cases(),
        filename=context.source_filename,
        compile_timeout_s=context.params.get("compile_timeout_s", 10),
        run_timeout_s=context.rules["time_limit_s"],
    )


def _run_memory_safety(context: AnalyzerContext) -> dict[str, Any]:
    from grading_agent.analyzers.memory_safety import run_memory_safety

    result = run_memory_safety(
        sandbox=context.sandbox,
        source_code=context.source_code,
        test_cases=context.get_test_cases(),
        filename=context.source_filename,
        compile_timeout_s=context.params.get("compile_timeout_s", 10),
        run_timeout_s=context.rules["time_limit_s"],
    )

    # The instructor sees one memory_safety metric. The LLM half runs later,
    # once the local model server is available; both components are then
    # combined by the deterministic memory-safety scoring helper.
    result["component_config"] = {
        "valgrind_weight": context.params["valgrind_weight"],
        "llm_audit_weight": context.params["llm_audit_weight"],
        "severity_penalties": context.params["severity_penalties"],
        "llm_audit_max_findings": context.params["llm_audit_max_findings"],
    }
    return result


def _run_static_analysis(context: AnalyzerContext) -> dict[str, Any]:
    from grading_agent.analyzers.static_analysis import run_static_analysis

    return run_static_analysis(
        sandbox=context.sandbox,
        source_code=context.source_code,
        filename=context.source_filename,
        timeout_s=context.params.get(
            "timeout_s",
            context.rules["time_limit_s"],
        ),
        warning_penalty=context.params.get("warning_penalty", 0.1),
    )


ANALYZER_REGISTRY: dict[str, Callable[[AnalyzerContext], dict[str, Any]]] = {
    "io_tests": _run_io_tests,
    "memory_safety": _run_memory_safety,
    "static_analysis": _run_static_analysis,
}


def run_analyzer(
    analyzer_name: str,
    *,
    sandbox: DockerSandbox,
    metric: dict[str, Any],
    rules: dict[str, Any],
    source_code: str,
    source_filename: str,
    test_case_cache: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """Dispatch one validated metric to its registered implementation."""
    try:
        handler = ANALYZER_REGISTRY[analyzer_name]
    except KeyError as exc:
        available = ", ".join(sorted(ANALYZER_REGISTRY))
        raise ValueError(
            f"No analyzer registered as '{analyzer_name}'. "
            f"Registered analyzers: {available}."
        ) from exc

    context = AnalyzerContext(
        sandbox=sandbox,
        metric=metric,
        rules=rules,
        source_code=source_code,
        source_filename=source_filename,
        test_case_cache=test_case_cache if test_case_cache is not None else {},
    )
    result = handler(context)
    result["metric_name"] = metric["name"]
    result["analyzer"] = analyzer_name
    result["weight"] = metric["weight"]
    return result
