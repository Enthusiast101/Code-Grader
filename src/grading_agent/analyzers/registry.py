"""Registry mapping instructor-selected metrics to analyzer implementations."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from grading_agent.config.rule_loader import load_test_cases


@dataclass
class AnalyzerContext:
    sandbox: Any
    metric: dict[str, Any]
    rules: dict[str, Any]
    source_code: str
    source_filename: str
    test_case_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def params(self) -> dict[str, Any]:
        return self.metric["params"]

    def get_test_cases(self) -> list[dict[str, Any]]:
        resolved = self.params.get("resolved_testset_path")
        if not resolved:
            raise ValueError(f"Analyzer '{self.metric['analyzer']}' requires params.testset_path.")
        path = Path(resolved)
        key = str(path)
        if key not in self.test_case_cache:
            self.test_case_cache[key] = load_test_cases(path)
        return self.test_case_cache[key]


def _run_io(context: AnalyzerContext) -> dict[str, Any]:
    from grading_agent.analyzers.io_tests import run_io_tests
    return run_io_tests(
        context.sandbox, context.source_code, context.get_test_cases(),
        filename=context.source_filename,
        compile_timeout_s=context.params.get("compile_timeout_s", 10),
        run_timeout_s=context.rules["time_limit_s"],
    )


def _run_edge(context: AnalyzerContext) -> dict[str, Any]:
    from grading_agent.analyzers.edge_cases import run_edge_case_runner
    return run_edge_case_runner(
        context.sandbox, context.source_code, context.get_test_cases(),
        filename=context.source_filename,
        compile_timeout_s=context.params.get("compile_timeout_s", 10),
        run_timeout_s=context.rules["time_limit_s"],
    )


def _run_runtime(context: AnalyzerContext) -> dict[str, Any]:
    from grading_agent.analyzers.runtime_profile import run_runtime_profile
    return run_runtime_profile(
        context.sandbox, context.source_code, context.get_test_cases(),
        filename=context.source_filename,
        compile_timeout_s=context.params.get("compile_timeout_s", 10),
        run_timeout_s=context.rules["time_limit_s"],
        target_runtime_ms=context.params.get("target_runtime_ms", 50.0),
        repetitions=context.params.get("repetitions", 3),
    )


def _run_compiler(context: AnalyzerContext) -> dict[str, Any]:
    from grading_agent.analyzers.compiler_diagnostics import run_compiler_diagnostics
    return run_compiler_diagnostics(
        context.sandbox, context.source_code,
        filename=context.source_filename,
        timeout_s=context.params.get("timeout_s", 10),
        flags=context.params.get("flags"),
        warning_penalty=context.params.get("warning_penalty", 5.0),
    )


def _run_static(context: AnalyzerContext) -> dict[str, Any]:
    from grading_agent.analyzers.static_analysis import run_static_analysis
    return run_static_analysis(
        context.sandbox, context.source_code,
        filename=context.source_filename,
        timeout_s=context.params.get("timeout_s", 15),
        severity_penalties=context.params.get("severity_penalties"),
    )


def _run_memory(context: AnalyzerContext) -> dict[str, Any]:
    from grading_agent.analyzers.memory_safety import run_memory_safety
    result = run_memory_safety(
        context.sandbox, context.source_code, context.get_test_cases(),
        filename=context.source_filename,
        compile_timeout_s=context.params.get("compile_timeout_s", 10),
        run_timeout_s=context.rules["time_limit_s"],
        run_sanitizers=context.params.get("run_sanitizers", True),
    )
    result["component_config"] = {
        "valgrind_weight": context.params["valgrind_weight"],
        "llm_audit_weight": context.params["llm_audit_weight"],
        "severity_penalties": context.params["severity_penalties"],
        "llm_audit_max_findings": context.params["llm_audit_max_findings"],
    }
    return result


def _run_modularity(context: AnalyzerContext) -> dict[str, Any]:
    from grading_agent.analyzers.modularity import run_modularity
    return run_modularity(
        context.source_code,
        min_functions=context.params.get("min_functions", 2),
        expect_helpers=context.params.get("expect_helpers", True),
    )


def _run_complexity(context: AnalyzerContext) -> dict[str, Any]:
    from grading_agent.analyzers.complexity_estimator import run_complexity_estimator
    return run_complexity_estimator(
        context.source_code,
        max_cyclomatic=context.params.get("max_cyclomatic", 8),
        max_function_length=context.params.get("max_function_length", 40),
        complexity_penalty=context.params.get("complexity_penalty", 8.0),
        length_penalty_per_line=context.params.get("length_penalty_per_line", 0.5),
    )


def _run_readability(context: AnalyzerContext) -> dict[str, Any]:
    from grading_agent.analyzers.readability import run_readability
    return run_readability(
        context.source_code,
        max_line_length=context.params.get("max_line_length", 100),
        long_line_penalty=context.params.get("long_line_penalty", 1.0),
        single_char_identifier_penalty=context.params.get("single_char_identifier_penalty", 1.0),
        missing_comment_penalty=context.params.get("missing_comment_penalty", 5.0),
    )


def _run_similarity(context: AnalyzerContext) -> dict[str, Any]:
    from grading_agent.analyzers.similarity import run_similarity_flag
    return run_similarity_flag(
        context.source_code,
        context.params["resolved_corpus_dir"],
        flag_threshold=context.params.get("flag_threshold", 0.85),
        max_files=context.params.get("max_files", 5000),
        shingle_size=context.params.get("shingle_size", 5),
        exclude_path=context.rules["submission"]["resolved_source_path"],
    )


ANALYZER_REGISTRY: dict[str, Callable[[AnalyzerContext], dict[str, Any]]] = {
    "io_tests": _run_io,
    "edge_case_runner": _run_edge,
    "runtime_profile": _run_runtime,
    "compiler_diagnostics": _run_compiler,
    "static_analysis": _run_static,
    "memory_safety": _run_memory,
    "modularity": _run_modularity,
    "complexity_estimator": _run_complexity,
    "readability": _run_readability,
    "similarity_flag": _run_similarity,
}


def run_analyzer(
    analyzer_name: str,
    *,
    sandbox: Any,
    metric: dict[str, Any],
    rules: dict[str, Any],
    source_code: str,
    source_filename: str,
    test_case_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    try:
        handler = ANALYZER_REGISTRY[analyzer_name]
    except KeyError as exc:
        raise ValueError(
            f"No analyzer registered as '{analyzer_name}'. Registered analyzers: {', '.join(sorted(ANALYZER_REGISTRY))}."
        ) from exc
    context = AnalyzerContext(
        sandbox=sandbox, metric=metric, rules=rules,
        source_code=source_code, source_filename=source_filename,
        test_case_cache=test_case_cache if test_case_cache is not None else {},
    )
    result = handler(context)
    result["metric_name"] = metric["name"]
    result["analyzer"] = analyzer_name
    result["weight"] = metric["weight"]
    return result
