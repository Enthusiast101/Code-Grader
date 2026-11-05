"""Application entry point for rule-driven C grading."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from grading_agent.analyzers.registry import run_analyzer
from grading_agent.config.rule_loader import load_rules
from grading_agent.llm.reviewer import evaluate_metrics_with_llm, build_fallback_llm_evaluation
from grading_agent.llm.evidence import build_evidence_ledger
from grading_agent.scoring.overall import apply_metric_scores, calculate_overall_score
from grading_agent.reporting.text_report import render_text_report
from grading_agent.llm.server import initiate_server
from grading_agent.config import settings
from grading_agent.preflight import run_preflight

logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT_APP)
logger = logging.getLogger(__name__)

PROJECT_ROOT = settings.PROJECT_ROOT
RULES_PATH = settings.DEFAULT_RULES_PATH


def _is_llm_server_ready() -> bool:
    """Return True when a compatible local llama-server is healthy."""
    import requests

    try:
        response = requests.get(
            settings.LLAMA_HEALTH_URL,
            timeout=settings.LLAMA_HEALTH_TIMEOUT_S,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def _stop_llm_server(process: Any | None) -> None:
    """Stop only the server process started by this application."""
    if process is None or process.poll() is not None:
        return

    logger.info("Stopping local Llama server...")
    process.terminate()
    try:
        process.wait(timeout=settings.LLAMA_SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _review_with_local_llm(
    report: dict[str, Any],
    *,
    rules: dict[str, Any],
) -> dict[str, Any]:
    """Start/reuse the local model for program understanding and global RSI."""
    process = None
    setup_started = time.perf_counter()
    try:
        if not _is_llm_server_ready():
            logger.info("Starting local Llama server...")
            process = initiate_server()
        else:
            logger.info("Using an already-running local Llama server.")
        server_setup_seconds = round(time.perf_counter() - setup_started, 3)
        result = evaluate_metrics_with_llm(
            report, rules,
            max_iterations=settings.runtime_value("LLM_REVIEW_MAX_ITERATIONS"),
        )
        result["server_setup_seconds"] = server_setup_seconds
        return result
    finally:
        _stop_llm_server(process)



def _collect_tool_versions(sandbox: DockerSandbox) -> dict[str, str]:
    """Record tool versions used for reproducibility/audit metadata."""
    versions: dict[str, str] = {}
    for tool, command in {
        "gcc": ["gcc", "--version"],
        "cppcheck": ["cppcheck", "--version"],
        "valgrind": ["valgrind", "--version"],
    }.items():
        try:
            result = sandbox.run(command, timeout_s=settings.ANALYZER_TOOL_VERSION_TIMEOUT_S)
            versions[tool] = result["output"].splitlines()[0] if result["output"] else f"unavailable (exit {result['exit_code']})"
        except Exception as exc:
            versions[tool] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        versions["lizard"] = version("lizard")
    except PackageNotFoundError:
        versions["lizard"] = "not-installed (lexical fallback active)"
    return versions


DOCKER_ANALYZERS = {
    "io_tests", "edge_case_runner", "runtime_profile", "compiler_diagnostics",
    "static_analysis", "memory_safety",
}
EXECUTION_ANALYZERS = {"io_tests", "edge_case_runner", "runtime_profile", "memory_safety"}


def _unavailable_metric(error: str) -> dict[str, Any]:
    """Create a structured non-score result while keeping the report complete."""
    result: dict[str, Any] = {
        "status": "unavailable",
        "score": None,
        "score_scale": "0-100",
        "reason": error,
        "analysis_error": True,
    }
    return result


def run_configured_analyzers(
    rules: dict[str, Any],
) -> dict[str, Any]:
    """Run each enabled analyzer independently; unavailable Docker never blocks LLM.

    Source-only analyzers still run without Docker. Docker-dependent metrics get
    explicit unavailable/inconclusive records, and all LLM work is performed
    later by the coordinator regardless of those records.
    """
    analyzer_pipeline_started = time.perf_counter()
    source_path = Path(rules["submission"]["resolved_source_path"])
    source_code = source_path.read_text(encoding="utf-8")
    source_filename = source_path.name
    test_case_cache: dict[str, list[dict[str, Any]]] = {}
    enabled_metrics = [metric for metric in rules["metrics"] if metric["enabled"]]
    docker_needed = any(metric["analyzer"] in DOCKER_ANALYZERS for metric in enabled_metrics)

    from grading_agent.analyzers.c_ast import analyze_c_source

    report: dict[str, Any] = {
        "submission": source_filename,
        "source_code": source_code,
        # Deterministic syntax-tree facts (pycparser), shared by test design,
        # the evidence ledger and every model grading stage.
        "syntax_tree": analyze_c_source(source_code),
        "source_sha256": hashlib.sha256(source_code.encode("utf-8")).hexdigest(),
        "problem": rules["problem"],
        "rules_hash": hashlib.sha256(Path(rules["rules_path"]).read_bytes()).hexdigest(),
        "configuration": {
            "time_limit_s": rules["time_limit_s"],
            "memory_limit_mb": rules["memory_limit_mb"],
            "problem_contract": rules.get("problem_contract", {}),
            "assignment_statement": rules.get("problem", ""),
            "enabled_metrics": [
                {
                    "name": metric["name"],
                    "analyzer": metric["analyzer"],
                    "weight": metric["weight"],
                    "max_points": metric.get("max_points", metric["weight"] * 100.0),
                    "grade_contribution": metric.get("grade_contribution", True),
                }
                for metric in enabled_metrics
            ],
        },
    }

    sandbox_manager = None
    sandbox = None
    sandbox_error: str | None = None
    if docker_needed:
        try:
            # Lazy import: a missing Docker SDK/daemon is a metric-level outage,
            # not an import-time failure that prevents LLM source review.
            from grading_agent.sandbox.docker_sandbox import DockerSandbox
            sandbox_manager = DockerSandbox(
                memory_limit=f'{rules["memory_limit_mb"]}m',
                cpu_limit=settings.runtime_value("DOCKER_CPU_LIMIT"),
            )
            sandbox = sandbox_manager.__enter__()
            report["sandbox"] = {"status": "available", "image": sandbox_manager.image}
        except Exception as exc:
            sandbox = None
            sandbox_error = f"{type(exc).__name__}: {exc}"
            report["sandbox"] = {"status": "unavailable", "error": sandbox_error}
            logger.warning("Docker sandbox unavailable; source analyzers and LLM RSI will still run: %s", sandbox_error)

    try:
        if sandbox is not None:
            try:
                report["tool_versions"] = _collect_tool_versions(sandbox)
                report["preflight"] = run_preflight(
                    sandbox,
                    source_code=source_code,
                    source_filename=source_filename,
                    rules=rules,
                )
            except Exception as exc:
                preflight_error = f"{type(exc).__name__}: {exc}"
                report["preflight"] = {
                    "status": "unavailable", "compiled": None,
                    "smoke_tested": False, "error": preflight_error,
                }
                logger.warning("Preflight failed; individual analyzers will be attempted independently: %s", preflight_error)
        elif docker_needed:
            report["tool_versions"] = {tool: f"unavailable: {sandbox_error}" for tool in ("gcc", "cppcheck", "valgrind")}
            report["preflight"] = {
                "status": "unavailable", "compiled": None,
                "smoke_tested": False,
                "error": sandbox_error or "Docker sandbox unavailable.",
            }
        else:
            report["sandbox"] = {"status": "not_required"}
            report["preflight"] = {"status": "not_run", "compiled": None, "smoke_tested": False}
            report["tool_versions"] = {}

        execution_blocked = report["preflight"].get("status") in {
            "compile_failed", "compile_timeout", "timeout", "runtime_failure"
        }

        # Generate a reproducible assignment-level suite from the natural-language
        # problem statement before running execution analyzers, unless the UI has
        # already generated and explicitly attached a validated suite. In the UI
        # workflow generation is a separate user action; grading must not call
        # the generator a second time.
        report["test_generation"] = {"status": "not_required"}
        testing_cfg = rules.get("test_generation", {})
        pre_generated_suite = (
            testing_cfg.get("pre_generated_suite")
            if isinstance(testing_cfg, dict) else None
        )
        if isinstance(pre_generated_suite, dict) and pre_generated_suite.get("status") == "completed":
            report["test_generation"] = {
                **pre_generated_suite,
                "status": "completed",
                "source": "pre_generated_by_ui",
                "request_events": [],
            }
            report["test_generation"].setdefault("suite_path", None)
            missing_bindings = [
                str(metric.get("name", metric.get("analyzer", "unknown")))
                for metric in enabled_metrics
                if metric.get("analyzer") in EXECUTION_ANALYZERS
                and not metric.get("params", {}).get("resolved_testset_path")
            ]
            if missing_bindings:
                report["test_generation"].update({
                    "status": "unavailable",
                    "error": "The pre-generated suite is missing testset bindings for: " + ", ".join(missing_bindings),
                })
        elif sandbox is not None and isinstance(testing_cfg, dict) and testing_cfg.get("enabled", True):
            execution_metrics_present = any(
                item.get("enabled", True) and item.get("analyzer") in
                {"io_tests", "edge_case_runner", "memory_safety", "runtime_profile"}
                for item in enabled_metrics
            )
            if execution_metrics_present:
                generation_process = None
                generation_request_events: list[dict[str, Any]] = []
                request_event_token = None
                try:
                    from grading_agent.testing.test_generation import prepare_assignment_test_suite
                    from grading_agent.llm.reviewer import _LLM_REQUEST_EVENTS, _request_json
                    request_event_token = _LLM_REQUEST_EVENTS.set(generation_request_events)
                    if not _is_llm_server_ready():
                        generation_process = initiate_server()

                    requested_test_cases = testing_cfg.get("max_cases", 12)
                    try:
                        requested_test_cases = max(1, int(requested_test_cases))
                    except (TypeError, ValueError):
                        requested_test_cases = 12
                    # Size the single test-generation response from the requested
                    # suite size. Invalid candidates are discarded, not replaced
                    # through additional generation/RSI passes.
                    generation_token_budget = max(
                        int(settings.runtime_value("LLM_GLOBAL_SCORE_MAX_TOKENS") or 0),
                        min(settings.LLAMA_CONTEXT_SIZE, 2048 + requested_test_cases * 192),
                    )

                    def _test_generation_request(messages, schema, stage_name):
                        parsed, _raw = _request_json(
                            messages,
                            max_tokens=generation_token_budget,
                            enable_thinking=None,
                            response_schema=schema,
                            schema_name=stage_name,
                            fit_context=True,
                        )
                        return parsed

                    report["test_generation"] = prepare_assignment_test_suite(
                        rules, sandbox=sandbox, request_json=_test_generation_request,
                        project_root=Path(rules.get("project_root") or Path(rules["rules_path"]).resolve().parent),
                        model_id=str(settings.runtime_value("MODEL")),
                        generation_seed=settings.SEED,
                        temperature=float(settings.runtime_value("TEMPERATURE")),
                        submission_source=source_code,
                        submission_filename=source_filename,
                    )
                    report["test_generation"]["request_events"] = generation_request_events
                except Exception as exc:
                    report["test_generation"] = {
                        "status": "unavailable",
                        "error": f"{type(exc).__name__}: {exc}",
                        "request_events": generation_request_events,
                    }
                    logger.warning("Assignment test generation unavailable; existing instructor testsets remain in use: %s", exc)
                finally:
                    if request_event_token is not None:
                        _LLM_REQUEST_EVENTS.reset(request_event_token)
                    _stop_llm_server(generation_process)
        elif docker_needed and sandbox is None:
            report["test_generation"] = {"status": "unavailable", "reason": sandbox_error or "Docker sandbox unavailable."}

        # Runtime profiling needs valid inputs, but it does not need expected
        # outputs or a generated reference solution. If automatic generation
        # fails (for example, because output formatting is ambiguous), reuse
        # an instructor-supplied correctness/edge-case input set instead of
        # marking runtime efficiency unmeasured.
        runtime_metrics = [
            item for item in enabled_metrics
            if item.get("enabled", True) and item.get("analyzer") == "runtime_profile"
        ]
        reusable_input_metrics = [
            item for analyzer in ("io_tests", "edge_case_runner", "memory_safety")
            for item in enabled_metrics
            if item.get("enabled", True)
            and item.get("analyzer") == analyzer
            and item.get("params", {}).get("resolved_testset_path")
        ]
        for runtime_metric in runtime_metrics:
            runtime_params = runtime_metric.setdefault("params", {})
            if runtime_params.get("resolved_testset_path") or not reusable_input_metrics:
                continue
            input_metric = reusable_input_metrics[0]
            runtime_params["resolved_testset_path"] = input_metric["params"]["resolved_testset_path"]
            runtime_params["runtime_input_source_metric"] = str(input_metric.get("name", input_metric.get("analyzer")))

        for metric in enabled_metrics:
            metric_name = metric["name"]
            analyzer_name = metric["analyzer"]
            params = metric.get("params", {})
            missing_testset = (
                analyzer_name in EXECUTION_ANALYZERS
                and not params.get("resolved_testset_path")
            )
            metric_started = time.perf_counter()
            logger.info("Running metric '%s' using analyzer '%s'.", metric_name, analyzer_name)

            if missing_testset:
                generation = report.get("test_generation", {})
                generation_detail = ""
                if isinstance(generation, dict):
                    detail = generation.get("error") or generation.get("reason")
                    if detail:
                        generation_detail = f" Test generation reported: {detail}"
                result = _unavailable_metric(
                    "No instructor testset was supplied and automatic test generation did not produce an executable suite. "
                    "This metric is left unscored." + generation_detail
                )
                result.update({
                    "status": "inconclusive",
                    "requires_executed_testset": True,
                    "test_generation_status": report.get("test_generation", {}).get("status"),
                })
            elif analyzer_name in DOCKER_ANALYZERS and sandbox is None:
                result = _unavailable_metric(
                    f"Docker-dependent analyzer '{analyzer_name}' unavailable: {sandbox_error or 'sandbox not initialized'}.",
                )
            elif execution_blocked and analyzer_name in EXECUTION_ANALYZERS:
                reason = f"Skipped after preflight reported '{report['preflight']['status']}'."
                result = _unavailable_metric(reason)
                result.update({"status": "inconclusive", "skipped_after_preflight": True})
            else:
                try:
                    result = run_analyzer(
                        analyzer_name,
                        sandbox=sandbox,
                        metric=metric,
                        rules=rules,
                        source_code=source_code,
                        source_filename=source_filename,
                        test_case_cache=test_case_cache,
                    )
                    if analyzer_name == "runtime_profile" and params.get("runtime_input_source_metric"):
                        result["test_input_source_metric"] = params["runtime_input_source_metric"]
                except Exception as exc:
                    logger.warning("Analyzer '%s' failed; continuing with remaining metrics: %s", metric_name, exc)
                    result = _unavailable_metric(f"{type(exc).__name__}: {exc}")

            result.setdefault("status", "completed" if result.get("score") is not None else "unavailable")
            result["elapsed_seconds"] = round(time.perf_counter() - metric_started, 3)
            result["metric_name"] = metric_name
            result["analyzer"] = analyzer_name
            result["weight"] = metric["weight"]
            report[metric_name] = result
            logger.info(
                "Metric '%s' status: %s (%.3f s).",
                metric_name, result.get("status", "unknown"), result["elapsed_seconds"],
            )
    finally:
        if sandbox_manager is not None:
            try:
                sandbox_manager.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("Sandbox cleanup failed; report generation continues: %s", exc)

    report["analyzer_pipeline_elapsed_seconds"] = round(
        time.perf_counter() - analyzer_pipeline_started, 3
    )
    report["analyzer_metric_elapsed_seconds"] = {
        metric["name"]: report.get(metric["name"], {}).get("elapsed_seconds")
        for metric in enabled_metrics
        if isinstance(report.get(metric["name"]), dict)
    }
    report["scoring"] = {"status": "pending", "note": "Awaiting program understanding, global RSI, and final metric aggregation."}
    logger.info(
        "Deterministic analyzer pipeline completed in %.3f s.",
        report["analyzer_pipeline_elapsed_seconds"],
    )
    return report


def _json_compatible(value: Any) -> Any:
    """Convert arbitrary report values into strict-JSON-safe primitives."""
    import math
    from pathlib import Path

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    return str(value)


def _grade_with_rules_file_impl(
    rules_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    report_dir: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Run the complete grading flow and return the report and artifact paths.

    This function is also the stable backend used by the Gradio interface.
    """
    grading_started = time.perf_counter()
    rules = load_rules(rules_path, project_root)
    report = run_configured_analyzers(rules)
    ui_metadata = rules.get("_run_metadata", {})
    ui_metadata = ui_metadata if isinstance(ui_metadata, dict) else {}
    report["rules_provenance"] = {
        "rules_path": Path(rules.get("rules_path", rules_path)).name,
        "rules_sha256": report.get("rules_hash"),
        "assignment_statement": rules.get("problem", ""),
        "problem_contract": rules.get("problem_contract", {}),
        "selection": ui_metadata.get("selection", {}) if isinstance(ui_metadata, dict) else {},
    }
    report["runtime_configuration"] = {
        "model": settings.runtime_value("MODEL"),
        "temperature": settings.runtime_value("TEMPERATURE"),
        "rsi_iterations": settings.runtime_value("LLM_REVIEW_MAX_ITERATIONS"),
        "understanding_max_tokens": settings.runtime_value("LLM_PROGRAM_UNDERSTANDING_MAX_TOKENS"),
        "rsi_max_tokens": settings.runtime_value("LLM_GLOBAL_RSI_MAX_TOKENS"),
        "scoring_max_tokens": settings.runtime_value("LLM_GLOBAL_SCORE_MAX_TOKENS"),
        "source_char_limit": settings.runtime_value("LLM_GLOBAL_SOURCE_MAX_CHARS"),
        "evidence_char_limit": settings.runtime_value("LLM_MAX_EVIDENCE_CHARS"),
        "max_findings": settings.runtime_value("LLM_GLOBAL_MAX_FINDINGS"),
        "llm_request_policy": "single_request_per_stage_no_retries",
        "json_schema_enforced": True,
        "thinking_enabled": settings.runtime_value("LLAMA_REASONING_ENABLED"),
        "llm_read_timeout_seconds": settings.runtime_value("LLAMA_REQUEST_TIMEOUT")[1],
        "docker_cpu_limit": settings.runtime_value("DOCKER_CPU_LIMIT"),
    }
    report["evidence_ledger"] = build_evidence_ledger(report, rules)
    try:
        report["llm_rsi_evaluation"] = _review_with_local_llm(
            report,
            rules=rules,
        )
        llm_result = report["llm_rsi_evaluation"]
        logger.info(
            "LLM program understanding + global RSI completed with status %s: "
            "%s calls, %s/%s RSI iterations, %.3f s total.",
            llm_result.get("status"),
            llm_result.get("llm_calls", llm_result.get("attempts", 0)),
            llm_result.get("iterations_completed", 0),
            llm_result.get("iterations_requested", 0),
            llm_result.get("latency_seconds", {}).get("total_llm", 0.0),
        )
    except Exception as exc:
        logger.warning(
            "LLM RSI evaluation failed; generating a complete deterministic fallback report: %s: %s",
            type(exc).__name__, exc,
        )
        report["llm_rsi_evaluation"] = build_fallback_llm_evaluation(
            report, rules, f"{type(exc).__name__}: {exc}"
        )

    # Use authoritative deterministic results where available, rubric points
    # otherwise, then calculate the consolidated point total in Python.
    apply_metric_scores(rules["metrics"], report, report["llm_rsi_evaluation"])
    report["scoring"] = calculate_overall_score(rules["metrics"], report)
    llm_result = report.get("llm_rsi_evaluation", {})
    if isinstance(llm_result, dict) and llm_result.get("status") != "completed":
        # Preserve an incomplete status when any required metric remains
        # unmeasured; warnings must not turn a missing grade into a completed one.
        if report["scoring"].get("score") is None:
            report["scoring"]["status"] = "failed_incomplete"
            warning = (
                "One or more grading stages could not establish sufficient evidence for a complete grade. "
                "Unmeasured metrics remain unscored; no neutral or zero value was fabricated."
            )
        else:
            report["scoring"]["status"] = "completed_with_warnings"
            warning = (
                "One or more LLM stages used a validated fallback or rejected unsupported RSI candidates. "
                "Fallback-derived scores are labelled and are not presented as successful LLM scores."
            )
        report["scoring"].setdefault("warnings", []).append(warning)
    reviewer_request_seconds = (
        float(llm_result.get("latency_seconds", {}).get("total_llm", 0.0) or 0.0)
        if isinstance(llm_result, dict) else 0.0
    )
    generation_events = report.get("test_generation", {}).get("request_events", [])
    generation_request_seconds = sum(
        float(event.get("duration_seconds", 0.0) or 0.0)
        for event in generation_events if isinstance(event, dict)
    ) if isinstance(generation_events, list) else 0.0
    report["timings"] = {
        "analyzer_pipeline_seconds": report.get("analyzer_pipeline_elapsed_seconds"),
        "analyzer_metric_seconds": report.get("analyzer_metric_elapsed_seconds", {}),
        "llm_server_setup_seconds": llm_result.get("server_setup_seconds") if isinstance(llm_result, dict) else None,
        "llm_request_seconds": reviewer_request_seconds,
        "test_generation_request_seconds": generation_request_seconds,
        "total_llm_request_seconds": round(reviewer_request_seconds + generation_request_seconds, 6),
        "total_through_scoring_seconds": round(time.perf_counter() - grading_started, 3),
    }
    destination = report_dir or settings.REPORTS_DIR
    destination.mkdir(parents=True, exist_ok=True)
    # stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", rules["problem"]).strip("._") or "submission"
    problem_name = str(rules.get("problem") or "submission")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", problem_name)
    slug = slug.strip("_-")[:48].rstrip("_-") or "submission"
    short_hash = hashlib.sha256(problem_name.encode("utf-8")).hexdigest()[:10]
    stem = f"{slug}_{short_hash}"
    report_path = destination / f"{stem}.json"
    text_report_path = destination / f"{stem}.txt"
    # Strict JSON serialization is a final presentation boundary. Non-finite
    # floats or unexpected analyzer objects are normalized instead of aborting
    # an otherwise completed grading run.
    safe_report = _json_compatible(report)
    report_path.write_text(
        json.dumps(safe_report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    text_report_path.write_text(render_text_report(safe_report), encoding="utf-8")
    return report, report_path, text_report_path



def grade_with_rules_file(
    rules_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    report_dir: Path | None = None,
    runtime_options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Run grading with request-local UI options and isolated concurrent state."""
    with settings.runtime_overrides(runtime_options):
        return _grade_with_rules_file_impl(
            rules_path, project_root=project_root, report_dir=report_dir
        )


def main() -> None:
    try:
        logger.info("Loading instructor rules from %s.", RULES_PATH)
        report, report_path, text_report_path = grade_with_rules_file(RULES_PATH)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info("JSON report saved to %s.", report_path)
        logger.info("Text report saved to %s.", text_report_path)
        logger.info("Grading pipeline completed with status %s.", report.get("scoring", {}).get("status"))
    except Exception:
        logger.exception("Grading pipeline failed during configuration or report setup.")
        raise


if __name__ == "__main__":
    main()
