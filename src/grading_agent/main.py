"""Application entry point for rule-driven C grading."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from grading_agent.analyzers.registry import run_analyzer
from grading_agent.config.rule_loader import load_rules
from grading_agent.llm.reviewer import evaluate_metrics_with_llm
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
    """Start/reuse the local model and score every enabled metric with RSI."""
    process = None
    try:
        if not _is_llm_server_ready():
            logger.info("Starting local Llama server...")
            process = initiate_server()
        else:
            logger.info("Using an already-running local Llama server.")
        return evaluate_metrics_with_llm(report, rules)
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
    source_path = Path(rules["submission"]["resolved_source_path"])
    source_code = source_path.read_text(encoding="utf-8")
    source_filename = source_path.name
    test_case_cache: dict[str, list[dict[str, Any]]] = {}
    enabled_metrics = [metric for metric in rules["metrics"] if metric["enabled"]]
    docker_needed = any(metric["analyzer"] in DOCKER_ANALYZERS for metric in enabled_metrics)

    report: dict[str, Any] = {
        "submission": source_filename,
        "source_code": source_code,
        "source_sha256": hashlib.sha256(source_code.encode("utf-8")).hexdigest(),
        "problem": rules["problem"],
        "rules_hash": hashlib.sha256(Path(rules["rules_path"]).read_bytes()).hexdigest(),
        "configuration": {
            "time_limit_s": rules["time_limit_s"],
            "memory_limit_mb": rules["memory_limit_mb"],
            "problem_contract": rules.get("problem_contract", {}),
            "enabled_metrics": [
                {
                    "name": metric["name"],
                    "analyzer": metric["analyzer"],
                    "weight": metric["weight"],
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
            sandbox_manager = DockerSandbox(memory_limit=f'{rules["memory_limit_mb"]}m')
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

        for metric in enabled_metrics:
            metric_name = metric["name"]
            analyzer_name = metric["analyzer"]
            logger.info("Running metric '%s' using analyzer '%s'.", metric_name, analyzer_name)

            if analyzer_name in DOCKER_ANALYZERS and sandbox is None:
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
                except Exception as exc:
                    logger.warning("Analyzer '%s' failed; continuing with remaining metrics: %s", metric_name, exc)
                    result = _unavailable_metric(f"{type(exc).__name__}: {exc}")

            result.setdefault("status", "completed" if result.get("score") is not None else "unavailable")
            result["metric_name"] = metric_name
            result["analyzer"] = analyzer_name
            result["weight"] = metric["weight"]
            report[metric_name] = result
            logger.info("Metric '%s' status: %s.", metric_name, result.get("status", "unknown"))
    finally:
        if sandbox_manager is not None:
            try:
                sandbox_manager.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("Sandbox cleanup failed; report generation continues: %s", exc)

    report["scoring"] = {"status": "pending", "note": "Awaiting LLM RSI and final metric aggregation."}
    return report


def grade_with_rules_file(
    rules_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    report_dir: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Run the complete grading flow and return the report and artifact paths.

    This function is also the stable backend used by the Gradio interface.
    """
    rules = load_rules(rules_path, project_root)
    report = run_configured_analyzers(rules)
    report["evidence_ledger"] = build_evidence_ledger(report, rules)
    try:
        report["llm_rsi_evaluation"] = _review_with_local_llm(
            report,
            rules=rules,
        )
        logger.info("LLM RSI metric evaluation completed with status %s.", report["llm_rsi_evaluation"].get("status"))
    except Exception as exc:
        logger.warning("LLM RSI evaluation unavailable; analyzer scores will be used: %s: %s", type(exc).__name__, exc)
        report["llm_rsi_evaluation"] = {
            "status": "unavailable",
            "model": settings.MODEL,
            "error": f"{type(exc).__name__}: {exc}",
            "metrics": [],
        }

    # Use the RSI metric score, fall back to the deterministic analyzer score,
    # then calculate the final weighted grade in Python.
    apply_metric_scores(rules["metrics"], report, report["llm_rsi_evaluation"])
    report["scoring"] = calculate_overall_score(rules["metrics"], report)
    destination = report_dir or settings.REPORTS_DIR
    destination.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", rules["problem"]).strip("._") or "submission"
    report_path = destination / f"{stem}.json"
    text_report_path = destination / f"{stem}.txt"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    text_report_path.write_text(render_text_report(report), encoding="utf-8")
    return report, report_path, text_report_path



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
