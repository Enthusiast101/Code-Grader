"""Application entry point for rule-driven C grading."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from grading_agent.analyzers.registry import run_analyzer
from grading_agent.config.rule_loader import load_rules
from grading_agent.llm.reviewer import (
    review_evidence,
    review_memory_safety_source,
)
from grading_agent.scoring.memory_safety import calculate_memory_safety_score
from grading_agent.scoring.overall import calculate_overall_score
from grading_agent.reporting.text_report import render_text_report
from grading_agent.llm.server import initiate_server
from grading_agent.config import settings
from grading_agent.sandbox.docker_sandbox import DockerSandbox
from grading_agent.preflight import run_preflight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = PROJECT_ROOT / "rules.json"


def _is_llm_server_ready() -> bool:
    """Return True when a compatible local llama-server is healthy."""
    import requests

    host = "127.0.0.1" if settings.HOST in {"0.0.0.0", "::"} else settings.HOST
    try:
        response = requests.get(
            f"http://{host}:{settings.PORT}/health",
            timeout=2,
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
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _record_unavailable_memory_audits(
    report: dict[str, Any],
    error: Exception,
) -> None:
    """Record an unavailable source audit and leave its score inconclusive."""
    for metric in report.get("configuration", {}).get("enabled_metrics", []):
        if metric.get("analyzer") != "memory_safety":
            continue

        memory_result = report.get(metric.get("name"))
        if not isinstance(memory_result, dict):
            continue

        component_config = memory_result.get("component_config", {})
        # A completed source audit remains valid even if the later general
        # evidence-review request fails.
        existing_audit = memory_result.get("llm_source_audit")
        if existing_audit is not None:
            continue

        memory_result["llm_source_audit"] = {
            "status": "unavailable",
            "model": settings.MODEL,
            "error": str(error),
        }
        if component_config:
            memory_result["composite_score"] = calculate_memory_safety_score(
                memory_result,
                memory_result["llm_source_audit"],
                valgrind_weight=component_config["valgrind_weight"],
                llm_audit_weight=component_config["llm_audit_weight"],
                severity_penalties=component_config["severity_penalties"],
            )
            memory_result["score"] = memory_result["composite_score"]["score"]
            memory_result["score_scale"] = "0-100"


def _review_with_local_llm(report: dict[str, Any], *, max_iterations: int = 3) -> dict[str, Any]:
    """Run the dedicated memory source audit and general evidence review."""
    process = None
    try:
        if not _is_llm_server_ready():
            logger.info("Starting local Llama server...")
            # initiate_server() waits for /health and returns the Popen process.
            process = initiate_server()
        else:
            logger.info("Using an already-running local Llama server.")

        # A single instructor metric triggers both Valgrind and this source
        # audit. The source audit is independent of the executed test inputs.
        for metric in report.get("configuration", {}).get("enabled_metrics", []):
            if metric.get("analyzer") != "memory_safety":
                continue

            metric_name = metric["name"]
            memory_result = report.get(metric_name)
            if not isinstance(memory_result, dict):
                continue

            component_config = memory_result.get("component_config", {})
            logger.info(
                "Running LLM source-level memory audit for metric '%s'.",
                metric_name,
            )
            try:
                llm_audit = review_memory_safety_source(
                    report["source_code"],
                    max_findings=component_config.get(
                        "llm_audit_max_findings", 30
                    ),
                )
            except Exception as exc:
                logger.exception("LLM memory source audit failed.")
                llm_audit = {
                    "status": "unavailable",
                    "model": settings.MODEL,
                    "error": str(exc),
                }

            memory_result["llm_source_audit"] = llm_audit
            memory_result["composite_score"] = calculate_memory_safety_score(
                memory_result,
                llm_audit,
                valgrind_weight=component_config["valgrind_weight"],
                llm_audit_weight=component_config["llm_audit_weight"],
                severity_penalties=component_config["severity_penalties"],
            )
            memory_result["score"] = memory_result["composite_score"]["score"]
            memory_result["score_scale"] = "0-100"

        # General evidence review sees the Valgrind logs, source-audit findings,
        # and composite memory score. It does not set or change any score.
        return review_evidence(report, max_iterations=max_iterations)
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
            result = sandbox.run(command, timeout_s=3)
            versions[tool] = result["output"].splitlines()[0] if result["output"] else f"unavailable (exit {result['exit_code']})"
        except Exception as exc:
            versions[tool] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        versions["lizard"] = version("lizard")
    except PackageNotFoundError:
        versions["lizard"] = "not-installed (lexical fallback active)"
    return versions


def run_configured_analyzers(
    rules: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch currently implemented analyzers from validated rules."""
    source_path = Path(rules["submission"]["resolved_source_path"])
    source_code = source_path.read_text(encoding="utf-8")
    source_filename = source_path.name

    # The registry shares this cache across analyzers that use the same tests.
    test_case_cache: dict[str, list[dict[str, str]]] = {}

    enabled_metrics = [
        metric for metric in rules["metrics"] if metric["enabled"]
    ]

    report: dict[str, Any] = {
        "submission": source_filename,
        "source_code": source_code,
        "source_sha256": hashlib.sha256(source_code.encode("utf-8")).hexdigest(),
        "problem": rules["problem"],
        "rules_hash": hashlib.sha256(
            Path(rules["rules_path"]).read_bytes()
        ).hexdigest(),
        "configuration": {
            "time_limit_s": rules["time_limit_s"],
            "memory_limit_mb": rules["memory_limit_mb"],
            "enabled_metrics": [
                {
                    "name": metric["name"],
                    "analyzer": metric["analyzer"],
                    "weight": metric["weight"],
                    **(
                        {
                            "component_weights": {
                                "valgrind": metric["params"]["valgrind_weight"],
                                "llm_source_audit": metric["params"]["llm_audit_weight"],
                            }
                        }
                        if metric["analyzer"] == "memory_safety"
                        else {}
                    ),
                }
                for metric in enabled_metrics
            ],
        },
    }

    memory_limit = f'{rules["memory_limit_mb"]}m'

    logger.info(
        "Starting Docker sandbox with memory limit %s.",
        memory_limit,
    )

    with DockerSandbox(memory_limit=memory_limit) as sandbox:
        logger.info("Docker sandbox started.")
        report["tool_versions"] = _collect_tool_versions(sandbox)
        logger.info("Running preflight compile and smoke checks.")
        report["preflight"] = run_preflight(
            sandbox,
            source_code=source_code,
            source_filename=source_filename,
            rules=rules,
        )
        logger.info("Preflight status: %s.", report["preflight"]["status"])

        execution_analyzers = {
            "io_tests", "edge_case_runner", "runtime_profile", "memory_safety"
        }
        execution_blocked = report["preflight"]["status"] in {
            "compile_failed", "compile_timeout", "timeout", "runtime_failure"
        }

        for metric in enabled_metrics:
            metric_name = metric["name"]
            analyzer_name = metric["analyzer"]

            logger.info(
                "Running metric '%s' using analyzer '%s'.",
                metric_name,
                analyzer_name,
            )

            if execution_blocked and analyzer_name in execution_analyzers:
                reason = (
                    "Skipped after preflight reported "
                    f"'{report['preflight']['status']}'."
                )
                result: dict[str, Any] = {
                    "status": "inconclusive",
                    "score": None,
                    "score_scale": "0-100",
                    "reason": reason,
                    "skipped_after_preflight": True,
                }
                if analyzer_name == "memory_safety":
                    result.update({
                        "analyzer": analyzer_name,
                        "compiled": False,
                        "analysis_error": True,
                        "tests_total": 0,
                        "tests_checked": 0,
                        "tests_clean": 0,
                        "results": [],
                        "runtime_tools": {
                            "status": "inconclusive",
                            "score": None,
                            "valgrind_score": None,
                            "asan_ubsan_score": None,
                            "valgrind_results": [],
                            "asan_ubsan_results": [],
                            "scoring_note": reason,
                        },
                        "component_config": {
                            "valgrind_weight": metric["params"]["valgrind_weight"],
                            "llm_audit_weight": metric["params"]["llm_audit_weight"],
                            "severity_penalties": metric["params"]["severity_penalties"],
                            "llm_audit_max_findings": metric["params"]["llm_audit_max_findings"],
                        },
                    })
                logger.warning("Skipping '%s': %s", metric_name, reason)
            else:
                result = run_analyzer(
                    analyzer_name,
                    sandbox=sandbox,
                    metric=metric,
                    rules=rules,
                    source_code=source_code,
                    source_filename=source_filename,
                    test_case_cache=test_case_cache,
                )

            result["metric_name"] = metric_name
            result["analyzer"] = analyzer_name
            result["weight"] = metric["weight"]
            report[metric_name] = result

            logger.info("Metric '%s' completed.", metric_name)

    # Overall scoring runs after the LLM source audit has supplied the second
    # required component of the composite memory-safety metric.
    report["scoring"] = {"status": "pending", "note": "Awaiting LLM source audit and final metric aggregation."}

    return report


def main() -> None:
    try:
        logger.info("Loading instructor rules from %s.", RULES_PATH)
        rules = load_rules(RULES_PATH, PROJECT_ROOT)

        logger.info(
            "Validated problem '%s' with %d enabled metrics.",
            rules["problem"],
            sum(metric["enabled"] for metric in rules["metrics"]),
        )

        # Execute the analyzers selected by rules.json.
        report = run_configured_analyzers(rules)

        # Review all collected analyzer evidence with the local LLM.
        try:
            report["llm_review"] = _review_with_local_llm(
                report,
                max_iterations=rules.get("llm_review", {}).get("max_iterations", 3),
            )
            logger.info("LLM evidence review completed.")
        except Exception as exc:
            logger.exception("LLM review unavailable.")
            _record_unavailable_memory_audits(report, exc)
            report["llm_review"] = {
                "status": "unavailable",
                "error": str(exc),
            }

        report["scoring"] = calculate_overall_score(
            rules["metrics"], report
        )

        report_dir = PROJECT_ROOT / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f'{rules["problem"]}.json'

        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        text_report_path = report_path.with_suffix(".txt")
        text_report_path.write_text(
            render_text_report(report),
            encoding="utf-8",
        )

        print(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info("JSON report saved to %s.", report_path)
        logger.info("Text report saved to %s.", text_report_path)
        logger.info("Grading pipeline completed.")

    except Exception:
        logger.exception("Grading pipeline failed.")
        raise


if __name__ == "__main__":
    main()
