"""Application entry point for rule-driven C grading."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from grading_agent.analyzers.registry import run_analyzer
from grading_agent.config.rule_loader import load_rules
from grading_agent.llm.reviewer import (
    review_evidence,
    review_memory_safety_source,
)
from grading_agent.scoring.memory_safety import calculate_memory_safety_score
from grading_agent.llm.server import initiate_server
from grading_agent.config import settings
from grading_agent.sandbox.docker_sandbox import DockerSandbox

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


def _review_with_local_llm(report: dict[str, Any]) -> dict[str, Any]:
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
        return review_evidence(report)
    finally:
        _stop_llm_server(process)


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

        for metric in enabled_metrics:
            metric_name = metric["name"]
            analyzer_name = metric["analyzer"]

            logger.info(
                "Running metric '%s' using analyzer '%s'.",
                metric_name,
                analyzer_name,
            )

            result = run_analyzer(
                analyzer_name,
                sandbox=sandbox,
                metric=metric,
                rules=rules,
                source_code=source_code,
                source_filename=source_filename,
                test_case_cache=test_case_cache,
            )

            report[metric_name] = result

            logger.info("Metric '%s' completed.", metric_name)

    # The memory_safety metric receives its own composite score after the
    # LLM source audit. The overall cross-metric grade is a separate next step.
    report["scoring"] = {
        "status": "pending",
        "note": (
            "The composite memory-safety subscore is calculated from "
            "Valgrind and LLM source-audit evidence. The overall weighted "
            "grade across all enabled metrics has not yet been integrated."
        ),
    }

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
            report["llm_review"] = _review_with_local_llm(report)
            logger.info("LLM evidence review completed.")
        except Exception as exc:
            logger.exception("LLM review unavailable.")
            _record_unavailable_memory_audits(report, exc)
            report["llm_review"] = {
                "status": "unavailable",
                "error": str(exc),
            }

        report_dir = PROJECT_ROOT / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f'{rules["problem"]}.json'

        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info("Report saved to %s.", report_path)
        logger.info("Grading pipeline completed.")

    except Exception:
        logger.exception("Grading pipeline failed.")
        raise


if __name__ == "__main__":
    main()
