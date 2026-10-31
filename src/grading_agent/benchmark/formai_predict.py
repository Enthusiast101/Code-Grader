"""Run resilient source-level FormAI-v2 vulnerability predictions.

Combined modes always attempt the LLM source audit independently of cppcheck.
The evidence-review (RSI) stage is an additional signal, never a gate that can
erase a valid cppcheck or LLM result. Each source record is emitted even when
one analysis stage fails, so infrastructure errors do not discard a batch.
"""
from __future__ import annotations

import argparse
from contextlib import suppress
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Iterator

import requests

from grading_agent.config import settings
from grading_agent.llm.evidence import build_evidence_ledger
from grading_agent.llm.reviewer import review_c_vulnerabilities_source, review_evidence
from grading_agent.llm.server import initiate_server

logger = logging.getLogger("formai_predict")
CPP_CHECK = re.compile(
    r"^.*?:(\d+):\d+:\s*(error|warning|style|performance|portability):\s*(.*?)\s*\[([^\]]+)\]",
    re.IGNORECASE,
)
VULNERABILITY_ID = re.compile(
    r"buffer|out[-_ ]?of[-_ ]?bounds|array.?index|mem(?:ory)?leak|invalid.?pointer|use.?after.?free|uninit|null.?pointer|integer.?overflow|dealloc|resource.?leak|invalidscanf|invalidfunctionarg|stack.?overflow|double.?free",
    re.IGNORECASE,
)


def _iter_inputs(path: Path) -> Iterator[dict[str, Any]]:
    """Yield a record for every nonblank input line, including malformed rows."""
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                yield {
                    "file_name": f"__invalid_input_line_{line_number}__",
                    "source_code": "",
                    "_input_error": f"Invalid JSONL at line {line_number}: {exc}",
                }
                continue
            if not isinstance(item, dict) or not isinstance(item.get("file_name"), str) or not isinstance(item.get("source_code"), str):
                label = item.get("file_name") if isinstance(item, dict) and isinstance(item.get("file_name"), str) else f"__invalid_input_line_{line_number}__"
                yield {
                    "file_name": label,
                    "source_code": item.get("source_code", "") if isinstance(item, dict) and isinstance(item.get("source_code"), str) else "",
                    "_input_error": f"Invalid FormAI source input at line {line_number}: expected file_name and source_code strings.",
                }
                continue
            yield item


def _server_ready() -> bool:
    try:
        return requests.get(settings.LLAMA_HEALTH_URL, timeout=settings.LLAMA_HEALTH_TIMEOUT_S).status_code == 200
    except requests.RequestException:
        return False


def _cppcheck_findings(output: str) -> list[dict[str, Any]]:
    findings = []
    for line in output.splitlines():
        match = CPP_CHECK.match(line.strip())
        if not match:
            continue
        severity, message, check_id = match.group(2).lower(), match.group(3), match.group(4)
        if VULNERABILITY_ID.search(check_id):
            findings.append({
                "line": int(match.group(1)),
                "severity": severity,
                "id": check_id,
                "description": message,
            })
    return findings


def _prediction_from_findings(findings: list[dict[str, Any]]) -> tuple[bool, int | None, str | None]:
    if not findings:
        return False, None, None
    selected = findings[0]
    return True, selected.get("line"), selected.get("id") or selected.get("category")


def _build_rsi_review_report(
    item: dict[str, Any],
    source: str,
    tool_findings: list[dict[str, Any]],
    tool_status: str,
    llm_audit: dict[str, Any],
) -> dict[str, Any]:
    """Adapt source-only benchmark signals to the shared evidence-review schema."""
    return {
        "submission": item["file_name"],
        "source_code": source,
        "problem": "FormAI-v2-source-vulnerability-review",
        "configuration": {
            "enabled_metrics": [
                {"name": name, "analyzer": analyzer, "weight": weight}
                for name, analyzer, weight in settings.FORM_AI_RSI_METRIC_DEFAULTS
            ]
        },
        "structure": {
            "status": tool_status,
            "tool": "cppcheck",
            "finding_count": len(tool_findings),
            "findings": tool_findings,
        },
        "memory_safety": {
            "status": llm_audit.get("status", "inconclusive"),
            "tests_total": 0,
            "tests_clean": 0,
            "memory_safe": None,
            "runtime_tools": {"status": "not_run", "score": None},
            "llm_source_audit": llm_audit,
        },
    }


def _prediction_from_rsi_review(
    review_result: dict[str, Any],
    ledger: list[dict[str, Any]],
) -> tuple[bool, int | None, str | None, list[dict[str, Any]]]:
    """Accept only claims anchored to source or analyzer evidence."""
    if review_result.get("status") not in {"completed", "completed_with_warnings"}:
        return False, None, None, []
    validation = review_result.get("claim_validation", {})
    if not isinstance(validation, dict) or validation.get("status") not in {"completed", "completed_with_warnings"}:
        return False, None, None, []
    review = review_result.get("review", {})
    if not isinstance(review, dict):
        return False, None, None, []
    by_id = {entry.get("id"): entry for entry in ledger}
    supported: list[dict[str, Any]] = []
    for claim in review.get("findings", []):
        if not isinstance(claim, dict) or claim.get("kind") not in {"observed", "potential"}:
            continue
        text = claim.get("text", "")
        if not isinstance(text, str) or not VULNERABILITY_ID.search(text):
            continue
        evidence = [by_id[ref] for ref in claim.get("evidence_ids", []) if ref in by_id]
        source_evidence = [
            entry for entry in evidence
            if entry.get("kind") in {"source_line", "static_finding", "model_audit_finding"}
        ]
        if not source_evidence:
            continue
        line = next((
            entry.get("facts", {}).get("line") for entry in source_evidence
            if isinstance(entry.get("facts", {}).get("line"), int)
        ), None)
        error_type = next((
            entry.get("facts", {}).get("category") or entry.get("facts", {}).get("id")
            for entry in source_evidence
            if entry.get("facts", {}).get("category") or entry.get("facts", {}).get("id")
        ), None)
        supported.append({
            "text": text[:400],
            "evidence_ids": claim.get("evidence_ids", [])[:5],
            "line": line,
            "error_type": error_type or "source_review_finding",
            "kind": claim.get("kind"),
        })
    if not supported:
        return False, None, None, []
    selected = supported[0]
    return True, selected.get("line"), selected.get("error_type"), supported


def _run_tool_stage(sandbox: Any, source: str, timeout_s: float) -> tuple[list[dict[str, Any]], str, str | None]:
    if sandbox is None:
        return [], "unavailable", "Docker sandbox is unavailable; cppcheck was not run."
    try:
        sandbox.write_file("formai_candidate.c", source)
        result = sandbox.run(
            [
                "cppcheck", "--enable=warning", "--inline-suppr",
                "--language=c", "--std=c11", "--template=gcc",
                "--error-exitcode=0", "formai_candidate.c",
            ],
            timeout_s=timeout_s,
        )
        findings = _cppcheck_findings(result.get("output", ""))
        complete = (
            not result.get("timed_out", False)
            and result.get("exit_code") != 127
            and not result.get("output_may_be_truncated", False)
        )
        if not complete:
            return findings, "inconclusive", "cppcheck timed out, was not found, or returned truncated output."
        return findings, "completed", None
    except Exception as exc:  # One deterministic tool failure must not stop LLM analysis.
        logger.warning("cppcheck failed for one FormAI record: %s", exc)
        return [], "unavailable", f"{type(exc).__name__}: {exc}"


def _run_llm_stage(source: str, max_findings: int, setup_error: str | None) -> dict[str, Any]:
    if setup_error:
        return {"status": "unavailable", "model": settings.MODEL, "error": setup_error}
    try:
        result = review_c_vulnerabilities_source(source, max_findings=max_findings)
        if not isinstance(result, dict):
            return {"status": "inconclusive", "model": settings.MODEL, "error": "LLM auditor returned an invalid response object."}
        return result
    except Exception as exc:  # Isolate network, server, and parser errors to this sample.
        logger.exception("LLM source audit failed for one FormAI record")
        return {"status": "unavailable", "model": settings.MODEL, "error": f"{type(exc).__name__}: {exc}"}


def _select_hybrid_prediction(
    *,
    mode: str,
    tool_status: str,
    tool_findings: list[dict[str, Any]],
    llm_audit: dict[str, Any] | None,
    rsi_review: dict[str, Any] | None = None,
    rsi_findings: list[dict[str, Any]] | None = None,
) -> tuple[bool | None, int | None, str | None, str]:
    """Combine completed primary methods; refinement can add but never veto."""
    tool_done = tool_status == "completed"
    llm_status = llm_audit.get("status") if isinstance(llm_audit, dict) else None
    llm_result_usable = llm_status in {"completed", "completed_with_warnings"}
    llm_chunks_failed = int((llm_audit or {}).get("chunks_failed", 0) or 0)
    llm_done = llm_result_usable and llm_chunks_failed == 0
    llm_findings = []
    if llm_result_usable:
        audit_payload = llm_audit.get("audit", {})
        if isinstance(audit_payload, dict) and isinstance(audit_payload.get("findings"), list):
            llm_findings = audit_payload["findings"]
    # Concrete findings survive a timeout/truncation status. An incomplete
    # tool run cannot support a negative prediction, but its emitted positive
    # diagnostics remain usable evidence.
    tool_positive, tool_line, tool_type = _prediction_from_findings(tool_findings)
    # A positive finding from a successfully audited chunk remains useful even
    # when another chunk failed. A partial audit with no findings cannot justify
    # a negative whole-file classification.
    llm_positive = bool(llm_findings) and llm_result_usable
    rsi_findings = rsi_findings or []
    rsi_positive = bool(rsi_findings) and isinstance(rsi_review, dict) and rsi_review.get("status") in {"completed", "completed_with_warnings"}

    if mode == "tools_only":
        if not tool_done and not tool_findings:
            return None, None, None, "inconclusive"
        if tool_positive:
            return True, tool_line, tool_type, "completed" if tool_done else "completed_with_warnings"
        if not tool_done:
            return None, None, None, "inconclusive"
        return False, None, None, "completed"
    if mode == "llm_only":
        if llm_positive:
            first = llm_findings[0]
            status = "completed" if llm_done and llm_status == "completed" else "completed_with_warnings"
            return True, first.get("line"), first.get("category"), status
        if not llm_done:
            return None, None, None, "inconclusive"
        status = "completed" if llm_status == "completed" else "completed_with_warnings"
        return False, None, None, status

    completed_primary = tool_done or llm_done or bool(tool_findings) or llm_positive
    if not completed_primary:
        return None, None, None, "inconclusive"

    # OR fusion preserves a positive signal from either completed method.
    predicted = tool_positive or llm_positive or rsi_positive
    if rsi_positive:
        rsi_first = rsi_findings[0]
        line, error_type = rsi_first.get("line"), rsi_first.get("error_type")
    elif llm_positive:
        first = llm_findings[0]
        line, error_type = first.get("line"), first.get("category")
    else:
        line, error_type = tool_line, tool_type

    all_expected_complete = tool_done and llm_status == "completed"
    if mode == "hybrid_rsi":
        rsi_done = isinstance(rsi_review, dict) and rsi_review.get("status") in {"completed", "completed_with_warnings"}
        all_expected_complete = all_expected_complete and rsi_done
    status = "completed" if all_expected_complete else "completed_with_warnings"
    return predicted, line, error_type, status


def run(args: argparse.Namespace) -> None:
    process = None
    sandbox = None
    sandbox_error: str | None = None
    needs_llm = args.mode in {"llm_only", "hybrid", "hybrid_rsi"}
    needs_tools = args.mode in {"tools_only", "hybrid", "hybrid_rsi"}
    server_error: str | None = None

    # LLM setup is independent of Docker setup. Failure to initialize one stage
    # is recorded but must not prevent the other stage from running.
    if needs_llm and not _server_ready():
        try:
            process = initiate_server()
        except Exception as exc:
            server_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Local LLM server unavailable: %s; per-record outputs will still be written.", server_error)

    if needs_tools:
        try:
            from grading_agent.sandbox.docker_sandbox import DockerSandbox
            sandbox_manager = DockerSandbox(memory_limit=args.memory_limit)
            sandbox = sandbox_manager.__enter__()
        except Exception as exc:
            sandbox = None
            sandbox_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Docker sandbox unavailable; combined mode will continue with LLM analysis: %s", sandbox_error)

    output_path = args.output
    count = 0
    errors_count = 0
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output:
            for item in _iter_inputs(args.inputs):
                if args.max_samples is not None and count >= args.max_samples:
                    break
                source = item["source_code"]
                if item.get("_input_error"):
                    prediction = {
                        "file_name": item["file_name"],
                        "predicted_vulnerable": None,
                        "predicted_line": None,
                        "predicted_error_type": None,
                        "prediction_status": "inconclusive",
                        "mode": args.mode,
                        "prediction_source": "none",
                        "tool_status": "not_run",
                        "tool_findings": [],
                        "llm_findings": [],
                        "llm_audit_status": "not_run",
                        "rsi_review_status": "not_run",
                        "rsi_findings": [],
                        "stage_errors": {"input": item["_input_error"]},
                    }
                    output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    output.flush()
                    count += 1
                    errors_count += 1
                    continue
                tool_findings: list[dict[str, Any]] = []
                tool_status = "not_run"
                tool_error = None
                llm_audit: dict[str, Any] | None = None
                rsi_review: dict[str, Any] | None = None
                rsi_findings: list[dict[str, Any]] = []
                rsi_error = None
                rsi_note = None

                if needs_tools:
                    tool_findings, tool_status, tool_error = _run_tool_stage(
                        sandbox, source, args.tool_timeout
                    )
                    if tool_status == "unavailable" and sandbox_error:
                        tool_error = sandbox_error

                if needs_llm:
                    llm_audit = _run_llm_stage(source, args.max_llm_findings, server_error)

                if args.mode == "hybrid_rsi" and (
                    (llm_audit or {}).get("status") in {"completed", "completed_with_warnings"}
                    or tool_status == "completed"
                    or bool(tool_findings)
                ):
                    # Evidence review runs whenever at least one primary source
                    # stage produced a usable result. It is not gated on both
                    # primary methods and cannot veto their prediction.
                    try:
                        review_report = _build_rsi_review_report(
                            item, source, tool_findings, tool_status, llm_audit or {}
                        )
                        rsi_ledger = build_evidence_ledger(review_report, {"metrics": []})
                        rsi_review = review_evidence(
                            review_report,
                            rules={"metrics": []},
                            max_iterations=args.rsi_iterations,
                        )
                        _, _, _, rsi_findings = _prediction_from_rsi_review(rsi_review, rsi_ledger)
                        if not rsi_findings:
                            rsi_note = "Evidence review completed without additional validated vulnerability claims; primary results are retained."
                    except Exception as exc:
                        rsi_error = f"{type(exc).__name__}: {exc}"
                        rsi_review = {"status": "unavailable", "error": rsi_error}
                        logger.warning("RSI evidence review failed for %s: %s", item["file_name"], exc)
                elif args.mode == "hybrid_rsi":
                    rsi_review = {
                        "status": "skipped",
                        "reason": "No primary analysis completed and no usable tool findings were available; primary-stage status is retained.",
                    }

                predicted, line, error_type, status = _select_hybrid_prediction(
                    mode=args.mode,
                    tool_status=tool_status,
                    tool_findings=tool_findings,
                    llm_audit=llm_audit,
                    rsi_review=rsi_review,
                    rsi_findings=rsi_findings,
                )
                llm_findings = []
                if (llm_audit or {}).get("status") == "completed":
                    audit_payload = (llm_audit or {}).get("audit", {})
                    if isinstance(audit_payload, dict) and isinstance(audit_payload.get("findings"), list):
                        llm_findings = audit_payload["findings"]
                if predicted is None:
                    errors_count += 1

                prediction = {
                    "file_name": item["file_name"],
                    "predicted_vulnerable": predicted,
                    "predicted_line": line,
                    "predicted_error_type": error_type,
                    "prediction_status": status,
                    "mode": args.mode,
                    "prediction_source": _prediction_source(args.mode, tool_status, tool_findings, llm_audit, rsi_findings),
                    "tool_status": tool_status,
                    "tool_findings": tool_findings,
                    "tool_error": tool_error,
                    "llm_findings": llm_findings,
                    "llm_audit_status": (llm_audit or {}).get("status", "not_run"),
                    "llm_audit_summary": ((llm_audit or {}).get("audit", {}) or {}).get("summary") if isinstance((llm_audit or {}).get("audit", {}), dict) else None,
                    "llm_error": (llm_audit or {}).get("error"),
                    "rsi_review_status": (rsi_review or {}).get("status", "not_run"),
                    "rsi_findings": rsi_findings,
                    "rsi_error": rsi_error or (rsi_review or {}).get("error"),
                    "rsi_note": rsi_note,
                    "stage_errors": {
                        key: value for key, value in {
                            "tools": tool_error,
                            "llm": (llm_audit or {}).get("error"),
                            "rsi": rsi_error or (rsi_review or {}).get("error"),
                        }.items() if value
                    },
                }
                output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                output.flush()
                count += 1
                if count % 25 == 0:
                    logger.info("Processed %d FormAI samples.", count)
    finally:
        if sandbox is not None:
            with suppress(Exception):
                sandbox_manager.__exit__(None, None, None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=settings.LLAMA_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    logger.info(
        "Wrote %d predictions to %s (%d without any completed primary analysis stage).",
        count, output_path, errors_count,
    )


def _prediction_source(
    mode: str,
    tool_status: str,
    tool_findings: list[dict[str, Any]],
    llm_audit: dict[str, Any] | None,
    rsi_findings: list[dict[str, Any]],
) -> str:
    if mode == "tools_only":
        return "cppcheck" if tool_status == "completed" or tool_findings else "none"
    if mode == "llm_only":
        return "llm" if (llm_audit or {}).get("status") in {"completed", "completed_with_warnings"} else "none"
    sources = []
    if tool_status == "completed" or tool_findings:
        sources.append("cppcheck")
    llm_status = (llm_audit or {}).get("status")
    llm_audit_body = (llm_audit or {}).get("audit", {})
    llm_findings = llm_audit_body.get("findings", []) if isinstance(llm_audit_body, dict) else []
    llm_chunks_failed = int((llm_audit or {}).get("chunks_failed", 0) or 0)
    if llm_status == "completed" or (llm_status == "completed_with_warnings" and (llm_chunks_failed == 0 or llm_findings)):
        sources.append("llm")
    if rsi_findings:
        sources.append("evidence_review")
    return "+".join(sources) if sources else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, help="Source-only JSONL created by formai_sample.")
    parser.add_argument("--mode", required=True, choices=settings.FORM_AI_PREDICTION_MODES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=settings.FORM_AI_DEFAULT_MAX_SAMPLES)
    parser.add_argument("--tool-timeout", type=float, default=settings.FORM_AI_DEFAULT_TOOL_TIMEOUT_S)
    parser.add_argument("--max-llm-findings", type=int, default=settings.FORM_AI_DEFAULT_MAX_FINDINGS)
    parser.add_argument("--rsi-iterations", type=int, default=settings.FORM_AI_DEFAULT_RSI_ITERATIONS, help=f"Number of bounded RSI passes for hybrid_rsi (1-{settings.LLM_REVIEW_MAX_ITERATIONS}).")
    parser.add_argument("--memory-limit", default=settings.FORM_AI_DEFAULT_MEMORY_LIMIT)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.tool_timeout <= 0:
        parser.error("--tool-timeout must be positive")
    if args.max_llm_findings < 1:
        parser.error("--max-llm-findings must be positive")
    if not 1 <= args.rsi_iterations <= settings.LLM_REVIEW_MAX_ITERATIONS:
        parser.error(f"--rsi-iterations must be between 1 and {settings.LLM_REVIEW_MAX_ITERATIONS}")
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT_BENCHMARK)
    run(args)


if __name__ == "__main__":
    main()
