"""Generate FormAI-v2 vulnerability predictions for one evaluation mode.

This benchmark runner performs source-level tool and/or LLM analysis. It does
not execute the dataset programs because the dataset does not provide a
standard per-program input oracle; running arbitrary programs without inputs
would not exercise meaningful paths. The normal assignment pipeline still
uses I/O tests, Valgrind, sanitizers, and runtime profiling.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import requests

from grading_agent.config import settings
from grading_agent.llm.evidence import build_evidence_ledger
from grading_agent.llm.reviewer import review_evidence, review_memory_safety_source
from grading_agent.llm.server import initiate_server

logger = logging.getLogger("formai_predict")
CPP_CHECK = re.compile(
    r"^.*?:(\d+):\d+:\s*(error|warning|style|performance|portability):\s*(.*?)\s*\[([^\]]+)\]",
    re.IGNORECASE,
)
VULNERABILITY_ID = re.compile(
    r"buffer|out[-_ ]?of[-_ ]?bounds|array.?index|mem(?:ory)?leak|invalid.?pointer|use.?after.?free|uninit|null.?pointer|integer.?overflow|dealloc|resource.?leak|invalidscanf|invalidfunctionarg",
    re.IGNORECASE,
)


def _iter_inputs(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item.get("file_name"), str) or not isinstance(item.get("source_code"), str):
                raise ValueError(f"Invalid FormAI source input at line {line_number}.")
            yield item


def _server_ready() -> bool:
    host = "127.0.0.1" if settings.HOST in {"0.0.0.0", "::"} else settings.HOST
    try:
        return requests.get(f"http://{host}:{settings.PORT}/health", timeout=2).status_code == 200
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
                {"name": "structure", "analyzer": "cppcheck_source", "weight": 0.5},
                {"name": "memory_safety", "analyzer": "llm_source_audit", "weight": 0.5},
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
    """Accept only validated vulnerability claims anchored to exact source evidence."""
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


def run(args: argparse.Namespace) -> None:
    process = None
    sandbox_class = None
    needs_llm = args.mode in {"llm_only", "hybrid", "hybrid_rsi"}
    needs_tools = args.mode in {"tools_only", "hybrid", "hybrid_rsi"}
    if needs_llm and not _server_ready():
        process = initiate_server()

    output_path = args.output
    count = 0
    try:
        if needs_tools:
            # Import Docker lazily so LLM-only evaluation and parsing helpers do
            # not need Docker installed on the evaluation host.
            from grading_agent.sandbox.docker_sandbox import DockerSandbox
            sandbox_class = DockerSandbox

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sandbox_context = (
            sandbox_class(memory_limit=args.memory_limit)
            if sandbox_class is not None else nullcontext(None)
        )
        with sandbox_context as sandbox, output_path.open("w", encoding="utf-8") as output:
            for item in _iter_inputs(args.inputs):
                if args.max_samples is not None and count >= args.max_samples:
                    break
                filename = "formai_candidate.c"
                source = item["source_code"]
                tool_findings: list[dict[str, Any]] = []
                llm_audit: dict[str, Any] | None = None
                tool_status = "not_run"

                if needs_tools:
                    if sandbox is None:
                        raise RuntimeError("Docker sandbox was not initialized for a tool-based mode.")
                    sandbox.write_file(filename, source)
                    tool_result = sandbox.run(
                        [
                            "cppcheck", "--enable=warning", "--inline-suppr",
                            "--language=c", "--std=c11", "--template=gcc",
                            "--error-exitcode=0", filename,
                        ],
                        timeout_s=args.tool_timeout,
                    )
                    tool_findings = _cppcheck_findings(tool_result["output"])
                    tool_status = (
                        "completed"
                        if (
                            not tool_result["timed_out"]
                            and tool_result["exit_code"] != 127
                            and not tool_result.get("output_may_be_truncated", False)
                        )
                        else "inconclusive"
                    )

                if needs_llm:
                    llm_audit = review_memory_safety_source(source, max_findings=args.max_llm_findings)

                tool_positive, tool_line, tool_type = _prediction_from_findings(tool_findings)
                llm_findings = (llm_audit or {}).get("audit", {}).get("findings", []) if (llm_audit or {}).get("status") == "completed" else []
                llm_positive = bool(llm_findings)
                rsi_review: dict[str, Any] | None = None
                rsi_findings: list[dict[str, Any]] = []
                rsi_positive = False
                rsi_line: int | None = None
                rsi_error_type: str | None = None
                if args.mode == "hybrid_rsi":
                    if tool_status == "completed" and (llm_audit or {}).get("status") == "completed":
                        review_report = _build_rsi_review_report(
                            item, source, tool_findings, tool_status, llm_audit or {}
                        )
                        rsi_ledger = build_evidence_ledger(review_report, {"metrics": []})
                        rsi_review = review_evidence(
                            review_report,
                            rules={"metrics": []},
                            max_iterations=args.rsi_iterations,
                        )
                        rsi_positive, rsi_line, rsi_error_type, rsi_findings = _prediction_from_rsi_review(
                            rsi_review, rsi_ledger
                        )
                    else:
                        rsi_review = {
                            "status": "inconclusive",
                            "reason": "Tool analysis or LLM source audit was incomplete; RSI was not run.",
                        }
                if args.mode == "tools_only":
                    status = tool_status
                    predicted = tool_positive if status == "completed" else None
                    line, error_type = tool_line, tool_type
                elif args.mode == "llm_only":
                    status = (llm_audit or {}).get("status", "inconclusive")
                    predicted = llm_positive if status == "completed" else None
                    first = llm_findings[0] if llm_findings else {}
                    line, error_type = first.get("line"), first.get("category")
                elif args.mode == "hybrid":
                    status = "completed" if tool_status == "completed" and (llm_audit or {}).get("status") == "completed" else "inconclusive"
                    predicted = (tool_positive or llm_positive) if status == "completed" else None
                    first = llm_findings[0] if llm_findings else {}
                    line = first.get("line") if llm_positive else tool_line
                    error_type = first.get("category") if llm_positive else tool_type
                else:  # hybrid_rsi
                    rsi_status = (rsi_review or {}).get("status", "inconclusive")
                    complete = rsi_status in {"completed", "completed_with_warnings"}
                    status = rsi_status if complete else "inconclusive"
                    predicted = (tool_positive or llm_positive or rsi_positive) if complete else None
                    if rsi_positive:
                        line, error_type = rsi_line, rsi_error_type
                    elif llm_positive:
                        first = llm_findings[0]
                        line, error_type = first.get("line"), first.get("category")
                    else:
                        line, error_type = tool_line, tool_type

                prediction = {
                    "file_name": item["file_name"],
                    "predicted_vulnerable": predicted,
                    "predicted_line": line,
                    "predicted_error_type": error_type,
                    "prediction_status": status,
                    "mode": args.mode,
                    "tool_findings": tool_findings,
                    "llm_findings": llm_findings,
                    "llm_audit_status": (llm_audit or {}).get("status", "not_run"),
                    "rsi_review_status": (rsi_review or {}).get("status", "not_run"),
                    "rsi_findings": rsi_findings,
                }
                output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                output.flush()
                count += 1
                if count % 25 == 0:
                    logger.info("Processed %d FormAI samples.", count)

    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    logger.info("Wrote %d predictions to %s", count, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, help="Source-only JSONL created by formai_sample.")
    parser.add_argument("--mode", required=True, choices=("tools_only", "llm_only", "hybrid", "hybrid_rsi"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tool-timeout", type=float, default=20.0)
    parser.add_argument("--max-llm-findings", type=int, default=30)
    parser.add_argument("--rsi-iterations", type=int, default=3, help="Number of bounded RSI passes for hybrid_rsi (1-3).")
    parser.add_argument("--memory-limit", default="512m")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.tool_timeout <= 0:
        parser.error("--tool-timeout must be positive")
    if args.max_llm_findings < 1:
        parser.error("--max-llm-findings must be positive")
    if not 1 <= args.rsi_iterations <= 3:
        parser.error("--rsi-iterations must be between 1 and 3")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(args)


if __name__ == "__main__":
    main()
