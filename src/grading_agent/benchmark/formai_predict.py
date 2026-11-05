"""Run resilient source-level FormAI-v2 vulnerability predictions.

Combined modes always attempt the LLM source audit independently of cppcheck.
In hybrid_rsi, RSI adjudicates each LLM-originated candidate against source-line
and analyzer evidence; confirmed cppcheck positives remain intact. An unresolved
positive summary/candidate is marked inconclusive rather than safe. Every input
record is emitted even when an analysis stage fails.
"""
from __future__ import annotations

import argparse
from contextlib import suppress
import json
import logging
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterator

import requests

from grading_agent.analyzers.c_properties import check_c_properties
from grading_agent.benchmark.formai_dataset import FAMILIES, error_family
from grading_agent.config import settings
from grading_agent.llm.evidence import build_evidence_ledger
from grading_agent.llm.reviewer import review_c_vulnerabilities_source, review_evidence, verify_vulnerability_claims
from grading_agent.llm.server import initiate_server

logger = logging.getLogger("formai_predict")
CPP_CHECK = re.compile(
    r"^.*?:(\d+):\d+:\s*(error|warning|style|performance|portability):\s*(.*?)\s*\[([^\]]+)\]",
    re.IGNORECASE,
)
VULNERABILITY_ID = re.compile(
    r"buffer|out[-_ ]?of[-_ ]?bounds|array.?index|mem(?:ory)?leak|invalid.?pointer|use.?after.?free|uninit|null.?pointer|integer.?overflow|integer.?truncation|dealloc|resource.?leak|invalidscanf|invalidfunctionarg|stack.?overflow|double.?free|division.?by.?zero|divide.?by.?zero|format.?string|dangling.?pointer|signed.?overflow",
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


def _validate_predictor_input(path: Path) -> dict[str, Any]:
    """Fail before starting Docker/LLM if the source split is malformed or duplicated."""
    names: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid source JSONL at line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Source JSONL line {line_number} must be an object.")
            name, source = item.get("file_name"), item.get("source_code")
            if not isinstance(name, str) or not name.strip() or not isinstance(source, str) or not source.strip():
                raise ValueError(f"Source JSONL line {line_number} must contain non-empty file_name and source_code strings.")
            if name in names:
                raise ValueError(f"Duplicate source program {name!r} at JSONL line {line_number}.")
            names.add(name)
            count += 1
    if count == 0:
        raise ValueError(f"Source input file contains no programs: {path}")
    return {"input_records": count, "unique_programs": count}


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
        "review_focus": (
            "This is a vulnerability-claim adjudication task, not a general assignment review. "
            "Inspect EACH cppcheck static finding and EACH LLM source-audit finding independently. "
            "The absence of a cppcheck warning does not refute an LLM candidate. For every candidate, "
            "compare its cited source line and nearby source-line evidence with the actual code. "
            "Retain a candidate as observed or potential only when the cited source supports the "
            "defect; otherwise reject that candidate and explain why. Do not require the two analyzers "
            "to agree. Mention the vulnerability class in the finding text (for example buffer overflow, "
            "out-of-bounds access, uninitialized read, null dereference, integer overflow, memory leak, "
            "or use-after-free) so the adjudicated claim can be mapped to the benchmark label. "
            "Evaluate every candidate; do not silently omit a candidate merely because it came from the LLM. "
            "For EVERY accepted finding, cite at least one exact source-line evidence ID of the form "
            "E-SOURCE-L### whose facts.line is the line being claimed, in addition to any analyzer/model finding ID. "
            "Do not cite a different nearby line as a substitute. Do not report an include directive as a vulnerability line. "
            "If the candidate's claimed line does not show or directly support the defect, reject that candidate."
        ),
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
    if review_result.get("status") not in {"completed", "completed_with_warnings", "partial"}:
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
        if not isinstance(text, str) or not text.strip():
            continue
        cited_ids = claim.get("evidence_ids", [])
        if not isinstance(cited_ids, list):
            continue
        evidence = [by_id[ref] for ref in cited_ids if isinstance(ref, str) and ref in by_id]
        evidence_terms = []
        for entry in evidence:
            facts = entry.get("facts", {}) if isinstance(entry.get("facts"), dict) else {}
            evidence_terms.extend(str(facts.get(key, "")) for key in ("category", "id", "code_excerpt", "description"))
            evidence_terms.append(str(entry.get("statement", "")))
        classification_text = " ".join([text, *evidence_terms])
        if not VULNERABILITY_ID.search(classification_text):
            continue

        # An ID merely existing in the ledger is not proof. The claim must be
        # anchored to an exact source line: either cited directly, or reached
        # through a cited analyzer/model candidate whose line exists in the
        # ledger (its quoted excerpt is checked against that line below).
        def _is_line_entry(entry: dict[str, Any]) -> bool:
            return (entry.get("kind") == "source_line" and isinstance(entry.get("facts"), dict)
                    and isinstance(entry["facts"].get("line"), int)
                    and not isinstance(entry["facts"].get("line"), bool))

        source_lines = [entry for entry in evidence if _is_line_entry(entry)]
        if not source_lines:
            cited_candidate_lines = {
                entry["facts"]["line"] for entry in evidence
                if entry.get("kind") in {"static_finding", "model_audit_finding"}
                and isinstance(entry.get("facts"), dict)
                and isinstance(entry["facts"].get("line"), int)
                and not isinstance(entry["facts"].get("line"), bool)
            }
            source_lines = [
                entry for entry in ledger
                if _is_line_entry(entry) and entry["facts"]["line"] in cited_candidate_lines
            ]
        if not source_lines:
            continue
        # Do not accept a header/include line as the vulnerable operation.
        source_lines = [
            entry for entry in source_lines
            if not str(entry.get("facts", {}).get("code", "")).lstrip().startswith("#include")
        ]
        if not source_lines:
            continue
        source_lines.sort(key=lambda entry: entry["facts"]["line"])
        # If the claim also cites an analyzer/model candidate with a concrete
        # line, the exact source-line citation must agree with that candidate.
        candidate_entries = [
            entry for entry in evidence
            if entry.get("kind") in {"static_finding", "model_audit_finding"}
            and isinstance(entry.get("facts"), dict)
            and isinstance(entry["facts"].get("line"), int)
            and not isinstance(entry["facts"].get("line"), bool)
        ]
        candidate_lines: set[int] = set()
        for candidate in candidate_entries:
            facts = candidate.get("facts", {})
            candidate_line = facts.get("line")
            matching_source = next(
                (entry for entry in source_lines if entry.get("facts", {}).get("line") == candidate_line),
                None,
            )
            if matching_source is None:
                continue
            # For LLM-originated evidence, verify the stored quoted excerpt
            # against the actual source-line ledger fact, not merely the ID.
            if candidate.get("kind") == "model_audit_finding":
                model_excerpt = facts.get("code_excerpt")
                source_text = matching_source.get("facts", {}).get("code")
                compact = lambda value: re.sub(r"\s+", "", str(value or ""))
                model_compact, source_compact = compact(model_excerpt), compact(source_text)
                if not model_compact or not source_compact:
                    continue
                # Evidence source lines are capped at 180 characters; permit
                # either side to be a prefix only to account for that cap.
                if model_compact not in source_compact and source_compact not in model_compact:
                    continue
            candidate_lines.add(candidate_line)
        if candidate_entries:
            if not candidate_lines:
                continue
            source_lines = [entry for entry in source_lines if entry["facts"]["line"] in candidate_lines]
            if not source_lines:
                continue
        line = source_lines[0]["facts"]["line"]
        line_entry = next((
            entry for entry in evidence
            if entry.get("kind") in {"static_finding", "model_audit_finding"}
            and isinstance(entry.get("facts"), dict)
            and entry["facts"].get("line") == line
        ), None)
        error_type = None
        if line_entry:
            facts = line_entry.get("facts", {})
            error_type = facts.get("category") or facts.get("id")
        if not error_type:
            # Use the claim's own text as a label rather than borrowing the
            # category from a different line's analyzer finding.
            text_lower = text.casefold()
            for label in (
                "Buffer Overflow", "Out-of-bounds access", "Uninitialized Read",
                "Use of uninitialized variable", "Use After Free", "Double Free",
                "Integer Overflow", "Division by Zero", "Null Pointer Dereference",
                "Memory Leak", "Format String Vulnerability", "Stack Overflow",
            ):
                if label.casefold() in text_lower:
                    error_type = label
                    break
        evidence_ids = [entry.get("id") for entry in evidence if entry.get("id")]
        supported.append({
            "text": text[:400],
            "evidence_ids": evidence_ids[:8],
            "line": line,
            "error_type": error_type or "source_review_finding",
            "kind": claim.get("kind"),
        })
    if not supported:
        return False, None, None, []
    selected = supported[0]
    return True, selected.get("line"), selected.get("error_type"), supported


def _run_property_stage(source: str) -> tuple[list[dict[str, Any]], str]:
    """Deterministic syntax-tree property checks (no Docker, no LLM)."""
    try:
        result = check_c_properties(source)
    except Exception as exc:  # never let one parser edge case stop the program
        logger.warning("Syntax-tree property checks failed: %s", exc)
        return [], "error"
    findings = [
        {"line": item["line"], "severity": "warning", "id": item["check"], "description": item["description"],
         "tool": "syntax_tree", "family": item["family"]}
        for item in result.get("findings", [])
    ]
    return findings, str(result.get("status"))


def _merge_tool_findings(cppcheck: list[dict[str, Any]], properties: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """cppcheck plus syntax-tree findings, one per (line, family), earliest line first."""
    merged: dict[tuple[Any, str], dict[str, Any]] = {}
    for finding in [*cppcheck, *properties]:
        family = error_family(finding.get("family") or f"{finding.get('id', '')} {finding.get('description', '')}")
        merged.setdefault((finding.get("line"), family), {"tool": "cppcheck", **finding, "family": family})
    return sorted(merged.values(), key=lambda f: (f.get("line") or 0, f["family"]))


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
        audit = result.get("audit")
        if isinstance(audit, dict) and isinstance(audit.get("findings"), list):
            # Findings outside every safety family (style, portability, "unexpected
            # behaviour") are kept for review but are not vulnerability evidence.
            safety = [f for f in audit["findings"] if not isinstance(f, dict) or error_family(f.get("category")) != "other"]
            other = [f for f in audit["findings"] if isinstance(f, dict) and error_family(f.get("category")) == "other"]
            if other:
                audit["findings"] = safety
                audit["non_safety_findings"] = other
        return result
    except Exception as exc:  # Isolate network, server, and parser errors to this sample.
        logger.exception("LLM source audit failed for one FormAI record")
        return {"status": "unavailable", "model": settings.MODEL, "error": f"{type(exc).__name__}: {exc}"}


def _vulnerability_family(value: Any) -> str:
    """Map common LLM/cppcheck labels to a compact defect family."""
    text = re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
    aliases = (
        ("uninitialized", ("uninit", "uninitialized")),
        ("bounds", ("outofbounds", "arrayindex", "arraybounds", "boundserror")),
        ("division_by_zero", ("divisionbyzero", "dividebyzero")),
        ("integer_overflow", ("integeroverflow", "signedoverflow", "integertruncation")),
        ("null_dereference", ("nullpointer", "null dereference", "nullpointeroutofmemory")),
        ("use_after_free", ("useafterfree", "danglingpointer")),
        ("double_free", ("doublefree",)),
        ("memory_leak", ("memleak", "memoryleak", "resourceleak")),
        ("unsafe_input", ("invalidscanf", "scanf", "bufferoverflow")),
        ("format_string", ("formatstring",)),
        ("stack_overflow", ("stackoverflow",)),
    )
    for family, patterns in aliases:
        if any(pattern in text for pattern in patterns):
            return family
    return text


def _rsi_confirms_llm_finding(
    finding: dict[str, Any], rsi_findings: list[dict[str, Any]]
) -> bool:
    line = finding.get("line")
    category = _vulnerability_family(finding.get("category"))
    for reviewed in rsi_findings:
        if not isinstance(reviewed, dict):
            continue
        reviewed_line = reviewed.get("line")
        reviewed_category = _vulnerability_family(reviewed.get("error_type"))
        if reviewed_category in {"", "sourcereviewfinding"}:
            reviewed_category = _vulnerability_family(reviewed.get("text"))
        same_category = bool(category and reviewed_category and category == reviewed_category)
        same_line = isinstance(line, int) and isinstance(reviewed_line, int) and line == reviewed_line
        if same_category and (same_line or reviewed_line is None or line is None):
            return True
    return False


def _select_hybrid_prediction(
    *,
    mode: str,
    tool_status: str,
    tool_findings: list[dict[str, Any]],
    llm_audit: dict[str, Any] | None,
    rsi_review: dict[str, Any] | None = None,
    rsi_findings: list[dict[str, Any]] | None = None,
) -> tuple[bool | None, int | None, str | None, str]:
    """Fuse cppcheck positives, RSI-adjudicated LLM candidates, and uncertainty."""
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
    rsi_findings = rsi_findings or []
    # In hybrid_rsi, the source auditor proposes claims and RSI must adjudicate
    # each LLM-originated candidate against the supplied source/analyzer ledger.
    # The regular hybrid and llm_only modes can use the source-validated LLM
    # findings directly because they do not promise the additional RSI pass.
    if mode == "hybrid_rsi":
        llm_findings_usable = [
            finding for finding in llm_findings
            if isinstance(finding, dict) and _rsi_confirms_llm_finding(finding, rsi_findings)
        ]
    else:
        llm_findings_usable = [finding for finding in llm_findings if isinstance(finding, dict)]
    llm_positive = bool(llm_findings_usable) and llm_result_usable
    unresolved_summary = bool((llm_audit or {}).get("unresolved_positive_summary"))
    unresolved_invalid_candidates = bool((llm_audit or {}).get("unresolved_invalid_candidates"))
    unconfirmed_llm_candidates = (
        bool(llm_findings) and len(llm_findings_usable) < len(llm_findings) and mode == "hybrid_rsi"
    )
    rsi_done = mode != "hybrid_rsi" or (
        isinstance(rsi_review, dict)
        and rsi_review.get("status") in {"completed", "completed_with_warnings"}
        and isinstance(rsi_review.get("claim_validation"), dict)
        and rsi_review["claim_validation"].get("status") in {"completed", "completed_with_warnings"}
    )
    # A partial review (e.g. the final synthesis was truncated) still carries
    # evidence-validated claims from an earlier pass; those support a positive.
    # Only a negative needs the review to have fully completed (rsi_done).
    rsi_positive = bool(rsi_findings) and (
        rsi_done
        or (
            isinstance(rsi_review, dict)
            and rsi_review.get("status") == "partial"
            and isinstance(rsi_review.get("claim_validation"), dict)
            and rsi_review["claim_validation"].get("status") in {"completed", "completed_with_warnings"}
        )
    )

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

    # Do not turn a positive LLM summary with failed line-level extraction into
    # a safe negative. Keep it inconclusive unless another method validates a
    # concrete positive claim.
    incomplete_combined = mode in {"hybrid", "hybrid_rsi"} and not (tool_done and llm_done and rsi_done)
    if (
        unresolved_summary or unresolved_invalid_candidates or unconfirmed_llm_candidates or incomplete_combined
    ) and not (tool_positive or llm_positive or rsi_positive):
        return None, None, None, "inconclusive"

    # OR fusion combines independently validated positive signals.
    predicted = tool_positive or llm_positive or rsi_positive
    if rsi_positive:
        rsi_first = rsi_findings[0]
        line, error_type = rsi_first.get("line"), rsi_first.get("error_type")
    elif llm_positive:
        first = llm_findings_usable[0]
        line, error_type = first.get("line"), first.get("category")
    else:
        line, error_type = tool_line, tool_type

    all_expected_complete = tool_done and llm_status == "completed"
    if mode == "hybrid_rsi":
        all_expected_complete = all_expected_complete and rsi_done
    status = (
        "completed"
        if all_expected_complete and not unconfirmed_llm_candidates and not unresolved_summary and not unresolved_invalid_candidates
        else "completed_with_warnings"
    )
    return predicted, line, error_type, status


def _accepted_findings(
    mode: str,
    tool_findings: list[dict[str, Any]],
    llm_findings: list[dict[str, Any]],
    rsi_findings: list[dict[str, Any]],
    verified_claims: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Findings the prediction actually relies on, mapped to benchmark families."""
    accepted: list[dict[str, Any]] = []
    if mode != "llm_only":
        for finding in tool_findings:
            accepted.append({
                "line": finding.get("line"), "source": finding.get("tool", "cppcheck"),
                "error_type": finding.get("id"),
                "family": error_family(finding.get("family") or f"{finding.get('id', '')} {finding.get('description', '')}"),
            })
    if mode != "tools_only":
        if mode == "hybrid_verify":
            usable = list(verified_claims or [])
        elif mode == "hybrid_rsi":
            usable = [f for f in llm_findings if isinstance(f, dict) and _rsi_confirms_llm_finding(f, rsi_findings)]
        else:
            usable = [f for f in llm_findings if isinstance(f, dict)]
        for finding in usable:
            accepted.append({
                "line": finding.get("line"), "source": "llm", "error_type": finding.get("category"),
                "family": error_family(f"{finding.get('category', '')} {finding.get('description', '')}"
                                       if finding.get("category") not in FAMILIES else finding.get("category")),
            })
        for finding in rsi_findings:
            accepted.append({
                "line": finding.get("line"), "source": "evidence_review", "error_type": finding.get("error_type"),
                "family": error_family(finding.get("error_type") if finding.get("error_type") not in {None, "source_review_finding"}
                                       else finding.get("text")),
            })
    unique: dict[tuple[Any, str], dict[str, Any]] = {}
    for finding in accepted:
        unique.setdefault((finding.get("line"), finding["family"]), finding)
    return sorted(unique.values(), key=lambda f: (f.get("line") or 0, f["family"]))


def _select_verified_prediction(
    *,
    tool_status: str,
    tool_findings: list[dict[str, Any]],
    llm_audit: dict[str, Any] | None,
    verification: dict[str, Any] | None,
) -> tuple[bool | None, int | None, str | None, str]:
    """hybrid_verify fusion: deterministic tool findings plus verified LLM claims.

    Vulnerable when a static tool reports a safety finding or a claim survives
    verification. Safe only when the static tools, the audit and the
    verification all completed and nothing survived; otherwise inconclusive.
    """
    audit_status = (llm_audit or {}).get("status")
    audit_done = audit_status in {"completed", "completed_with_warnings"} and not int((llm_audit or {}).get("chunks_failed", 0) or 0)
    verification_done = verification is None or verification.get("status") == "completed"
    tool_done = tool_status == "completed"
    confirmed = list((verification or {}).get("confirmed_claims") or [])
    positives = [(f.get("line") or 0, f.get("id") or f.get("family")) for f in tool_findings]
    positives += [(c.get("line") or 0, c.get("category")) for c in confirmed]
    complete = tool_done and audit_done and verification_done
    if positives:
        line, kind = min(positives, key=lambda item: item[0])
        return True, line or None, kind, "completed" if complete else "completed_with_warnings"
    if complete:
        return False, None, None, "completed"
    return None, None, None, "inconclusive"


class PredictionSession:
    """Shared LLM server / Docker sandbox for predicting many programs.

    ``predict`` handles one program at a time so callers can report progress;
    a failure in one stage of one program never stops the others.
    """

    def __init__(
        self,
        *,
        mode: str,
        tool_timeout: float = settings.FORM_AI_DEFAULT_TOOL_TIMEOUT_S,
        max_llm_findings: int = settings.FORM_AI_DEFAULT_MAX_FINDINGS,
        rsi_iterations: int = settings.FORM_AI_DEFAULT_RSI_ITERATIONS,
        memory_limit: str = settings.FORM_AI_DEFAULT_MEMORY_LIMIT,
        cpu_limit: float = settings.DOCKER_CPU_LIMIT,
    ) -> None:
        if mode not in settings.FORM_AI_PREDICTION_MODES:
            raise ValueError(f"Unknown mode {mode!r}; choose one of {settings.FORM_AI_PREDICTION_MODES}.")
        self.mode = mode
        self.tool_timeout = float(tool_timeout)
        self.max_llm_findings = int(max_llm_findings)
        self.rsi_iterations = int(rsi_iterations)
        self.memory_limit = memory_limit
        self.cpu_limit = float(cpu_limit)
        self.needs_llm = mode in {"llm_only", "hybrid", "hybrid_rsi", "hybrid_verify"}
        self.needs_tools = mode in {"tools_only", "hybrid", "hybrid_rsi", "hybrid_verify"}
        self.process = None
        self.sandbox = None
        self._sandbox_manager = None
        self.server_error: str | None = None
        self.sandbox_error: str | None = None

    def __enter__(self) -> "PredictionSession":
        # LLM setup is independent of Docker setup: a failure in one is recorded
        # per program and the other stage still runs.
        if self.needs_llm and not _server_ready():
            try:
                self.process = initiate_server()
            except Exception as exc:
                self.server_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Local LLM server unavailable: %s; per-record outputs will still be written.", self.server_error)
        if self.needs_tools:
            try:
                from grading_agent.sandbox.docker_sandbox import DockerSandbox
                self._sandbox_manager = DockerSandbox(memory_limit=self.memory_limit, cpu_limit=self.cpu_limit)
                self.sandbox = self._sandbox_manager.__enter__()
            except Exception as exc:
                self.sandbox = None
                self.sandbox_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Docker sandbox unavailable; combined mode will continue with LLM analysis: %s", self.sandbox_error)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self.sandbox is not None and self._sandbox_manager is not None:
            with suppress(Exception):
                self._sandbox_manager.__exit__(None, None, None)
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=settings.LLAMA_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def predict(self, item: dict[str, Any], on_stage: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Predict one program: ``item`` holds ``file_name`` and ``source_code`` only."""
        def stage(name: str) -> None:
            if on_stage is not None:
                with suppress(Exception):
                    on_stage(name)

        mode = self.mode
        source = item.get("source_code", "")
        if item.get("_input_error"):
            return {
                "file_name": item["file_name"], "predicted_vulnerable": None, "predicted_line": None,
                "predicted_error_type": None, "predicted_findings": [], "prediction_status": "inconclusive",
                "mode": mode, "prediction_source": "none", "tool_status": "not_run", "tool_findings": [],
                "llm_findings": [], "llm_audit_status": "not_run", "rsi_review_status": "not_run",
                "rsi_findings": [], "stage_errors": {"input": item["_input_error"]},
            }
        started = time.monotonic()
        timings: dict[str, float] = {}
        tool_findings: list[dict[str, Any]] = []
        tool_status = "not_run"
        property_status = "not_run"
        tool_error = None
        llm_audit: dict[str, Any] | None = None
        rsi_review: dict[str, Any] | None = None
        rsi_findings: list[dict[str, Any]] = []
        rsi_error = None
        rsi_note = None
        verification: dict[str, Any] | None = None

        if self.needs_tools:
            stage("static analysis")
            clock = time.monotonic()
            tool_findings, tool_status, tool_error = _run_tool_stage(self.sandbox, source, self.tool_timeout)
            if tool_status == "unavailable" and self.sandbox_error:
                tool_error = self.sandbox_error
            property_findings, property_status = _run_property_stage(source)
            tool_findings = _merge_tool_findings(tool_findings, property_findings)
            timings["tools_s"] = round(time.monotonic() - clock, 2)

        if self.needs_llm:
            stage("LLM source audit")
            clock = time.monotonic()
            llm_audit = _run_llm_stage(source, self.max_llm_findings, self.server_error)
            timings["llm_audit_s"] = round(time.monotonic() - clock, 2)

        if mode == "hybrid_verify" and (llm_audit or {}).get("status") in {"completed", "completed_with_warnings"}:
            claims = ((llm_audit or {}).get("audit") or {}).get("findings") or []
            if claims:
                stage("claim verification")
                clock = time.monotonic()
                try:
                    verification = verify_vulnerability_claims(source, claims, max_passes=self.rsi_iterations)
                except Exception as exc:  # isolate this program
                    logger.warning("Claim verification failed for %s: %s", item["file_name"], exc)
                    verification = {"status": "incomplete", "error": f"{type(exc).__name__}: {exc}", "confirmed_claims": []}
                timings["claim_verification_s"] = round(time.monotonic() - clock, 2)

        if mode == "hybrid_rsi" and (
            (llm_audit or {}).get("status") in {"completed", "completed_with_warnings"}
            or tool_status == "completed"
            or bool(tool_findings)
        ):
            # Review runs when either primary stage produced usable evidence. It
            # adjudicates LLM candidates; cppcheck positives are retained.
            stage("evidence review (RSI)")
            clock = time.monotonic()
            try:
                review_report = _build_rsi_review_report(item, source, tool_findings, tool_status, llm_audit or {})
                rsi_ledger = build_evidence_ledger(review_report, {"metrics": []})
                rsi_review = review_evidence(review_report, rules={"metrics": []}, max_iterations=self.rsi_iterations)
                _, _, _, rsi_findings = _prediction_from_rsi_review(rsi_review, rsi_ledger)
                if not rsi_findings:
                    llm_candidates = (llm_audit or {}).get("audit", {}).get("findings", [])
                    if isinstance(llm_candidates, list) and llm_candidates:
                        rsi_note = (
                            "RSI completed but did not validate the LLM candidate finding(s). "
                            "The candidates are retained in the record; absent another validated "
                            "positive, the prediction is inconclusive rather than safe."
                        )
                    elif (llm_audit or {}).get("unresolved_positive_summary"):
                        rsi_note = (
                            "The LLM summary suggested possible vulnerabilities, but neither the "
                            "source audit nor RSI established a line-grounded claim. Review manually; "
                            "the prediction is inconclusive rather than safe."
                        )
                    else:
                        rsi_note = "Evidence review completed without additional validated vulnerability claims; primary results are retained."
            except Exception as exc:
                rsi_error = f"{type(exc).__name__}: {exc}"
                rsi_review = {"status": "unavailable", "error": rsi_error}
                logger.warning("RSI evidence review failed for %s: %s", item["file_name"], exc)
            timings["evidence_review_s"] = round(time.monotonic() - clock, 2)
        elif mode == "hybrid_rsi":
            rsi_review = {
                "status": "skipped",
                "reason": "No primary analysis completed and no usable tool findings were available; primary-stage status is retained.",
            }

        if mode == "hybrid_verify":
            predicted, line, error_type, status = _select_verified_prediction(
                tool_status=tool_status, tool_findings=tool_findings, llm_audit=llm_audit, verification=verification,
            )
        else:
            predicted, line, error_type, status = _select_hybrid_prediction(
                mode=mode, tool_status=tool_status, tool_findings=tool_findings,
                llm_audit=llm_audit, rsi_review=rsi_review, rsi_findings=rsi_findings,
            )
        llm_findings: list[dict[str, Any]] = []
        audit_body = (llm_audit or {}).get("audit", {})
        if (llm_audit or {}).get("status") in {"completed", "completed_with_warnings"}:
            if isinstance(audit_body, dict) and isinstance(audit_body.get("findings"), list):
                llm_findings = audit_body["findings"]
        llm_findings_not_confirmed_by_rsi = []
        if mode == "hybrid_rsi":
            llm_findings_not_confirmed_by_rsi = [
                finding for finding in llm_findings
                if isinstance(finding, dict) and not _rsi_confirms_llm_finding(finding, rsi_findings)
            ]
        audit_body = audit_body if isinstance(audit_body, dict) else {}
        timings["total_s"] = round(time.monotonic() - started, 2)
        return {
            "file_name": item["file_name"],
            "predicted_vulnerable": predicted,
            "predicted_line": line,
            "predicted_error_type": error_type,
            "predicted_family": error_family(error_type) if predicted else None,
            "predicted_findings": _accepted_findings(
                mode, tool_findings, llm_findings, rsi_findings, (verification or {}).get("confirmed_claims"),
            ) if predicted is not None else [],
            "prediction_status": status,
            "mode": mode,
            "prediction_source": _prediction_source(mode, tool_status, tool_findings, llm_audit, rsi_findings),
            "tool_status": tool_status,
            "syntax_tree_status": property_status,
            "tool_findings": tool_findings,
            "tool_error": tool_error,
            "llm_findings": llm_findings,
            "llm_findings_not_confirmed_by_rsi": llm_findings_not_confirmed_by_rsi,
            "llm_audit_status": (llm_audit or {}).get("status", "not_run"),
            "llm_audit_summary": audit_body.get("summary"),
            "llm_audit_limitations": audit_body.get("limitations", []),
            "llm_rejected_findings": audit_body.get("rejected_findings", []),
            "llm_finding_recovery_attempted": (llm_audit or {}).get("finding_recovery_attempted", False),
            "llm_finding_recovery_succeeded": (llm_audit or {}).get("finding_recovery_succeeded"),
            "llm_finding_recovery_completed": (llm_audit or {}).get("finding_recovery_completed"),
            "llm_unresolved_positive_summary": (llm_audit or {}).get("unresolved_positive_summary", False),
            "llm_unresolved_invalid_candidates": (llm_audit or {}).get("unresolved_invalid_candidates", False),
            "llm_finding_recovery_error": (llm_audit or {}).get("finding_recovery_error"),
            "llm_error": (llm_audit or {}).get("error"),
            "rsi_review_status": (rsi_review or {}).get("status", "not_run"),
            "rsi_preserved_prior_findings": (rsi_review or {}).get("preserved_prior_findings", 0),
            "rsi_findings": rsi_findings,
            "rsi_error": rsi_error or (rsi_review or {}).get("error"),
            "rsi_note": rsi_note,
            "claim_verification": {key: value for key, value in (verification or {}).items() if key != "confirmed_claims"}
            if verification is not None else None,
            "timings": timings,
            "stage_errors": {
                key: value for key, value in {
                    "tools": tool_error,
                    "llm": (llm_audit or {}).get("error"),
                    "rsi": rsi_error or (rsi_review or {}).get("error"),
                    "verification": (verification or {}).get("error"),
                }.items() if value
            },
        }


def run(args: argparse.Namespace) -> None:
    """CLI entry: predict every program in a source-only JSONL file."""
    input_manifest = _validate_predictor_input(args.inputs)
    output_path = args.output
    count = 0
    errors_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PredictionSession(
        mode=args.mode,
        tool_timeout=args.tool_timeout,
        max_llm_findings=args.max_llm_findings,
        rsi_iterations=args.rsi_iterations,
        memory_limit=args.memory_limit,
        cpu_limit=getattr(args, "cpu_limit", settings.DOCKER_CPU_LIMIT),
    ) as session, output_path.open("w", encoding="utf-8") as output:
        for item in _iter_inputs(args.inputs):
            if args.max_samples is not None and count >= args.max_samples:
                break
            prediction = session.predict(item)
            if prediction["predicted_vulnerable"] is None:
                errors_count += 1
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            output.flush()
            count += 1
            if count % 25 == 0:
                logger.info("Processed %d FormAI samples.", count)

    run_manifest = {
        **input_manifest,
        "prediction_records": count,
        "missing_prediction_records": max(0, input_manifest["input_records"] - count),
        "max_samples": args.max_samples,
        "complete_input_processed": count == input_manifest["input_records"],
        "mode": args.mode,
    }
    output_path.with_suffix(output_path.suffix + ".run.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    logger.info(
        "Wrote %d/%d predictions to %s (%d without any completed primary analysis stage).",
        count, input_manifest["input_records"], output_path, errors_count,
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
    if mode == "hybrid_verify" and "llm" in sources:
        sources[sources.index("llm")] = "llm(verified)"
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
    parser.add_argument("--cpu-limit", type=float, default=settings.DOCKER_CPU_LIMIT, help="Docker CPU quota in cores.")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.tool_timeout <= 0:
        parser.error("--tool-timeout must be positive")
    if args.max_llm_findings < 1:
        parser.error("--max-llm-findings must be positive")
    if not 1 <= args.rsi_iterations <= settings.LLM_REVIEW_MAX_ITERATIONS:
        parser.error(f"--rsi-iterations must be between 1 and {settings.LLM_REVIEW_MAX_ITERATIONS}")
    if not 0.25 <= args.cpu_limit <= 16:
        parser.error("--cpu-limit must be between 0.25 and 16 cores")
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT_BENCHMARK)
    run(args)


if __name__ == "__main__":
    main()
