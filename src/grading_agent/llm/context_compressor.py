"""Loss-aware compression using the configured grading model itself.

Only natural-language summaries are compressible. Submitted source, source line
numbers, finding identities, analyzer measurements, and evidence IDs remain
protected. The model is asked to shorten only sufficiently long prose fields;
strict per-field output limits make useful reductions enforceable rather than
merely requested. Deterministic compaction remains the safe fallback.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from contextvars import ContextVar
from typing import Any

import requests

from grading_agent.config import settings

logger = logging.getLogger(__name__)

# Shared with reviewer.py so compression requests are visible in run metadata.
COMPRESSION_EVENTS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "grading_context_compression_events", default=None
)

# Avoid spending a model call on short fields that cannot materially shrink.
# Deterministic fallbacks cap semantic fields below this value, preventing the
# same unchanged content from triggering another compression request next pass.
MIN_COMPRESSIBLE_FIELD_CHARS = 240
MIN_EXPECTED_SAVED_CHARS = 120
RETAIN_RATIO = 0.60
MAX_COMPRESSED_FIELD_CHARS = 180
MAX_COMPRESSION_RESPONSE_TOKENS = 768


def _record_compression_event(stage: str, metadata: dict[str, Any]) -> None:
    events = COMPRESSION_EVENTS.get()
    if not isinstance(events, list):
        return
    events.append({
        "stage": stage,
        "method": metadata.get("method", "unchanged"),
        "model": settings.runtime_value("MODEL"),
        "model_calls": int(metadata.get("model_calls", 0)),
        "input_chars": int(metadata.get("input_chars", 0)),
        "output_chars": int(metadata.get("output_chars", 0)),
        "saved_chars": int(metadata.get("saved_chars", 0)),
        "potential_saved_chars": int(metadata.get("potential_saved_chars", 0)),
        "compressible_chars": int(metadata.get("compressible_chars", 0)),
        "latency_seconds": float(metadata.get("latency_seconds", 0.0)),
        "reason": metadata.get("reason"),
        "error": metadata.get("error"),
    })


def _finish_compression_event(stage: str, metadata: dict[str, Any], started: float) -> None:
    metadata["latency_seconds"] = round(time.perf_counter() - started, 3)
    _record_compression_event(stage, metadata)


def _target_length(value: str) -> int:
    """Return a firm output cap that demands a meaningful reduction."""
    return max(80, min(MAX_COMPRESSED_FIELD_CHARS, math.floor(len(value) * RETAIN_RATIO)))


def _parse_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip().lstrip("\ufeff")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:index + 1])
                    except json.JSONDecodeError:
                        return None
                    return parsed if isinstance(parsed, dict) else None
        return None


def _same_model_chat(
    system_prompt: str,
    user_payload: dict[str, Any],
    response_schema: dict[str, Any],
    *,
    response_token_budget: int | None = None,
) -> dict[str, Any] | None:
    """Run one schema-constrained compression request through the grading model."""
    if not settings.runtime_value("LLM_CONTEXT_COMPRESSION_ENABLED"):
        return None

    schema_text = json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"))
    messages = [
        {
            "role": "system",
            "content": (
                system_prompt
                + "\nReturn exactly one JSON object matching this schema, with no Markdown or prose:\n"
                + schema_text
            ),
        },
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    configured_budget = int(settings.runtime_value("LLM_CONTEXT_COMPRESSION_MAX_TOKENS"))
    requested_budget = response_token_budget or configured_budget
    body: dict[str, Any] = {
        "model": str(settings.runtime_value("MODEL")),
        "stream": False,
        "temperature": 0.0,
        "seed": settings.SEED,
        # A response schema with several summaries can need more than a tiny
        # 256-token cap. Keep the increase bounded for low-memory deployments.
        "max_tokens": min(
            MAX_COMPRESSION_RESPONSE_TOKENS,
            max(configured_budget, requested_budget),
        ),
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "schema": response_schema,
        },
        "reasoning_effort": "none",
        "reasoning_format": "none",
    }

    response = requests.post(
        f"{settings.LLAMA_BASE_URL}{settings.LLAMA_API_PATH}",
        json=body,
        timeout=settings.runtime_value("LLAMA_REQUEST_TIMEOUT"),
    )
    response.raise_for_status()
    data = response.json()
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "Same-model context compression returned an unexpected chat response."
        ) from exc

    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        )
    parsed = _parse_json_object(content)
    if parsed is None:
        choice = data.get("choices", [{}])[0] if isinstance(data, dict) else {}
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        raise ValueError(
            "Same-model compression did not return schema-constrained final JSON "
            f"(finish_reason={choice.get('finish_reason', 'unknown')}, "
            f"completion_tokens={usage.get('completion_tokens', 'unknown')})."
        )
    return parsed


def _deterministic_review_compaction(
    review: dict[str, Any],
    *,
    max_description_chars: int = 360,
    max_assessment_chars: int = 700,
) -> dict[str, Any]:
    """Compact a previous RSI response without dropping findings or evidence."""
    findings = review.get("findings", [])
    compact_findings: list[dict[str, Any]] = []
    if isinstance(findings, list):
        for item in findings:
            if not isinstance(item, dict):
                continue
            compact_findings.append({
                # Finding identity/anchors are copied exactly; only prose
                # descriptions are eligible for compression.
                "metric": item.get("metric", "unmapped"),
                "severity": item.get("severity", "medium"),
                "title": item.get("title", "Potential issue"),
                "description": str(item.get("description", item.get("text", "")))[:max_description_chars],
                "line_start": item.get("line_start"),
                "line_end": item.get("line_end", item.get("line_start")),
                "evidence_ids": list(item.get("evidence_ids", []))
                if isinstance(item.get("evidence_ids", []), list) else [],
                "confidence": item.get("confidence", 0.5),
            })
    questions = review.get("unresolved_questions", [])
    return {
        "overall_assessment": str(
            review.get("overall_assessment", review.get("summary", "RSI review state retained."))
        )[:max_assessment_chars],
        "findings": compact_findings,
        "unresolved_questions": [str(item)[:180] for item in questions[:5]]
        if isinstance(questions, list) else [],
    }


def compress_review_history(review: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compress only long review prose while preserving every finding anchor."""
    started = time.perf_counter()
    original = _deterministic_review_compaction(review)
    original_text = json.dumps(original, ensure_ascii=False)
    threshold = int(settings.runtime_value("LLM_CONTEXT_COMPRESSION_THRESHOLD_CHARS"))
    metadata: dict[str, Any] = {
        "enabled": bool(settings.runtime_value("LLM_CONTEXT_COMPRESSION_ENABLED")),
        "method": "unchanged",
        "model_calls": 0,
        "model": settings.runtime_value("MODEL"),
        "input_chars": len(original_text),
        "output_chars": len(original_text),
        "saved_chars": 0,
        "latency_seconds": 0.0,
    }

    if not metadata["enabled"]:
        metadata["reason"] = "same-model compression disabled"
        _finish_compression_event("review_history", metadata, started)
        return original, metadata
    if len(original_text) <= threshold:
        metadata["reason"] = "history below compression threshold"
        _finish_compression_event("review_history", metadata, started)
        return original, metadata

    assessment = original["overall_assessment"]
    compress_assessment = len(assessment) > MIN_COMPRESSIBLE_FIELD_CHARS
    assessment_target = _target_length(assessment) if compress_assessment else len(assessment)

    finding_targets: dict[str, int] = {}
    finding_inputs: dict[str, str] = {}
    for index, item in enumerate(original["findings"]):
        description = str(item.get("description", ""))
        if len(description) > MIN_COMPRESSIBLE_FIELD_CHARS:
            field_name = f"finding_{index}"
            finding_inputs[field_name] = description
            finding_targets[field_name] = _target_length(description)

    selected_lengths = ([len(assessment)] if compress_assessment else []) + [
        len(value) for value in finding_inputs.values()
    ]
    target_lengths = ([assessment_target] if compress_assessment else []) + list(finding_targets.values())
    compressible_chars = sum(selected_lengths)
    potential_saved = sum(
        source_length - target_length
        for source_length, target_length in zip(selected_lengths, target_lengths)
    )
    metadata.update({
        "compressible_chars": compressible_chars,
        "potential_saved_chars": potential_saved,
    })

    if not selected_lengths or potential_saved < MIN_EXPECTED_SAVED_CHARS:
        metadata["reason"] = "not enough long review prose to justify a model call"
        _finish_compression_event("review_history", metadata, started)
        return original, metadata

    properties: dict[str, Any] = {}
    required: list[str] = []
    if compress_assessment:
        properties["overall_assessment"] = {
            "type": "string", "minLength": 1, "maxLength": assessment_target
        }
        required.append("overall_assessment")
    if finding_inputs:
        properties["finding_descriptions"] = {
            "type": "object",
            "properties": {
                name: {"type": "string", "minLength": 1, "maxLength": finding_targets[name]}
                for name in finding_inputs
            },
            "required": list(finding_inputs),
            "additionalProperties": False,
        }
        required.append("finding_descriptions")
    response_schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }

    try:
        payload: dict[str, Any] = {
            "target_total_chars": potential_saved,
        }
        if compress_assessment:
            payload["overall_assessment"] = assessment
        if finding_inputs:
            payload["finding_descriptions"] = finding_inputs

        metadata["model_calls"] = 1
        compressed = _same_model_chat(
            "Shorten the supplied review prose for the next RSI pass. Rewrite each supplied field "
            "substantially more concisely; do not copy it verbatim. Preserve the defect meaning, "
            "certainty, negations, conditions, and important technical details. Do not invent facts. "
            "The JSON schema imposes a strict maximum length for every field. Finding descriptions "
            "are keyed by finding identity; do not add, remove, or rename keys. All metrics, titles, "
            "severities, locations, confidence values, and evidence IDs are retained by the caller.",
            payload,
            response_schema,
            response_token_budget=max(256, math.ceil((compressible_chars - potential_saved) / 3) + 128),
        )
        if not isinstance(compressed, dict):
            raise ValueError("the grading model returned no structured review compression.")

        result_findings = [dict(item) for item in original["findings"]]
        if finding_inputs:
            candidate_descriptions = compressed.get("finding_descriptions")
            if not isinstance(candidate_descriptions, dict):
                raise ValueError("the grading model omitted the finding_descriptions object.")
            if set(candidate_descriptions) != set(finding_inputs):
                raise ValueError("same-model compression changed the finding description keys.")
            for field_name, source_description in finding_inputs.items():
                candidate = candidate_descriptions[field_name]
                if not isinstance(candidate, str) or not candidate.strip():
                    raise ValueError(f"empty or invalid compressed description: {field_name}")
                if len(candidate.strip()) > finding_targets[field_name]:
                    raise ValueError(f"compressed description exceeded its length budget: {field_name}")
                finding_index = int(field_name.removeprefix("finding_"))
                result_findings[finding_index]["description"] = candidate.strip()

        result_assessment = assessment
        if compress_assessment:
            candidate_assessment = compressed.get("overall_assessment")
            if not isinstance(candidate_assessment, str) or not candidate_assessment.strip():
                raise ValueError("the grading model omitted a valid overall_assessment.")
            if len(candidate_assessment.strip()) > assessment_target:
                raise ValueError("compressed overall_assessment exceeded its length budget.")
            result_assessment = candidate_assessment.strip()

        result = {
            "overall_assessment": result_assessment,
            "findings": result_findings,
            "unresolved_questions": original["unresolved_questions"],
        }
        result_text = json.dumps(result, ensure_ascii=False)
        saved_chars = len(original_text) - len(result_text)
        if saved_chars < MIN_EXPECTED_SAVED_CHARS:
            raise ValueError(
                f"compression saved only {saved_chars} chars; "
                f"at least {MIN_EXPECTED_SAVED_CHARS} are required."
            )
        metadata.update({
            "method": "same_model",
            "output_chars": len(result_text),
            "saved_chars": saved_chars,
            "reason": "model compression met per-field and total reduction requirements",
        })
        return result, metadata
    except Exception as exc:  # Compression must never break grading.
        # Shorten the prose enough that the same uncompressible history is not
        # sent to the model repeatedly on the next RSI pass. All finding anchors
        # and evidence IDs remain intact.
        compact = _deterministic_review_compaction(
            review, max_description_chars=200, max_assessment_chars=220
        )
        compact_text = json.dumps(compact, ensure_ascii=False)
        metadata.update({
            "method": "deterministic_fallback",
            "output_chars": len(compact_text),
            "saved_chars": max(0, len(original_text) - len(compact_text)),
            "error": f"{type(exc).__name__}: {exc}"[:300],
        })
        logger.info(
            "same-model review compression rejected; deterministic compaction used: %s", exc
        )
        return compact, metadata
    finally:
        _finish_compression_event("review_history", metadata, started)


def compress_semantic_sections(
    payload: dict[str, Any],
    *,
    target_chars: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compress long program-understanding prose; keep structured evidence immutable."""
    started = time.perf_counter()
    copied = dict(payload)
    serialized = json.dumps(copied, ensure_ascii=False)
    metadata: dict[str, Any] = {
        "method": "unchanged",
        "input_chars": len(serialized),
        "output_chars": len(serialized),
        "saved_chars": 0,
        "potential_saved_chars": 0,
        "compressible_chars": 0,
        "model_calls": 0,
        "enabled": bool(settings.runtime_value("LLM_CONTEXT_COMPRESSION_ENABLED")),
        "model": settings.runtime_value("MODEL"),
        "latency_seconds": 0.0,
    }

    def finish(result: dict[str, Any], reason: str | None = None):
        if reason:
            metadata["reason"] = reason
        _finish_compression_event("program_understanding", metadata, started)
        return result, metadata

    if len(serialized) <= target_chars:
        return finish(copied, "payload below compression target")

    original_understanding = copied.get("program_understanding")
    if not isinstance(original_understanding, dict):
        compacted, metadata = _deterministic_semantic_compaction(copied, metadata)
        return finish(compacted, "program_understanding object unavailable")

    prose_keys = (
        "actual_behavior",
        "program_purpose",
        "intended_behavior",
        "contract_alignment",
        "algorithm_overview",
    )
    prose = {
        key: str(original_understanding.get(key, ""))[:4_000]
        for key in prose_keys
        if isinstance(original_understanding.get(key), str)
        and len(str(original_understanding.get(key, ""))) > MIN_COMPRESSIBLE_FIELD_CHARS
    }
    if not metadata["enabled"]:
        compacted, metadata = _deterministic_semantic_compaction(copied, metadata)
        return finish(compacted, "same-model compression disabled")
    if not prose:
        # The payload may be large because of protected structured data. Do not
        # spend an LLM call trying to compress data the compressor must preserve.
        compacted, metadata = _deterministic_semantic_compaction(copied, metadata)
        return finish(compacted, "no sufficiently long compressible prose fields")

    target_lengths = {key: _target_length(value) for key, value in prose.items()}
    compressible_chars = sum(len(value) for value in prose.values())
    potential_saved = sum(len(prose[key]) - target_lengths[key] for key in prose)
    metadata.update({
        "compressible_chars": compressible_chars,
        "potential_saved_chars": potential_saved,
    })
    if potential_saved < MIN_EXPECTED_SAVED_CHARS:
        compacted, metadata = _deterministic_semantic_compaction(copied, metadata)
        return finish(compacted, "not enough compressible prose to justify a model call")

    prose_schema = {
        "type": "object",
        "properties": {
            "program_understanding_prose": {
                "type": "object",
                "properties": {
                    key: {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": target_lengths[key],
                    }
                    for key in prose
                },
                "required": list(prose),
                "additionalProperties": False,
            }
        },
        "required": ["program_understanding_prose"],
        "additionalProperties": False,
    }

    try:
        metadata["model_calls"] = 1
        compressed = _same_model_chat(
            "Compress the supplied program-understanding prose for a C grader. Rewrite each field "
            "as a materially shorter summary; do not copy sentences verbatim. Preserve the distinction "
            "between actual source behavior and assignment intent, all important operations, conditions, "
            "negations, and behavior mismatches. Do not add facts, functions, line numbers, findings, "
            "scores, or assumptions. The JSON schema gives a strict maximum character count for each "
            "field. Return every requested field exactly once.",
            {
                "program_understanding_prose": prose,
                "target_total_chars": sum(target_lengths.values()),
            },
            prose_schema,
            response_token_budget=max(256, math.ceil(sum(target_lengths.values()) / 3) + 128),
        )
        candidate = (
            compressed.get("program_understanding_prose")
            if isinstance(compressed, dict) else None
        )
        if not isinstance(candidate, dict):
            raise ValueError("the grading model omitted the program_understanding_prose object.")
        if set(candidate) != set(prose):
            raise ValueError("same-model compression changed the requested prose fields.")

        new_understanding = dict(original_understanding)
        for key, source_text in prose.items():
            value = candidate[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"the grading model returned an empty or invalid {key} summary.")
            value = value.strip()
            if len(value) > target_lengths[key]:
                raise ValueError(
                    f"the grading model exceeded the {key} length budget "
                    f"({len(value)} > {target_lengths[key]})."
                )
            new_understanding[key] = value

        # Replace only selected natural-language fields. Every other payload
        # field and every structured program-understanding field stays untouched.
        result = dict(copied)
        result["program_understanding"] = new_understanding
        new_text = json.dumps(result, ensure_ascii=False)
        saved_chars = len(serialized) - len(new_text)
        if saved_chars < MIN_EXPECTED_SAVED_CHARS:
            raise ValueError(
                f"compression saved only {saved_chars} chars; "
                f"at least {MIN_EXPECTED_SAVED_CHARS} are required."
            )

        metadata.update({
            "method": "same_model",
            "output_chars": len(new_text),
            "saved_chars": saved_chars,
            "reason": "model compression met per-field and total reduction requirements",
        })
        return finish(result)
    except Exception as exc:  # Compression must never break evaluation.
        logger.info(
            "same-model semantic compression rejected; deterministic compaction used: %s", exc
        )
        metadata["error"] = f"{type(exc).__name__}: {exc}"[:300]
        compacted, metadata = _deterministic_semantic_compaction(copied, metadata)
        return finish(compacted)


def _deterministic_semantic_compaction(
    payload: dict[str, Any], metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Loss-aware fallback: shorten prose and keep all metric/evidence anchors."""
    compacted = dict(payload)
    understanding = compacted.get("program_understanding")
    if isinstance(understanding, dict):
        shortened = dict(understanding)
        for key in (
            "actual_behavior",
            "program_purpose",
            "intended_behavior",
            "contract_alignment",
            "algorithm_overview",
        ):
            value = shortened.get(key)
            if isinstance(value, str):
                shortened[key] = value[:220]
        for key in ("data_structures", "execution_flow", "assumptions", "ambiguities"):
            value = shortened.get(key)
            if isinstance(value, list):
                shortened[key] = [str(item)[:140] for item in value[:5]]
        # Function names, roles, and exact line numbers are preserved.
        compacted["program_understanding"] = shortened

    for key in ("review_constraints", "instructions", "scoring_rules"):
        values = compacted.get(key)
        if isinstance(values, list):
            compacted[key] = [str(item)[:180] for item in values[:8]]

    new_text = json.dumps(compacted, ensure_ascii=False)
    metadata.update({
        "method": "deterministic_fallback",
        "output_chars": len(new_text),
        "saved_chars": max(0, len(json.dumps(payload, ensure_ascii=False)) - len(new_text)),
    })
    return compacted, metadata
