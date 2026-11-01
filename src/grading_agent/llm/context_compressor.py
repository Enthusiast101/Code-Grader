"""Loss-aware compression using the configured grading model itself.

Only natural-language summaries are compressible. Submitted source, source line
numbers, finding identities, analyzer measurements, and evidence IDs remain
protected; deterministic compaction is used if same-model compression fails.
"""
from __future__ import annotations

import json
import logging
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
        "latency_seconds": float(metadata.get("latency_seconds", 0.0)),
        "error": metadata.get("error"),
    })


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
) -> dict[str, Any] | None:
    """Run one schema-constrained compression request through the grading model.

    Compression never selects a second model or provider. JSON-schema response_format
    constrains its output shape; the caller still validates all
    protected anchors before accepting the shortened text.
    """
    if not settings.runtime_value("LLM_CONTEXT_COMPRESSION_ENABLED"):
        return None

    # Compression is a structured transformation, not an open-ended analysis
    # turn. Use the same grading model but direct constrained-output mode so
    # JSON Schema decoding is active immediately; Qwen3 lazy grammar + thinking can
    # emit unquoted-key pseudo-objects and cause the compressor to be rejected.
    thinking = False
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
    body: dict[str, Any] = {
        "model": str(settings.runtime_value("MODEL")),
        "stream": False,
        "temperature": 0.0,
        "seed": settings.SEED,
        "max_tokens": int(settings.runtime_value("LLM_CONTEXT_COMPRESSION_MAX_TOKENS")),
        "chat_template_kwargs": {"enable_thinking": thinking},
        "messages": messages,
    }
    # Use the documented /v1/chat/completions JSON Schema contract. The raw
    # `grammar` field is not the response-format mechanism for this endpoint.
    # Thinking is disabled for this structured transform so schema decoding is
    # active from the first assistant token on Qwen3/llama.cpp.
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", "context_compression")[:48]
    # llama.cpp expects the schema directly under response_format.schema.
    # Do not send the OpenAI nested `json_schema` wrapper here.
    body["response_format"] = {
        "type": "json_schema",
        "schema": response_schema,
    }
    body["reasoning_effort"] = "none"
    body["reasoning_format"] = "none"

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
        raise ValueError("Same-model context compression returned an unexpected chat response.") from exc
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text", "")) for part in content
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


def _deterministic_review_compaction(review: dict[str, Any]) -> dict[str, Any]:
    """Compact a previous RSI response without dropping findings or evidence."""
    findings = review.get("findings", [])
    compact_findings: list[dict[str, Any]] = []
    if isinstance(findings, list):
        for item in findings:
            if not isinstance(item, dict):
                continue
            compact_findings.append({
                "metric": str(item.get("metric", "unmapped"))[:80],
                "severity": str(item.get("severity", "medium"))[:20],
                "title": str(item.get("title", "Potential issue"))[:140],
                "description": str(item.get("description", item.get("text", "")))[:360],
                "line_start": item.get("line_start"),
                "line_end": item.get("line_end", item.get("line_start")),
                "evidence_ids": list(dict.fromkeys(str(ref) for ref in item.get("evidence_ids", []) if ref))[:8]
                if isinstance(item.get("evidence_ids", []), list) else [],
                "confidence": item.get("confidence", 0.5),
            })
    questions = review.get("unresolved_questions", [])
    return {
        "overall_assessment": str(review.get("overall_assessment", review.get("summary", "RSI review state retained.")))[:700],
        "findings": compact_findings,
        "unresolved_questions": [str(item)[:180] for item in questions[:5]] if isinstance(questions, list) else [],
    }


def compress_review_history(review: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compress the preceding RSI response with the configured grading model when it is large.

    All findings must survive compression with the same metric, source location,
    and evidence IDs. A summary that drops or changes those anchors is rejected.
    """
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
        "latency_seconds": 0.0,
    }
    if not metadata["enabled"]:
        metadata["reason"] = "same-model compression disabled"
        return original, metadata
    if len(original_text) <= threshold:
        metadata["reason"] = "history below compression threshold"
        return original, metadata

    try:
        description_schema = {
            "type": "object",
            "properties": {
                "overall_assessment": {"type": "string", "maxLength": 240},
                "finding_descriptions": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 220},
                    "maxItems": len(original["findings"]),
                },
            },
            "required": ["overall_assessment", "finding_descriptions"],
        }
        metadata["model_calls"] = 1
        compressed = _same_model_chat(
            "Compress the prose in this C-program review for the next RSI pass. "
            "Return only overall_assessment and finding_descriptions. The descriptions "
            "must remain in the same order and one-to-one with the input findings. Preserve "
            "all technical meaning, defect certainty, and negations. The caller reattaches "
            "the original finding objects, metrics, severity, titles, source locations, and "
            "evidence IDs, so do not reproduce or modify those fields.",
            {
                "overall_assessment": original["overall_assessment"],
                "finding_descriptions": [item["description"] for item in original["findings"]],
                "target_chars": max(900, threshold),
            },
            description_schema,
        )
        if not isinstance(compressed, dict) or not isinstance(compressed.get("finding_descriptions"), list):
            raise ValueError("the grading model returned no structured description array.")
        candidate_descriptions = compressed["finding_descriptions"]
        if len(candidate_descriptions) != len(original["findings"]):
            raise ValueError("same-model compression changed the number of descriptions.")
        validated: list[dict[str, Any]] = []
        for original_item, candidate_description in zip(original["findings"], candidate_descriptions):
            description = str(candidate_description).strip()
            if not description:
                raise ValueError("same-model compression returned an empty finding description.")
            validated.append({**original_item, "description": description[:360]})
        summary = str(compressed.get("overall_assessment", "")).strip() or original["overall_assessment"]
        result = {
            "overall_assessment": summary[:700],
            "findings": validated,
            "unresolved_questions": original["unresolved_questions"],
        }
        result_text = json.dumps(result, ensure_ascii=False)
        if len(result_text) >= len(original_text):
            raise ValueError("the grading model output did not reduce the history size.")
        metadata.update({
            "method": "same_model",
            "output_chars": len(result_text),
            "saved_chars": len(original_text) - len(result_text),
        })
        return result, metadata
    except Exception as exc:  # Compression must never break grading.
        compact = _deterministic_review_compaction(review)
        compact_text = json.dumps(compact, ensure_ascii=False)
        metadata.update({
            "method": "deterministic_fallback",
            "output_chars": len(compact_text),
            "saved_chars": max(0, len(original_text) - len(compact_text)),
            "error": f"{type(exc).__name__}: {exc}"[:300],
        })
        logger.info("same-model context compression unavailable; deterministic compaction used: %s", exc)
        return compact, metadata
    finally:
        metadata["latency_seconds"] = round(time.perf_counter() - started, 3)
        _record_compression_event("review_history", metadata)


def compress_semantic_sections(
    payload: dict[str, Any],
    *,
    target_chars: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compress semantic prose while leaving source, metrics and evidence immutable.

    The configured grading model may shorten only the natural-language fields in the prior
    program-understanding summary. Function names/roles/line ranges, rubric rows,
    analyzer measurements, source text, and evidence IDs are copied verbatim.
    """
    started = time.perf_counter()
    copied = dict(payload)
    serialized = json.dumps(copied, ensure_ascii=False)
    metadata: dict[str, Any] = {
        "method": "unchanged", "input_chars": len(serialized),
        "output_chars": len(serialized), "saved_chars": 0, "model_calls": 0,
        "enabled": bool(settings.runtime_value("LLM_CONTEXT_COMPRESSION_ENABLED")),
        "model": settings.runtime_value("MODEL"),
    }
    if len(serialized) <= target_chars:
        return copied, metadata

    original_understanding = copied.get("program_understanding")
    if not isinstance(original_understanding, dict):
        return _deterministic_semantic_compaction(copied, metadata)

    # Only prose fields enter the compression model. Structured function and
    # source-location data are retained locally and reattached after validation.
    prose_keys = (
        "actual_behavior", "program_purpose", "intended_behavior",
        "contract_alignment", "algorithm_overview",
    )
    prose = {
        key: str(original_understanding.get(key, ""))[:4_000]
        for key in prose_keys if original_understanding.get(key)
    }
    if not prose:
        return _deterministic_semantic_compaction(copied, metadata)
    if not metadata["enabled"]:
        return _deterministic_semantic_compaction(copied, metadata)

    try:
        prose_schema = {
            "type": "object",
            "properties": {
                "program_understanding_prose": {
                    "type": "object",
                    "properties": {key: {"type": "string", "maxLength": 600} for key in prose},
                    "required": list(prose),
                }
            },
            "required": ["program_understanding_prose"],
        }
        metadata["model_calls"] = 1
        compressed = _same_model_chat(
            "Compress this program-understanding prose for a C grader. Preserve actual source "
            "behavior separately from assignment intent, all important operations and negations, "
            "and any behavior mismatch. Do not add facts, functions, line numbers, findings, or scores. "
            "Use concise sentences and keep the provided fields unchanged.",
            {"program_understanding_prose": prose, "target_chars": max(400, target_chars // 3)},
            prose_schema,
        )
        candidate = compressed.get("program_understanding_prose") if isinstance(compressed, dict) else None
        if not isinstance(candidate, dict):
            raise ValueError("The grading model omitted the program_understanding_prose object.")

        new_understanding = dict(original_understanding)
        for key in prose_keys:
            if key not in prose:
                continue
            value = candidate.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"the grading model returned an empty or invalid {key} summary.")
            # Reject expansions: the model must genuinely compact the prose.
            if len(value) > len(prose[key]):
                raise ValueError(f"The grading model expanded {key} instead of compressing it.")
            new_understanding[key] = value.strip()[:1_200]

        # Keep all structured semantics untouched.
        copied["program_understanding"] = new_understanding
        new_text = json.dumps(copied, ensure_ascii=False)
        if len(new_text) >= len(serialized):
            raise ValueError("the grading model output did not reduce the complete prompt payload.")
        metadata.update({
            "method": "same_model",
            "output_chars": len(new_text),
            "saved_chars": len(serialized) - len(new_text),
        })
        return copied, metadata
    except Exception as exc:  # Compression must never break evaluation.
        logger.info("same-model semantic compression rejected; deterministic compaction used: %s", exc)
        metadata["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return _deterministic_semantic_compaction(copied, metadata)
    finally:
        metadata["latency_seconds"] = round(time.perf_counter() - started, 3) if "started" in locals() else 0.0
        _record_compression_event("program_understanding", metadata)


def _deterministic_semantic_compaction(
    payload: dict[str, Any], metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Loss-aware fallback: shorten prose and keep all metric/evidence anchors."""
    compacted = dict(payload)
    understanding = compacted.get("program_understanding")
    if isinstance(understanding, dict):
        shortened = dict(understanding)
        for key in (
            "actual_behavior", "program_purpose", "intended_behavior",
            "contract_alignment", "algorithm_overview",
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

