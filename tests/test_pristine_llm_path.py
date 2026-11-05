"""Focused regression tests for schema-safe generation and same-model compression."""
from __future__ import annotations

import json

import pytest

from grading_agent.config import settings
from grading_agent.llm import context_compressor
from grading_agent.llm.gbnf import json_schema_to_gbnf
from grading_agent.llm import reviewer
from grading_agent.reporting.rationale import compact_rationale
from grading_agent.reporting.text_report import render_text_report
from grading_agent.scoring.overall import apply_metric_scores
from grading_agent.ui.app import _metric_summary


class _FakeResponse:
    ok = True
    status_code = 200
    text = ""
    reason = "OK"

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_json_schema_compiles_to_valid_looking_gbnf():
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "scores": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 1,
                "maxItems": 3,
            },
            "line": {"type": ["integer", "null"]},
        },
        "required": ["summary", "scores", "line"],
    }
    grammar = json_schema_to_gbnf(schema)
    assert "root ::= object-" in grammar
    assert '"summary" ws ":" ws json-string' in grammar
    assert '"scores" ws ":" ws array-' in grammar
    assert '"line" ws ":" ws union-' in grammar
    assert "{0,2}" in grammar
    assert "json-char" in grammar


def test_grading_request_uses_chat_completions_json_schema_once(monkeypatch):
    sent = []

    def fake_post(url, *, json, timeout):
        sent.append((url, json, timeout))
        return _FakeResponse({"choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}]})

    monkeypatch.setattr(reviewer.requests, "post", fake_post)
    monkeypatch.setattr(reviewer, "_fit_grading_messages", lambda messages, requested: (messages, requested, {}))
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    parsed, _raw = reviewer._request_json(
        [{"role": "system", "content": "test"}, {"role": "user", "content": "return ok"}],
        max_tokens=256,
        enable_thinking=True,
        response_schema=schema,
    )
    assert parsed == {"ok": True}
    assert len(sent) == 1
    body = sent[0][1]
    assert body["model"] == settings.runtime_value("MODEL")
    assert "grammar" not in body
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert "schema" not in body["response_format"]  # top-level schema is ignored by llama-server
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    assert body["reasoning_effort"] == "none"


def test_schema_aware_json_extractor_skips_leading_nonmatching_object():
    content = (
        '{"stage":"verifier","status":"working"}\n'
        '{"verifications":[{"id":"case-1","input":"","input_is_valid":true,'
        '"expected_output":"READY\\n","reason":"No stdin required."}]}'
    )
    parsed = reviewer._extract_json_object(
        content,
        required_keys={"verifications"},
    )
    assert parsed is not None
    assert "stage" not in parsed
    assert parsed["verifications"][0]["id"] == "case-1"


def test_truncated_json_does_not_return_nested_fragment():
    truncated = (
        '{"spec_summary":"x","ambiguities":[],"inferred_contract":{"input_format":"two ints",'
        '"output_format":"one int","constraints":[],"semantics":["sum"],"formatting":[]},'
        '"test_inputs":[{"id":"a","input":"1 2","expected_output":"3"},{"id":"b","inp'
    )
    assert reviewer._extract_json_object(
        truncated,
        required_keys={"ambiguities", "inferred_contract", "test_inputs"},
    ) is None
    assert reviewer._extract_json_object(truncated) is None


def test_structured_request_selects_json_object_matching_response_schema(monkeypatch):
    raw = '{"stage":"verifier","status":"working"}\n{"ok":true}'

    def fake_post(url, *, json, timeout):
        return _FakeResponse({
            "choices": [{
                "message": {"content": raw},
                "finish_reason": "stop",
            }]
        })

    monkeypatch.setattr(reviewer.requests, "post", fake_post)
    monkeypatch.setattr(
        reviewer,
        "_fit_grading_messages",
        lambda messages, requested: (messages, requested, {}),
    )
    parsed, _ = reviewer._request_json(
        [{"role": "system", "content": "Return JSON."}],
        max_tokens=256,
        enable_thinking=False,
        response_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
    )
    assert parsed == {"ok": True}


def test_compression_uses_same_model_native_schema_and_one_request(monkeypatch):
    sent = []

    def fake_post(url, *, json, timeout):
        sent.append(json)
        return _FakeResponse({"choices": [{"message": {"content": '{"overall_assessment":"short","finding_descriptions":[]}'}}]})

    monkeypatch.setattr(context_compressor.requests, "post", fake_post)
    monkeypatch.setattr(settings, "LLAMA_BASE_URL", "http://127.0.0.1:8080")
    with settings.runtime_overrides({
        "LLM_CONTEXT_COMPRESSION_ENABLED": True,
        "LLAMA_JSON_GRAMMAR_ENABLED": True,
        "LLAMA_REASONING_ENABLED": True,
        "LLM_CONTEXT_COMPRESSION_MAX_TOKENS": 512,
    }):
        parsed = context_compressor._same_model_chat(
            "compress", {"finding_descriptions": []},
            {"type": "object", "properties": {"overall_assessment": {"type": "string"}, "finding_descriptions": {"type": "array", "items": {"type": "string"}}}, "required": ["overall_assessment", "finding_descriptions"]},
        )
    assert parsed["overall_assessment"] == "short"
    assert len(sent) == 1
    assert sent[0]["model"] == settings.runtime_value("MODEL")
    assert "grammar" not in sent[0]
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert sent[0]["response_format"]["json_schema"]["schema"] == {
        "type": "object", "properties": {"overall_assessment": {"type": "string"},
        "finding_descriptions": {"type": "array", "items": {"type": "string"}}},
        "required": ["overall_assessment", "finding_descriptions"],
    }
    assert "schema" not in sent[0]["response_format"]  # top-level schema is ignored by llama-server
    assert sent[0]["chat_template_kwargs"]["enable_thinking"] is False
    assert sent[0]["reasoning_effort"] == "none"



def test_unquoted_key_pseudo_json_is_rejected():
    malformed = '{ overall_assessment: "review", findings: [], unresolved_questions: [] }'
    assert reviewer._extract_json_object(malformed) is None
    assert context_compressor._parse_json_object(malformed) is None


def test_schema_constrained_request_disables_qwen_lazy_reasoning(monkeypatch):
    sent = []
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}

    def fake_post(url, *, json, timeout):
        sent.append(json)
        return _FakeResponse({"choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}]})

    monkeypatch.setattr(reviewer.requests, "post", fake_post)
    monkeypatch.setattr(reviewer, "_fit_grading_messages", lambda messages, requested: (messages, requested, {}))
    with settings.runtime_overrides({"LLAMA_REASONING_ENABLED": True, "LLAMA_JSON_GRAMMAR_ENABLED": True}):
        parsed, _ = reviewer._request_json(
            [{"role": "system", "content": "Return JSON"}, {"role": "user", "content": "ok"}],
            max_tokens=128,
            enable_thinking=True,
            response_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        )
    assert parsed == {"ok": True}
    assert len(sent) == 1
    assert sent[0]["chat_template_kwargs"]["enable_thinking"] is False
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert sent[0]["response_format"]["json_schema"]["schema"] == schema
    assert "schema" not in sent[0]["response_format"]  # top-level schema is ignored by llama-server
    assert "grammar" not in sent[0]

def test_same_model_compression_runs_and_records_actual_model_call(monkeypatch):
    monkeypatch.setattr(
        context_compressor,
        "_same_model_chat",
        lambda *_args, **_kwargs: {
            "overall_assessment": "Short assessment.",
            "finding_descriptions": ["A concise finding description."],
        },
    )
    long_review = {
        "overall_assessment": "A very long overall assessment that should be compressed " * 8,
        "findings": [{
            "metric": "memory_safety", "severity": "high", "title": "Use after free",
            "description": "A very long technical description that must be compacted while keeping the exact evidence and source location. " * 5,
            "line_start": 12, "line_end": 12, "evidence_ids": ["E1"], "confidence": 0.9,
        }],
        "unresolved_questions": [],
    }
    events = []
    token = context_compressor.COMPRESSION_EVENTS.set(events)
    try:
        with settings.runtime_overrides({"LLM_CONTEXT_COMPRESSION_ENABLED": True, "LLM_CONTEXT_COMPRESSION_THRESHOLD_CHARS": 10}):
            compacted, metadata = context_compressor.compress_review_history(long_review)
    finally:
        context_compressor.COMPRESSION_EVENTS.reset(token)
    assert metadata["method"] == "same_model"
    assert metadata["model_calls"] == 1
    assert compacted["findings"][0]["evidence_ids"] == ["E1"]
    assert compacted["findings"][0]["line_start"] == 12
    assert events and events[0]["method"] == "same_model"
    assert events[0]["model_calls"] == 1


def test_valid_llm_zero_score_remains_authoritative():
    metrics = [{"name": "correctness", "enabled": True, "weight": 1.0}]
    report = {"correctness": {"score": None, "status": "inconclusive"}}
    llm = {"metrics": [{"name": "correctness", "score": 0.0, "rationale": "The submission violates the explicit required input and output contract.", "origin": "llm"}]}
    apply_metric_scores(metrics, report, llm)
    assert report["correctness"]["score"] == 0.0
    assert report["correctness"]["score_origin"] == "llm"
    assert report["correctness"]["score_rationale"]


def test_report_explanations_are_short_and_score_basis_is_removed():
    rationale = "The implementation uses dynamic allocation without releasing the allocated message buffer, creating a confirmed memory leak that reduces safety."
    compact = compact_rationale(rationale)
    assert len(compact) <= 120
    assert len(compact.split()) <= 24
    report = {
        "problem": "grading",
        "submission": "sample.c",
        "configuration": {"enabled_metrics": [{"name": "memory_safety", "weight": 1.0}]},
        "memory_safety": {"score": 85, "status": "completed", "score_rationale": rationale, "scoring_basis": "verbose"},
        "scoring": {"score": 85, "status": "completed", "warnings": []},
        "llm_rsi_evaluation": {"status": "completed", "context_compression": {"enabled": True, "status": "same_model_used", "same_model_calls": 1, "model": "Qwen"}},
        "llm_rsi_findings": [],
    }
    text = render_text_report(report)
    ui = _metric_summary(report)
    assert "Assessment:" in text and "Assessment:" in ui
    assert "Score basis" not in text and "Score basis" not in ui
    assert "same grading model used" in text
    assert len(compact_rationale(report["memory_safety"]["score_rationale"])) <= 120


def test_clean_compiler_success_is_not_reported_as_a_finding():
    entry = {
        "id": "E1", "kind": "compiler_diagnostics",
        "statement": "Compiler: compiled=True; warnings=0; errors=0.",
        "facts": {"compiled": True, "warning_count": 0, "error_count": 0},
    }
    assert reviewer._is_clean_compiler_result(entry)
    assert reviewer._is_positive_compiler_statement(entry["statement"])
    review = reviewer._normalise_global_review(
        {"overall_assessment": "Compiler succeeded.", "findings": [{
            "metric": "syntax", "severity": "high", "title": "Compiler success",
            "description": entry["statement"], "line_start": None,
            "line_end": None, "evidence_ids": ["E1"], "confidence": 1.0,
        }]},
        [{"name": "syntax"}], "int main(void) { return 0; }", [entry], {1},
    )
    assert review is not None
    assert review["findings"] == []


def test_structured_json_grammar_cannot_be_disabled_by_legacy_toggle(monkeypatch):
    sent = []

    def fake_post(url, *, json, timeout):
        sent.append(json)
        return _FakeResponse({"choices": [{"message": {"content": '{"ok":true}'}}]})

    monkeypatch.setattr(reviewer.requests, "post", fake_post)
    monkeypatch.setattr(reviewer, "_fit_grading_messages", lambda messages, requested: (messages, requested, {}))
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    with settings.runtime_overrides({"LLAMA_JSON_GRAMMAR_ENABLED": False}):
        parsed, _ = reviewer._request_json(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "return JSON"}],
            max_tokens=128, enable_thinking=False, response_schema=schema,
        )
    assert parsed == {"ok": True}
    assert len(sent) == 1
    assert "grammar" not in sent[0]
    assert sent[0]["response_format"]["type"] == "json_schema"


def test_reviewer_and_compressor_share_the_same_compression_event_channel():
    assert reviewer._CONTEXT_COMPRESSION_EVENTS is context_compressor.COMPRESSION_EVENTS


def test_semantic_compression_uses_configured_model_and_preserves_structure(monkeypatch):
    calls = []

    def fake_chat(system_prompt, payload, schema):
        calls.append((system_prompt, payload, schema))
        prose = payload["program_understanding_prose"]
        return {"program_understanding_prose": {key: value[:30] for key, value in prose.items()}}

    monkeypatch.setattr(context_compressor, "_same_model_chat", fake_chat)
    payload = {
        "program_understanding": {
            "actual_behavior": "The function allocates a buffer, writes a message, prints it, and then releases it. " * 8,
            "intended_behavior": "Print the required message and exit normally. " * 8,
            "functions": [{"name": "main", "line_start": 1, "line_end": 24}],
            "data_structures": ["heap buffer"],
        },
        "evidence_ledger": [{"id": "E1", "statement": "Compiler completed."}],
    }
    with settings.runtime_overrides({"LLM_CONTEXT_COMPRESSION_ENABLED": True}):
        compacted, metadata = context_compressor.compress_semantic_sections(payload, target_chars=300)
    assert len(calls) == 1
    assert metadata["method"] == "same_model"
    assert compacted["program_understanding"]["functions"] == payload["program_understanding"]["functions"]
    assert compacted["evidence_ledger"] == payload["evidence_ledger"]
    assert len(compacted["program_understanding"]["actual_behavior"]) < len(payload["program_understanding"]["actual_behavior"])


def test_failed_compression_uses_anchor_preserving_fallback_without_retry(monkeypatch):
    calls = []

    def fail_once(*_args, **_kwargs):
        calls.append(1)
        raise ValueError("invalid compressor response")

    monkeypatch.setattr(context_compressor, "_same_model_chat", fail_once)
    review = {
        "overall_assessment": "A long assessment. " * 80,
        "findings": [{
            "metric": "memory_safety", "severity": "high", "title": "Use after free",
            "description": "A long description. " * 40, "line_start": 17, "line_end": 17,
            "evidence_ids": ["E17"], "confidence": 0.9,
        }],
        "unresolved_questions": [],
    }
    with settings.runtime_overrides({"LLM_CONTEXT_COMPRESSION_ENABLED": True, "LLM_CONTEXT_COMPRESSION_THRESHOLD_CHARS": 50}):
        compacted, metadata = context_compressor.compress_review_history(review)
    assert len(calls) == 1
    assert metadata["method"] == "deterministic_fallback"
    assert compacted["findings"][0]["line_start"] == 17
    assert compacted["findings"][0]["evidence_ids"] == ["E17"]
