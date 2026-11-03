#!/usr/bin/env python3
"""Constrain LLM advisory scores to documented 5-point bands.

Run from the repository root after copying score_bands.py and
rubrics/default_metric_rubric.json into the repository.
"""
from pathlib import Path

path = Path("src/grading_agent/llm/reviewer.py")
s = path.read_text(encoding="utf-8")

def replace_once(old: str, new: str, label: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"Refusing LLM rubric patch ({label}): expected 1 match, got {count}.")
    s = s.replace(old, new, 1)

replace_once(
'''from typing import Any

import requests
''',
'''from typing import Any
from pathlib import Path

import requests
''', "path import")
replace_once(
'''from grading_agent.config import settings
from grading_agent.llm.evidence import build_evidence_ledger, validate_review_payload
''',
'''from grading_agent.config import settings
from grading_agent.scoring.score_bands import BAND_LABELS, score_from_band, rubric_prompt_block
from grading_agent.llm.evidence import build_evidence_ledger, validate_review_payload
''', "score band imports")
replace_once(
'''                "score": {"type": "number"},
                "rationale": {"type": "string", "maxLength": 120},
''',
'''                "score_band": {"type": "string", "enum": list(BAND_LABELS)},
                "rationale": {"type": "string", "maxLength": 120},
''', "score response schema")
replace_once(
'''            "required": ["score", "rationale", "zero_score_reason", "confidence"],
''',
'''            "required": ["score_band", "rationale", "zero_score_reason", "confidence"],
''', "required score band")
replace_once(
'''        score = item.get("score", item.get("value"))
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        score = float(score)
        if not 0.0 <= score <= 100.0:
            continue
''',
'''        score_band = item.get("score_band")
        score = score_from_band(score_band)
        if score is None:
            # Reject arbitrary free-form numeric scores. Each band maps to one
            # fixed midpoint, so the model cannot make up a score like 83 or 91.
            continue
''', "score band normalization")
replace_once(
'''            "score": round(score, 2),
            "rationale": rationale,
''',
'''            "score": round(score, 2),
            "score_band": score_band,
            "rationale": rationale,
''', "retain score band")
replace_once(
'''    problem_contract = _grading_context_value(rules.get("problem_contract", {}))
    contract_is_specific = _assignment_contract_is_specific(task_statement, rules.get("problem_contract", {}))
    payload = {
''',
'''    problem_contract = _grading_context_value(rules.get("problem_contract", {}))
    contract_is_specific = _assignment_contract_is_specific(task_statement, rules.get("problem_contract", {}))
    metric_rubrics = rules.get("metric_rubrics", {})
    if not isinstance(metric_rubrics, dict):
        metric_rubrics = {}
    try:
        default_rubric_path = settings.PROJECT_ROOT / "rubrics" / "default_metric_rubric.json"
        default_rubric = json.loads(default_rubric_path.read_text(encoding="utf-8"))
        default_metric_rubrics = default_rubric.get("metrics", {})
        if isinstance(default_metric_rubrics, dict):
            metric_rubrics = {**default_metric_rubrics, **metric_rubrics}
    except (OSError, json.JSONDecodeError):
        pass
    payload = {
''', "load metric-specific rubric")
replace_once(
'''        "rubric_and_analyzer_summaries": [_compact_metric_context(metric, report) for metric in enabled],
        "evidence_ledger": _compact_ledger(ledger),
        "enabled_metric_names": enabled_names,
    }
''',
'''        "rubric_and_analyzer_summaries": [_compact_metric_context(metric, report) for metric in enabled],
        "evidence_ledger": _compact_ledger(ledger),
        "enabled_metric_names": enabled_names,
        "score_band_rubric": rubric_prompt_block(metric_rubrics),
    }
''', "include score rubric in payload")
replace_once(
'''        "You are the final expert evaluator for this C assignment. Read the submitted numbered "
        "source itself and score each enabled metric independently from 0 to 100 using the "
        "assignment, instructor rubric, validated findings, and analyzer evidence. The LLM is "
        "the scoring authority; analyzers are supporting evidence, not scores to copy. Do not "
        "invent requirements. Missing test data or runtime measurements are not failed tests. "
        "Assess logical correctness and runtime complexity from source when measurements are "
        "unavailable. Trace pointer ownership, allocations, frees, and later dereferences for "
        "memory safety. Distinguish confirmed defects from potential concerns. A zero requires "
        "a specific, fully demonstrated failure; provide zero_score_reason, otherwise use an "
        "appropriate nonzero score. For every metric return a concise, source-specific rationale "
        "of at most 18 words and 120 characters. Include zero_score_reason as an empty string "
        "when not applicable and confidence as a number from 0 to 1. Return exactly one JSON "
        "object with scores containing every enabled metric exactly once. No Markdown."
''',
'''        "You are the advisory rubric reviewer for this C assignment. Read the submitted numbered "
        "source and review every enabled metric using the assignment, instructor rubric, validated "
        "findings and deterministic analyzer evidence. Choose exactly one score_band from the "
        "score_band_rubric for every metric; never invent a numeric score or select an unlisted band. "
        "A band maps to a fixed midpoint. Your scores are ADVISORY ONLY: deterministic analyzers "
        "alone determine the final numerical grade. Explain disagreements with analyzer evidence; "
        "do not override measurements. Do not invent requirements. Missing test data or runtime "
        "measurements are unmeasured, not failed tests. Assess asymptotic complexity only where "
        "source and constraints support it; cyclomatic complexity is not Big-O. Trace pointer "
        "ownership, allocations, frees and dereferences for memory safety. Distinguish confirmed "
        "defects from potential concerns. For every metric return a concise source-specific rationale "
        "(at most 18 words and 120 characters), zero_score_reason, and confidence from 0 to 1. "
        "Return one JSON object with every enabled metric exactly once. No Markdown."
''', "advisory band instructions")
replace_once(
'''            "name": name,
            "score": score_item["score"],
            "rationale": score_item["rationale"],
''',
'''            "name": name,
            "score": score_item["score"],
            "score_band": score_item.get("score_band"),
            "rationale": score_item["rationale"],
''', "expose selected band")
s = s.replace(
    "The LLM is the rubric-scoring authority. Python validates range/schema and evidence references, but never silently edits a valid model score.",
    "The LLM band is advisory only; deterministic analyzer scores remain authoritative. Python validates the band and evidence references."
)
path.write_text(s, encoding="utf-8")
print("Applied 5-point band constraints to LLM advisory metric review.")
