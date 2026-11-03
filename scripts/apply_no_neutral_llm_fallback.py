#!/usr/bin/env python3
"""Remove neutral numeric scores from LLM-only fallback reporting."""
from pathlib import Path
path = Path("src/grading_agent/llm/reviewer.py")
s = path.read_text(encoding="utf-8")
old = '''    return {
        "name": metric_name, "score": 70.0,
        "rationale": evidence_note + " A neutral provisional score is used; the findings remain visible for human review.",
        "basis": "limited_evidence",
        "evidence_ids": list(dict.fromkeys(
            ref for item in relevant for ref in item.get("evidence_ids", [])
        ))[:6],
        "origin": "insufficient_evidence_fallback",
    }
'''
new = '''    return {
        "name": metric_name, "score": None, "score_band": None,
        "rationale": evidence_note + " No advisory score is assigned because the metric was not measured.",
        "basis": "insufficient_evidence",
        "evidence_ids": list(dict.fromkeys(
            ref for item in relevant for ref in item.get("evidence_ids", [])
        ))[:6],
        "origin": "insufficient_evidence_fallback",
    }
'''
if s.count(old) != 1:
    raise SystemExit(f"Refusing neutral-fallback patch: expected exactly one target, got {s.count(old)}.")
path.write_text(s.replace(old, new, 1), encoding="utf-8")
print("Applied no-neutral LLM fallback: unmeasured metrics remain N/A.")
