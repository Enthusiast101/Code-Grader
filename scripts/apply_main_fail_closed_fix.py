#!/usr/bin/env python3
"""Keep deterministic grades valid when the advisory LLM stage is unavailable."""
from pathlib import Path
path = Path("src/grading_agent/main.py")
s = path.read_text(encoding="utf-8")
old = '''    if isinstance(llm_result, dict) and llm_result.get("status") != "completed":
        report["scoring"]["status"] = "completed_with_warnings"
        report["scoring"].setdefault("warnings", []).append(
'''
new = '''    if isinstance(llm_result, dict) and llm_result.get("status") != "completed":
        # LLM review is explanatory/advisory only. Preserve `completed` when
        # every deterministic analyzer produced a valid score, and preserve
        # `failed_incomplete` when any required analyzer was unmeasured.
        report["scoring"].setdefault("warnings", []).append(
'''
if s.count(old) != 1:
    raise SystemExit(f"Refusing main.py patch: expected one scoring-status block, found {s.count(old)}.")
s = s.replace(old, new, 1)
warning_old = '''            "One or more LLM stages used a validated deterministic fallback. Every enabled metric "
            "has a finite score; fallback-derived scores are explicitly labelled and are not "
            "presented as successful LLM scores."
'''
warning_new = '''            "The advisory LLM review was unavailable or incomplete. It cannot change analyzer scores; "
            "the overall grade is issued only when every enabled deterministic analyzer completed."
'''
if s.count(warning_old) != 1:
    raise SystemExit(f"Refusing warning update: expected one LLM fallback warning, found {s.count(warning_old)}.")
s = s.replace(warning_old, warning_new, 1)
path.write_text(s, encoding="utf-8")
print("Applied status policy: deterministic grades survive LLM outage; incomplete metrics still fail closed.")
