"""Deterministic syntax-tree analysis and its use in test design and grading."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grading_agent.analyzers.c_ast import analyze_c_source, compact_ast_summary, strip_comments  # noqa: E402
from grading_agent.llm.evidence import build_evidence_ledger  # noqa: E402
from grading_agent.llm.reviewer import _finding_prevalidation_reason  # noqa: E402
from grading_agent.testing.test_generation import (  # noqa: E402
    analyze_submission_for_test_generation,
    generate_assignment_test_spec,
)

EXAMPLES = ROOT / "examples"


def source(relative: str) -> str:
    return (EXAMPLES / relative).read_text(encoding="utf-8")


def test_every_example_file_parses():
    files = sorted(EXAMPLES.rglob("*.c"))
    failures = [str(f) for f in files if analyze_c_source(f.read_text(encoding="utf-8"))["status"] != "completed"]
    assert files and not failures


def test_comments_are_removed_without_shifting_lines():
    text = 'int a; /* one\ntwo */ int b; // tail\nchar *s = "/* not a comment */";\n'
    stripped = strip_comments(text)
    assert stripped.count("\n") == text.count("\n")
    assert '"/* not a comment */"' in stripped and "tail" not in stripped


def test_memory_facts_are_exact_and_error_path_free_is_not_a_use_after_free():
    tree = analyze_c_source(source("assignments/01-memory-errors/submission.c"))
    assert tree["uses_after_free"] == [{"variable": "values", "freed_line": 23, "used_line": 24}]
    assert [(a["variable"], a["line"]) for a in tree["allocations_never_freed"]] == [("message", 15)]


def test_loop_depth_follows_called_functions():
    rescanning = analyze_c_source(source("leetcode/0239-sliding-window-maximum/submission.c"))
    assert rescanning["max_loop_depth"] == 1 and rescanning["effective_loop_depth"] == 2
    assert analyze_c_source(source("assignments/04-bruteforce-triplets/submission.c"))["effective_loop_depth"] == 3


def test_edge_case_targets_point_at_the_planted_defects():
    overflow = analyze_c_source(source("leetcode/0042-trapping-rain-water/submission.c"))["edge_case_targets"]
    assert overflow[0]["kind"] == "integer_overflow" and overflow[0]["line"] == 19
    merge = analyze_c_source(source("leetcode/0056-merge-intervals/submission.c"))["edge_case_targets"]
    assert any(t["line"] == 28 and "items[i].start < last->end" in t["detail"] for t in merge[:3])
    leap = analyze_c_source(source("assignments/02-logical-leap-year/submission.c"))["edge_case_targets"]
    assert leap[0]["kind"] == "modulus_condition"


def test_evidence_ledger_includes_syntax_tree_facts_and_risks():
    report = {"syntax_tree": analyze_c_source(source("assignments/01-memory-errors/submission.c")),
              "configuration": {"enabled_metrics": []}}
    ledger = build_evidence_ledger(report, {"metrics": []})
    kinds = {item["kind"] for item in ledger}
    assert {"syntax_tree", "syntax_tree_risk"} <= kinds
    risks = [item for item in ledger if item["kind"] == "syntax_tree_risk"]
    assert any(item["facts"]["category"] == "use_after_free" and item["facts"]["line"] == 24 for item in risks)


def test_finding_about_a_call_the_program_never_makes_is_rejected():
    code = source("assignments/01-memory-errors/submission.c")
    report = {"syntax_tree": analyze_c_source(code)}
    finding = {"metric": "memory_safety", "title": "Buffer overflow in memcpy", "description": "memcpy copies too much.",
               "line_start": 20, "line_end": 20, "evidence_ids": []}
    reason = _finding_prevalidation_reason(finding, code, {"metrics": []}, report)
    assert reason and "never calls it" in reason


def test_analysis_stage_gets_the_tree_and_drops_ungrounded_weak_points():
    tree = compact_ast_summary(analyze_c_source(source("leetcode/0200-number-of-islands/submission.c")))
    seen = {}

    def request(messages, schema, stage):
        seen["payload"] = json.loads(messages[1]["content"])
        return {
            "implementation_summary": "Counts islands with an explicit stack.",
            "input_parsing_and_validation": ["scanf result checked"],
            "control_flow_and_data_types": ["iterative flood fill"],
            "likely_weak_points": ["Grid rows are never freed.", "strcpy may overflow the row buffer."],
            "recommended_test_dimensions": ["single cell", "full grid"],
        }

    analysis = analyze_submission_for_test_generation(
        problem="Count islands.", problem_contract={}, submission_source=source("leetcode/0200-number-of-islands/submission.c"),
        request_json=request, syntax_tree=tree,
    )
    assert seen["payload"]["syntax_tree"]["functions"][0]["name"] == "sink_island"
    assert analysis["likely_weak_points"] == ["Grid rows are never freed."]
    assert "strcpy may overflow the row buffer." in analysis["dropped_contradictory_weak_points"]


def test_generator_receives_line_anchored_targets():
    tree = compact_ast_summary(analyze_c_source(source("leetcode/0056-merge-intervals/submission.c")))
    seen = {}

    def request(messages, schema, stage):
        payload = json.loads(messages[1]["content"])
        if stage.startswith("assignment_test_generation"):
            seen["targets"] = payload.get("syntax_tree_edge_case_targets")
            return {"spec_summary": "Merge intervals.", "ambiguities": [],
                    "inferred_contract": {"input_format": "n then n pairs of integers", "output_format": "merged count then pairs",
                                          "constraints": [], "semantics": ["merge overlapping closed intervals"], "formatting": []},
                    "test_inputs": [{"id": "touch", "input": "2\n1 4\n4 5\n", "expected_output": "1\n1 5\n"}]}
        return {"verifications": [{"id": item["id"], "input": item["input"], "input_is_valid": True,
                                   "expected_output": "1\n1 5\n", "reason": "touching intervals merge"}
                                  for item in payload["test_inputs_to_solve"]]}

    generate_assignment_test_spec(problem="Read n closed intervals, merge every overlapping group, and print the count then each merged interval.",
                                  problem_contract={},
                                  request_json=request, max_cases=1, syntax_tree=tree)
    assert any(target["line"] == 28 for target in seen["targets"])


def test_grid_targets_name_the_ignored_input_and_stay_hand_writable():
    # Regression: targets asked for a full 30 x 30 grid (truncated generation,
    # miscounted cells) and for loop counters / scanf return values that no
    # input can set, while the real defect (blocked cells unused) was absent.
    tree = analyze_c_source(source("assignments/09-dp-grid-paths/submission.c"))
    targets = tree["edge_case_targets"]
    assert targets[0]["kind"] == "ignored_input" and targets[0]["line"] == 12
    assert [item["variable"] for item in tree["ignored_inputs"]] == ["blocked"]
    details = [t["detail"] for t in targets]
    assert not any(d.startswith(("i ", "j ")) for d in details), "loop counters are not inputs"
    assert not any("scanf(" in d and t["kind"] != "ignored_input" for d, t in zip(details, targets))
    assert not any("maximum size allowed" in t["suggestion"] for t in targets)
    assert analyze_c_source(source("assignments/09-dp-grid-paths/reference_solution.c"))["ignored_inputs"] == []
