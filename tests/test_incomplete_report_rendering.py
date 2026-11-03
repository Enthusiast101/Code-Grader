from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grading_agent.ui.app import _metric_summary
from grading_agent.reporting.text_report import render_text_report


def incomplete_report():
    return {
        "submission": "submission.c",
        "problem": "Graph bridges assignment",
        "configuration": {"enabled_metrics": [
            {"name": "correctness", "weight": 0.5},
            {"name": "syntax", "weight": 0.5},
        ]},
        "correctness": {"status": "inconclusive", "score": None, "reason": "No tests configured."},
        "syntax": {"status": "completed", "score": 100.0, "score_rationale": "Compiled cleanly."},
        "scoring": {"status": "failed_incomplete", "score": None, "missing_metrics": [
            {"metric": "correctness", "analyzer": "io_tests", "status": "inconclusive", "reason": "No tests configured."}
        ], "warnings": ["Final grade withheld."]},
    }


def test_gradio_displays_na_and_missing_reason_not_zero_grade():
    rendered = _metric_summary(incomplete_report())
    assert "Overall grade: N/A" in rendered
    assert "correctness: N/A" in rendered
    assert "No tests configured" in rendered
    assert "0.00/100" not in rendered


def test_text_report_displays_na_and_root_cause_not_zero_grade():
    rendered = render_text_report(incomplete_report())
    assert "Final grade: N/A" in rendered
    assert "Missing metric: correctness" in rendered
    assert "No tests configured" in rendered
    assert "Final grade: 0.00/100" not in rendered


def test_true_zero_is_still_displayed_as_zero_for_a_completed_grade():
    report = incomplete_report()
    report["correctness"] = {"status": "completed", "score": 0.0}
    report["scoring"] = {"status": "completed", "score": 0.0, "contributions": []}
    assert "Overall grade: 0.00/100" in _metric_summary(report)
    assert "Final grade: 0.00/100" in render_text_report(report)
