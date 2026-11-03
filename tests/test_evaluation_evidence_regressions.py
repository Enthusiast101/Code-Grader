from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

from grading_agent.analyzers.memory_safety import run_memory_safety
from grading_agent.analyzers.compiler_diagnostics import run_compiler_diagnostics
from grading_agent.analyzers.runtime_profile import run_runtime_profile
from grading_agent.llm import reviewer
from grading_agent.reporting.text_report import render_text_report
from grading_agent.scoring.overall import apply_metric_scores, calculate_overall_score


FIXTURE = Path(__file__).parent / "fixtures" / "computation_average_report_baseline.json"


def test_baseline_report_fixture_preserves_the_historical_77_point_calculation():
    baseline = json.loads(FIXTURE.read_text(encoding="utf-8"))
    computed = sum(item["weight"] * item["score"] for item in baseline["metrics"])
    assert computed == baseline["reported_final_grade"] == 77.0
    assert baseline["metrics"][6]["runtime_testset_configured"] is False


def test_r01_and_r02_executed_results_are_authoritative_and_logical_rationale_is_consistent():
    metrics = [
        {"name": "correctness", "analyzer": "io_tests", "weight": 0.5, "enabled": True},
        {"name": "logical", "analyzer": "edge_case_runner", "weight": 0.5, "enabled": True},
    ]
    report = {
        "correctness": {"status": "completed", "tests_total": 2, "tests_passed": 0, "results": [{"passed": False}, {"passed": False}]},
        "logical": {"status": "completed", "tests_total": 2, "tests_passed": 2, "results": [{"passed": True}, {"passed": True}], "rationale": "logical could not be measured from available evidence."},
    }
    llm = {"metrics": [
        {"name": "correctness", "score": 10, "origin": "llm", "rationale": "Looks correct."},
        {"name": "logical", "score": 0, "origin": "llm", "rationale": "No edge tests."},
    ]}

    apply_metric_scores(metrics, report, llm)

    assert report["correctness"]["score"] == 0
    assert report["logical"]["score"] == 100
    assert report["logical"]["score_origin"] == "deterministic_test_results"
    assert "could not be measured" not in report["logical"]["score_rationale"].lower()
    assert "passed 2/2" in report["logical"]["score_rationale"]


def test_r03_runtime_with_missing_inputs_is_not_measured_or_llm_scored():
    runtime = run_runtime_profile(Mock(), "int main(void) { return 0; }", [])
    metric = {"name": "runtime_efficiency", "analyzer": "runtime_profile", "weight": 1.0, "enabled": True}
    report = {"runtime_efficiency": runtime}
    apply_metric_scores(metric_list := [metric], report, {"metrics": [
        {"name": "runtime_efficiency", "score": 10, "origin": "llm", "rationale": "O(1) source code."}
    ]})
    grade = calculate_overall_score(metric_list, report)
    assert runtime["status"] == "inconclusive"
    assert runtime["score"] is None
    assert report["runtime_efficiency"]["score"] is None
    assert report["runtime_efficiency"]["unvalidated_llm_proposal"] == 10
    assert grade["status"] == "failed_incomplete"


def test_r04_report_does_not_claim_dynamic_memory_verification_without_runtime_tools():
    rendered = render_text_report({
        "problem": "memory check", "configuration": {"enabled_metrics": [{"name": "memory_safety", "weight": 1.0}]},
        "memory_safety": {"status": "inconclusive", "score": None, "runtime_tools": {
            "status": "not_run", "valgrind_status": "not_run", "asan_ubsan_status": "not_run",
        }},
        "scoring": {"status": "failed_incomplete", "score": None},
    })
    assert "Dynamic verification: not completed" in rendered
    assert "Valgrind status/score: not_run" in rendered
    assert "ASan/UBSan status/score: not_run" in rendered


def test_r05_asan_buffer_overflow_caps_memory_safety_score():
    clean_valgrind = """==1== HEAP SUMMARY:\n==1== All heap blocks were freed -- no leaks are possible\n==1== ERROR SUMMARY: 0 errors from 0 contexts\n"""
    sandbox = Mock()
    sandbox.run.side_effect = [
        {"exit_code": 0, "output": "", "timed_out": False, "output_may_be_truncated": False},
        {"exit_code": 0, "output": "", "timed_out": False, "output_may_be_truncated": False},
        {"exit_code": 0, "output": clean_valgrind, "timed_out": False, "output_may_be_truncated": False},
        {"exit_code": 0, "output": "", "timed_out": False, "output_may_be_truncated": False},
        {"exit_code": 1, "output": "ERROR: AddressSanitizer: heap-buffer-overflow", "timed_out": False, "output_may_be_truncated": False},
    ]
    result = run_memory_safety(
        sandbox, "int main(void) { return 0; }", [{"input": "", "expected": ""}], run_sanitizers=True,
    )
    assert result["runtime_tools"]["valgrind_score"] == 100
    assert result["runtime_tools"]["asan_ubsan_score"] == 0
    assert result["score"] == 0
    assert result["runtime_tools"]["asan_ubsan_results"][0]["address_sanitizer_finding"] is True


def test_r06_program_understanding_fallback_status_and_repair_diagnostics_are_preserved():
    report = {"problem": "Read one integer and print it.", "source_code": "int main(void) { return 0; }\n"}
    rules = {"problem": report["problem"], "problem_contract": {}, "metrics": []}
    diagnostics = ["program_understanding attempt 1/2: invalid object", "program_understanding attempt 2/2: invalid object"]
    with patch.object(reviewer, "_call_grading_json", return_value=(None, diagnostics, [0.1, 0.2], 2)):
        understanding, metadata = reviewer.understand_program_with_llm(report, rules)
    assert understanding is not None
    assert metadata["status"] == "completed_with_fallback"
    assert metadata["fallback_used"] is True
    assert len(metadata["diagnostics"]) == 2
    assert "invalid object" in metadata["error"]


def test_compiler_diagnostics_preserve_separate_streams_and_elapsed_time():
    class StreamSandbox:
        def write_file(self, filename, source):
            pass

        def run_with_streams(self, command, timeout_s):
            return {
                "exit_code": 0, "timed_out": False,
                "stdout": "", "stderr": "submission.c:1: warning: example warning\n",
                "output": "submission.c:1: warning: example warning\n",
            }

    result = run_compiler_diagnostics(StreamSandbox(), "int main(void) { return 0; }\n")
    assert result["compiler_streams_separated"] is True
    assert result["compiler_stdout"] == ""
    assert "warning" in result["compiler_stderr"]
    assert result["compile_elapsed_ms"] >= 0


def test_r07_invalid_finding_line_is_retained_only_as_explicitly_unresolved():
    source = "int main(void) {\n  return 0;\n}\n"
    result = reviewer._normalise_global_review(
        {"overall_assessment": "A possible defect.", "findings": [{
            "metric": "correctness", "severity": "high", "title": "Wrong output",
            "description": "The program prints the wrong value.", "line_start": 99, "line_end": 99,
            "evidence_ids": ["E-TEST-1"], "confidence": 0.9,
        }]},
        [{"name": "correctness"}], source,
        [{"id": "E-TEST-1", "kind": "test_result", "facts": {}}],
        {1, 2, 3},
    )
    assert result is not None
    assert result["findings"][0]["line_start"] is None
    assert result["findings"][0]["source_line_status"] == "unresolved_invalid_reference"


def test_r08_unsupported_generic_rsi_finding_is_rejected_with_reason():
    finding = {"metric": "correctness", "title": "Potential logic issue", "description": "A logic issue may exist.", "line_start": 1}
    reason = reviewer._finding_prevalidation_reason(
        finding, "int main(void) { return 0; }", {"problem": "Compute an arithmetic mean.", "problem_contract": {}},
        {"problem": "Compute an arithmetic mean.", "correctness": {"tests_total": 0}},
    )
    assert reason is not None
    assert "generic" in reason.lower() or "unmeasurable" in reason.lower()


def test_r09_llm_request_events_reconcile_to_the_reported_aggregate():
    class Response:
        ok = True
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "{\"ok\":true}"}}]}

    events: list[dict] = []
    token = reviewer._LLM_REQUEST_EVENTS.set(events)
    try:
        with patch.object(reviewer.requests, "post", return_value=Response()):
            reviewer._request_json([{"role": "user", "content": "Return JSON."}], schema_name="metric_scoring")
    finally:
        reviewer._LLM_REQUEST_EVENTS.reset(token)
    assert len(events) == 1
    assert events[0]["stage"] == "metric_scoring"
    assert events[0]["clock"] == "monotonic"
    assert events[0]["ended_monotonic_seconds"] >= events[0]["started_monotonic_seconds"]
    assert abs(events[0]["duration_seconds"] - (events[0]["ended_monotonic_seconds"] - events[0]["started_monotonic_seconds"])) < 0.000002
    total = sum(event["duration_seconds"] for event in events)
    assert total == events[0]["duration_seconds"]
    assert events[0]["status"] == "response_received"


def test_r10_completed_logical_tests_are_not_described_as_unavailable():
    metrics = [{"name": "logical", "analyzer": "edge_case_runner", "weight": 1.0, "enabled": True}]
    report = {"logical": {"status": "completed", "tests_total": 2, "tests_passed": 2, "rationale": "logical could not be measured from available evidence."}}
    apply_metric_scores(metrics, report, None)
    rendered = render_text_report({
        "configuration": {"enabled_metrics": [{"name": "logical", "weight": 1.0}]},
        "logical": report["logical"], "scoring": {"status": "completed", "score": 100, "weights_sum": 1, "contributions": [{"metric": "logical", "weight": 1, "score": 100, "contribution": 100}]},
    })
    assert "logical tests passed 2/2" in rendered
    assert "could not be measured" not in rendered.lower()


def test_r11_runtime_target_overrun_is_explicit_and_reports_summary_statistics():
    class SlowSandbox:
        def write_file(self, filename, source):
            pass

        def run(self, command, timeout_s, stdin=None):
            if command[0] == "gcc":
                return {"exit_code": 0, "output": "", "timed_out": False}
            time.sleep(0.002)
            return {"exit_code": 0, "output": "", "timed_out": False, "output_may_be_truncated": False}

    result = run_runtime_profile(
        SlowSandbox(), "int main(void) { return 0; }", [{"id": "normal", "input": "", "expected": ""}],
        target_runtime_ms=0.001, repetitions=2,
    )
    assert result["status"] == "completed"
    assert result["mean_runtime_ms"] > 0
    assert result["median_runtime_ms"] > 0
    assert result["max_runtime_ms"] >= result["median_runtime_ms"]
    assert result["target_runtime_met"] is False
    assert result["score"] < 100


def test_r12_missing_objective_evidence_is_not_replaced_by_an_llm_success_score():
    metrics = [{"name": "runtime_efficiency", "analyzer": "runtime_profile", "weight": 1.0, "enabled": True}]
    report = {"runtime_efficiency": {"status": "inconclusive", "score": None, "reason": "No runtime test set."}}
    apply_metric_scores(metrics, report, {"metrics": [{"name": "runtime_efficiency", "score": 10, "origin": "llm"}]})
    final = calculate_overall_score(metrics, report)
    assert report["runtime_efficiency"]["score"] is None
    assert report["runtime_efficiency"]["score_evidence_class"] == "unavailable"
    assert final["status"] == "failed_incomplete"
    assert final["score"] is None
