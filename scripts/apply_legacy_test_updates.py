#!/usr/bin/env python3
"""Update legacy tests whose assumptions conflict with analyzer-authoritative grading."""
from pathlib import Path
import re


def replace_once(path: str, old: str, new: str, label: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Refusing test update {label}: expected one block in {path}, found {count}.")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_method(path: str, name: str, replacement: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    indent = "    " if re.search(rf"(?m)^    def {re.escape(name)}\(", text) else ""
    pattern = re.compile(rf"(?ms)^{re.escape(indent)}def {re.escape(name)}\(.*?(?=^{re.escape(indent)}def |^if __name__ ==|\Z)")
    updated, count = pattern.subn(replacement.rstrip() + "\n\n", text, count=1)
    if count != 1:
        raise SystemExit(f"Refusing test update: method {name} was not found uniquely in {path}.")
    file.write_text(updated, encoding="utf-8")

# Global RSI tests: the LLM returns anchored bands; its proposed midpoint is
# advisory. Analyzer scores remain the deterministic grade.
replace_once(
    "tests/test_global_rsi.py",
    '''def metric_score_response(correctness: int = 95, syntax: int = 94) -> dict:
    # The production scoring format is deliberately compact to avoid token-cap truncation.
    return {"scores": {"correctness": correctness, "syntax": syntax}}
''',
    '''def metric_score_response(correctness: str = "95-100", syntax: str = "90-95") -> dict:
    # The model must choose a documented five-point band, not a free-form number.
    return {"scores": {
        "correctness": {"score_band": correctness, "rationale": "The test evidence supports this advisory band.", "zero_score_reason": "", "confidence": 0.9},
        "syntax": {"score_band": syntax, "rationale": "The compiler evidence supports this advisory band.", "zero_score_reason": "", "confidence": 0.9},
    }}
''',
    "global RSI score response helper",
)
replace_once("tests/test_global_rsi.py", "self.assertEqual(result[\"metrics\"][0][\"score\"], 95)", "self.assertEqual(result[\"metrics\"][0][\"score\"], 97.5)", "global RSI correctness midpoint")
replace_once("tests/test_global_rsi.py", "self.assertEqual(result[\"metrics\"][1][\"score\"], 94)", "self.assertEqual(result[\"metrics\"][1][\"score\"], 92.5)", "global RSI syntax midpoint")
replace_once("tests/test_global_rsi.py", "self.assertEqual(report[\"correctness\"][\"score\"], 95)", "self.assertEqual(report[\"correctness\"][\"score\"], 100)", "global RSI deterministic correctness")
replace_once("tests/test_global_rsi.py", "self.assertEqual(report[\"syntax\"][\"score\"], 94)", "self.assertEqual(report[\"syntax\"][\"score\"], 100)", "global RSI deterministic syntax")
replace_once("tests/test_global_rsi.py", "self.assertEqual(overall[\"score\"], 94.6)", "self.assertEqual(overall[\"score\"], 100.0)", "global RSI analyzer weighted grade")
replace_once(
    "tests/test_global_rsi.py",
    '''                    "score": 88,
                    "rationale": "Functional behavior matches the supplied contract.",
''',
    '''                    "score_band": "85-90",
                    "rationale": "Functional behavior matches the supplied contract.",
''',
    "partial metric response score band",
)
replace_once("tests/test_global_rsi.py", "self.assertEqual(result[\"metrics\"][0][\"score\"], 88)", "self.assertEqual(result[\"metrics\"][0][\"score\"], 87.5)", "partial metric score midpoint")

# Reviewer context tests: keep their test focus but use the new score-band contract.
replace_once(
    "tests/test_reviewer_context.py",
    '''            "name": "correctness", "score": 95, "rationale": "Configured case passed.",
''',
    '''            "name": "correctness", "score_band": "95-100", "rationale": "Configured case passed.",
''',
    "reviewer context partial band",
)
replace_once("tests/test_reviewer_context.py", "self.assertEqual(result[\"metrics\"][0][\"score\"], 95)", "self.assertEqual(result[\"metrics\"][0][\"score\"], 97.5)", "reviewer context partial midpoint")
replace_once(
    "tests/test_reviewer_context.py",
    '''            "name": "logical", "score": 84, "rationale": "Source inspection suggests the control flow is sound; runtime behavior was not tested.",
''',
    '''            "name": "logical", "score_band": "80-85", "rationale": "Source inspection suggests the control flow is sound; runtime behavior was not tested.",
''',
    "reviewer context source-only band",
)
replace_once("tests/test_reviewer_context.py", "self.assertEqual(result[\"metrics\"][0][\"score\"], 84)", "self.assertEqual(result[\"metrics\"][0][\"score\"], 82.5)", "reviewer context source-only midpoint")

# The pure scoring and fallback tests now assert N/A for unmeasured metrics.
replace_method(
    "tests/test_final_user_fixes.py",
    "test_llm_score_is_authoritative_over_analyzer_score",
    '''    def test_llm_band_is_advisory_and_analyzer_score_is_authoritative(self):
        metrics = [{"name": "correctness", "weight": 1.0, "enabled": True}]
        report = {"correctness": {"score": 80.0, "status": "completed"}}
        llm = {"metrics": [{
            "name": "correctness", "score": 97.5, "score_band": "95-100",
            "rationale": "The source and test evidence support a very strong result.", "origin": "llm",
        }]}
        apply_metric_scores(metrics, report, llm)
        overall = calculate_overall_score(metrics, report)
        self.assertEqual(report["correctness"]["score"], 80.0)
        self.assertEqual(report["correctness"]["llm_review_score"], 97.5)
        self.assertEqual(report["correctness"]["score_origin"], "deterministic_analyzer")
        self.assertEqual(overall["score"], 80.0)
        self.assertEqual(report["correctness"]["llm_review"]["score_band"], "95-100")''',
)
replace_method(
    "tests/test_final_user_fixes.py",
    "test_inconclusive_zero_analyzer_score_is_not_used_as_a_zero",
    '''    def test_inconclusive_analyzer_and_llm_cannot_create_a_grade(self):
        metrics = [{"name": "correctness", "weight": 1.0, "enabled": True}]
        report = {"correctness": {"score": 0.0, "status": "inconclusive", "reason": "No tests executed."}}
        apply_metric_scores(metrics, report, {"metrics": []})
        overall = calculate_overall_score(metrics, report)
        self.assertIsNone(report["correctness"]["score"])
        self.assertEqual(report["correctness"]["score_origin"], "unmeasured")
        self.assertEqual(overall["status"], "failed_incomplete")
        self.assertIsNone(overall["score"])''',
)
replace_method(
    "tests/test_final_user_fixes.py",
    "test_inconclusive_analyzer_does_not_create_severity_based_zero_when_llm_score_is_missing",
    '''    def test_unmeasured_llm_fallback_is_not_a_neutral_numeric_score(self):
        fallback = reviewer._fallback_metric_score(
            "correctness",
            {"correctness": {"score": 0.0, "status": "inconclusive"}},
            [{"metric": "correctness", "severity": "critical", "title": "Possible contract mismatch", "evidence_ids": ["E-1"]}],
        )
        self.assertIsNone(fallback["score"])
        self.assertEqual(fallback["origin"], "insufficient_evidence_fallback")
        self.assertIn("not measured", fallback["rationale"])''',
)
replace_method(
    "tests/test_final_user_fixes.py",
    "test_generic_metric_rationale_is_replaced_without_discarding_score",
    '''    def test_generic_metric_rationale_is_replaced_for_a_valid_band(self):
        enabled = [{"name": "syntax", "enabled": True}]
        generic = {"scores": {"syntax": {
            "score_band": "95-100",
            "rationale": "No metric-specific defect was retained; available analyzer evidence supports this score.",
            "zero_score_reason": "", "confidence": 0.8,
        }}}
        specific = {"scores": {"syntax": {
            "score_band": "95-100",
            "rationale": "GCC compiled the source with the configured C11 warning flags and reported no diagnostics.",
            "zero_score_reason": "", "confidence": 0.9,
        }}}
        generic_result = reviewer._normalise_metric_scores(generic, enabled, [])
        self.assertEqual(generic_result["syntax"]["score"], 97.5)
        self.assertIn("did not provide a metric-specific explanation", generic_result["syntax"]["rationale"])
        self.assertEqual(reviewer._normalise_metric_scores(specific, enabled, [])["syntax"]["score_band"], "95-100")''',
)
replace_method(
    "tests/test_final_user_fixes.py",
    "test_valid_llm_score_is_not_discarded_when_rationale_is_generic",
    '''    def test_arbitrary_numeric_llm_score_is_rejected_without_a_band(self):
        parsed = {"scores": {"syntax": {
            "score": 96,
            "rationale": "No metric-specific defect was retained; available analyzer evidence supports this score.",
            "zero_score_reason": "", "confidence": 0.8,
        }}}
        self.assertEqual(reviewer._normalise_metric_scores(parsed, [{"name": "syntax"}], []), {})''',
)
replace_method(
    "tests/test_final_user_fixes.py",
    "test_scoring_reads_source_and_preserves_valid_llm_zero_without_retry",
    '''    def test_scoring_reads_source_and_accepts_only_a_documented_band(self):
        source = "int main(void) { puts(\\"hello\\"); return 0; }\\n"
        rules = {
            "problem": "grading", "problem_contract": {},
            "metrics": [{"name": "correctness", "enabled": True, "weight": 1.0, "analyzer": "io_tests", "params": {}}],
        }
        report = {"problem": "grading", "source_code": source, "correctness": {"tests_total": 0, "status": "inconclusive"}}
        first = {"scores": {"correctness": {
            "score_band": "0-5",
            "rationale": "The source violates the explicit required input and output contract.",
            "zero_score_reason": "The required input is never read and the output is absent.",
            "confidence": 0.95,
        }}}
        calls = []
        def fake_request(messages, **kwargs):
            calls.append((messages, kwargs))
            return first, "{}"
        with settings.runtime_overrides({
            "LLAMA_REASONING_ENABLED": True, "LLM_GLOBAL_SCORE_MAX_TOKENS": 2048,
            "LLM_GLOBAL_SOURCE_MAX_CHARS": 8000, "LLM_GLOBAL_MAX_FINDINGS": 12,
        }), patch.object(reviewer, "_request_json", side_effect=fake_request):
            scores, diagnostics, _latencies, request_count = reviewer._score_metrics_after_rsi(
                report, rules, {}, {"findings": [], "overall_assessment": "No concrete issues found."}, []
            )
        self.assertEqual(request_count, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(scores["correctness"]["score"], 2.5)
        self.assertEqual(scores["correctness"]["score_band"], "0-5")
        self.assertEqual(scores["correctness"]["origin"], "llm")
        self.assertIn("source_code_with_line_numbers", calls[0][0][1]["content"])
        self.assertFalse(calls[0][1]["enable_thinking"])
        self.assertEqual(diagnostics, [])''',
)

replace_method(
    "tests/test_pristine_llm_path.py",
    "test_valid_llm_zero_score_remains_authoritative",
    '''def test_valid_llm_band_is_advisory_when_analyzer_is_unmeasured():
    metrics = [{"name": "correctness", "enabled": True, "weight": 1.0}]
    report = {"correctness": {"score": None, "status": "inconclusive", "reason": "No tests executed."}}
    llm = {"metrics": [{
        "name": "correctness", "score": 2.5, "score_band": "0-5",
        "rationale": "The implementation does not satisfy the specified input and output contract.", "origin": "llm",
    }]}
    apply_metric_scores(metrics, report, llm)
    assert report["correctness"]["score"] is None
    assert report["correctness"]["score_origin"] == "unmeasured"
    assert report["correctness"]["llm_review_score"] == 2.5
    assert report["correctness"]["llm_review"]["score_band"] == "0-5"''',
)

# Analyzer fallback grades are never replaced by numeric LLM proposals; if a
# required analyzer cannot run, the overall report is N/A.
replace_method(
    "tests/test_grading_fallback.py",
    "test_llm_failure_still_returns_analyzer_weighted_grade",
    '''    def test_llm_failure_does_not_hide_incomplete_analyzer_metrics(self, _llm_rsi):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, rules_file = self._make_project(root)
            self._original_import = builtins.__import__
            with patch("builtins.__import__", side_effect=self._block_docker_import):
                report, _, _ = grade_with_rules_file(
                    rules_file, project_root=root, report_dir=root / "reports"
                )
        self.assertEqual(report["scoring"]["status"], "failed_incomplete")
        self.assertIsNone(report["scoring"]["score"])
        self.assertEqual(report["llm_rsi_evaluation"]["status"], "unavailable")
        self.assertTrue(any(item["metric"] == "syntax" for item in report["scoring"]["missing_metrics"]))''',
)
replace_method(
    "tests/test_grading_fallback.py",
    "test_final_grade_uses_rsi_scores_and_python_weights",
    '''    def test_llm_band_is_advisory_and_cannot_fill_missing_analyzer(self, llm_rsi):
        llm_rsi.return_value = {
            "status": "completed",
            "metrics": [
                {"name": "modularity", "score": 77.5, "score_band": "75-80", "origin": "llm", "rationale": "The source has the required helper decomposition."},
                {"name": "syntax", "score": 87.5, "score_band": "85-90", "origin": "llm", "rationale": "The source appears syntactically valid."},
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, rules_file = self._make_project(root)
            with patch("builtins.__import__", side_effect=self._block_docker_import):
                report, json_path, text_path = grade_with_rules_file(
                    rules_file, project_root=root, report_dir=root / "reports"
                )
                self.assertTrue(json_path.is_file())
                self.assertTrue(text_path.is_file())
        llm_rsi.assert_called_once()
        self.assertEqual(report["modularity"]["analyzer_score"], 100.0)
        self.assertEqual(report["modularity"]["score"], 100.0)
        self.assertEqual(report["modularity"]["llm_review_score"], 77.5)
        self.assertIsNone(report["syntax"]["analyzer_score"])
        self.assertIsNone(report["syntax"]["score"])
        self.assertEqual(report["syntax"]["llm_review_score"], 87.5)
        self.assertEqual(report["scoring"]["status"], "failed_incomplete")
        self.assertIsNone(report["scoring"]["score"])''',
)

print("Updated legacy score-authority tests to the five-point advisory-band contract.")

# Older one-request-per-metric tests no longer represent the current chained
# RSI coordinator. Replace them with direct schema/normalization invariants.
replace_method(
    "tests/test_reviewer_context.py",
    "test_rsi_uses_one_bounded_request_per_metric",
    '''    def test_metric_review_uses_only_documented_score_bands(self, request_json):
        from grading_agent.llm import reviewer
        schema = reviewer._metric_score_response_schema([{"name": "correctness"}])
        metric_schema = schema["properties"]["scores"]["properties"]["correctness"]
        self.assertIn("score_band", metric_schema["properties"])
        self.assertNotIn("score", metric_schema["properties"])
        payload = {"scores": {"correctness": {
            "score_band": "90-95",
            "rationale": "The test evidence supports a strong correctness result.",
            "zero_score_reason": "", "confidence": 0.9,
        }}}
        normalized = reviewer._normalise_metric_scores(payload, [{"name": "correctness"}], [])
        self.assertEqual(normalized["correctness"]["score"], 92.5)
        self.assertEqual(normalized["correctness"]["score_band"], "90-95")''',
)
replace_method(
    "tests/test_reviewer_context.py",
    "test_context_overflow_retries_compact_metric_payload",
    '''    def test_arbitrary_numeric_metric_score_is_not_accepted(self, request_json):
        from grading_agent.llm import reviewer
        payload = {"scores": {"correctness": {
            "score": 88,
            "rationale": "The implementation appears to meet the assignment requirements.",
            "zero_score_reason": "", "confidence": 0.8,
        }}}
        normalized = reviewer._normalise_metric_scores(
            payload, [{"name": "correctness"}], []
        )
        self.assertEqual(normalized, {})''',
)
print("Updated legacy reviewer-context tests for the anchored band-only advisory rubric.")

replace_method(
    "tests/test_reviewer_context.py",
    "test_metric_rsi_keeps_partial_scores_when_one_metric_is_missing",
    '''    def test_metric_rsi_keeps_partial_advisory_bands_when_one_metric_is_missing(self, request_json):
        from grading_agent.llm import reviewer
        enabled = [
            {"name": "correctness", "enabled": True, "weight": 0.5, "analyzer": "io_tests", "params": {}},
            {"name": "logical", "enabled": True, "weight": 0.5, "analyzer": "edge_case_runner", "params": {}},
        ]
        payload = {"metrics": [{
            "name": "correctness", "score_band": "95-100",
            "rationale": "Configured cases passed and the source follows the contract.",
            "zero_score_reason": "", "confidence": 0.9,
        }]}
        normalized = reviewer._normalise_metric_scores(payload, enabled, [])
        self.assertEqual(normalized["correctness"]["score"], 97.5)
        self.assertEqual(normalized["correctness"]["score_band"], "95-100")
        self.assertNotIn("logical", normalized)''',
)
replace_method(
    "tests/test_reviewer_context.py",
    "test_source_only_rsi_score_is_kept_when_metric_has_no_tool_evidence",
    '''    def test_source_only_advisory_band_is_distinct_from_analyzer_grade(self, request_json):
        from grading_agent.llm import reviewer
        enabled = [{"name": "logical", "enabled": True, "weight": 1.0, "analyzer": "edge_case_runner", "params": {}}]
        payload = {"metrics": [{
            "name": "logical", "score_band": "80-85",
            "rationale": "The source control flow appears consistent with the described invariant.",
            "zero_score_reason": "", "confidence": 0.7,
        }]}
        normalized = reviewer._normalise_metric_scores(payload, enabled, [])
        self.assertEqual(normalized["logical"]["score"], 82.5)
        self.assertEqual(normalized["logical"]["origin"], "llm")
        self.assertEqual(normalized["logical"]["evidence_ids"], [])''',
)
print("Updated legacy partial-metric tests to test advisory band normalization directly.")
