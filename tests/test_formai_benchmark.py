"""Stratified FormAI benchmark: index, sampling, metrics, property checks, run."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from grading_agent.analyzers.c_ast import analyze_c_source, parse_c, preprocess  # noqa: E402
from grading_agent.analyzers.c_properties import check_c_properties  # noqa: E402
from grading_agent.benchmark import formai_dataset, formai_metrics, formai_sampling  # noqa: E402
from grading_agent.benchmark.formai_benchmark import BenchmarkConfig, iter_benchmark  # noqa: E402
from grading_agent.config import settings  # noqa: E402

TEST_SPEC = formai_dataset.DatasetSpec(key="t", name="FormAI-test", directory="formai_t", downloads=(), kind="json_7z")


def _record(name, category, line=None, error=None, source=None):
    return {"file_name": name, "category": category, "source_code": source or f"// {name}\nint main(void) {{ return 0; }}\n",
            "line": line, "function": "main", "error_type": error, "num_lines": 2, "cyclomatic_complexity": 1.0}


def _synthetic_records():
    records = []
    # 30 programs: 12 scanf, 8 null deref, 10 safe; plus mixed and excluded ones.
    for i in range(12):
        records.append(_record(f"gpt35-{i}.c", "VULNERABLE", 2, "buffer overflow on scanf"))
    for i in range(12, 20):
        records.append(_record(f"falcon180b-{i}.c", "VULNERABLE", 2, "dereference failure: NULL pointer"))
    for i in range(20, 30):
        records.append(_record(f"gemma7b-{i}.c", "SAFE"))
    records += [
        _record("mixed-1.c", "VULNERABLE", 9, "dereference failure: NULL pointer"),
        _record("mixed-1.c", "VULNERABLE", 4, "arithmetic overflow on add"),
        _record("mixed-1.c", "UNKNOWN"),
        _record("timeout-1.c", "UNKNOWN"),
        _record("timeout-2.c", "SAFE"),
        _record("timeout-2.c", "UNKNOWN"),
        _record("conflict-1.c", "VULNERABLE", 1, "division by zero", source="int a;\n"),
        _record("conflict-1.c", "VULNERABLE", 1, "division by zero", source="int b;\n"),
    ]
    return records


class _IndexCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patches = [
            patch.object(settings, "FORM_AI_DATA_ROOT", Path(self.temp.name)),
            patch.object(settings, "FORM_AI_RUNS_DIR", Path(self.temp.name) / "runs"),
            patch.dict(formai_dataset.CATALOG, {"t": TEST_SPEC}),
            patch.object(formai_dataset, "iter_records", lambda version, progress=None: iter(_synthetic_records())),
        ]
        for item in self.patches:
            item.start()
        formai_dataset.build_index("t")

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()


class DatasetIndexTests(_IndexCase):
    def test_program_labels_follow_the_policy(self):
        summary = formai_dataset.index_summary("t")
        self.assertEqual(summary["labels"], {"VULNERABLE": 21, "SAFE": 10, "UNKNOWN": 3})
        with formai_dataset.connect("t") as connection:
            labels = formai_dataset.load_labels(connection, ["mixed-1.c", "timeout-2.c", "conflict-1.c"])
            row = connection.execute("SELECT primary_family, family_count FROM programs WHERE file_name='mixed-1.c'").fetchone()
        mixed = labels["mixed-1.c"]
        self.assertEqual(mixed.label, "VULNERABLE")          # a counterexample wins over a time-out
        self.assertFalse(mixed.verification_complete)        # ...but verification was incomplete
        self.assertEqual(mixed.lines, [4, 9])
        self.assertEqual(tuple(row), ("integer_overflow", 2))  # family of the earliest line
        self.assertEqual(labels["timeout-2.c"].label, "UNKNOWN")   # a time-out is not evidence of safety
        self.assertEqual(labels["conflict-1.c"].label, "UNKNOWN")  # two sources under one name

    def test_generators_and_sources_are_separate_from_labels(self):
        with formai_dataset.connect("t") as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(programs)")}
            sources = formai_dataset.load_sources(connection, ["gpt35-1.c"])
        self.assertNotIn("source_code", columns)
        self.assertIn("int main", sources["gpt35-1.c"])
        self.assertEqual(set(formai_dataset.index_summary("t")["generators"]) >= {"gpt35", "falcon180b", "gemma7b"}, True)

    def test_family_mapping_covers_esbmc_cppcheck_and_llm_vocabularies(self):
        cases = {
            "buffer overflow on scanf": "input_buffer_overflow", "invalidscanf": "input_buffer_overflow",
            "dereference failure: NULL pointer": "null_dereference", "nullPointerOutOfMemory": "null_dereference",
            "dereference failure: array bounds violated": "out_of_bounds", "arrayIndexOutOfBounds": "out_of_bounds",
            "dereference failure: forgotten memory": "memory_leak", "memleak": "memory_leak",
            "arithmetic overflow on mul": "integer_overflow", "arithmetic overflow on floating-point ieee_div": "float_overflow",
            "division by zero": "division_by_zero", "dereference failure: invalidated dynamic object": "invalid_pointer",
            "doubleFree": "invalid_pointer", "uninitvar": "other", "": "other", "out_of_bounds": "out_of_bounds",
        }
        for text, family in cases.items():
            self.assertEqual(formai_dataset.error_family(text), family, text)


class SamplingTests(_IndexCase):
    def test_proportional_allocation_sums_and_gives_every_stratum_a_program(self):
        with formai_dataset.connect("t") as connection:
            sample = formai_sampling.draw_sample(connection, dataset="t", n=10, seed=3)
        sizes = {s.key: s.allocated for s in sample.strata}
        self.assertEqual(sample.size, 10)
        self.assertTrue(all(count >= 1 for count in sizes.values()))
        for stratum in sample.strata:
            self.assertAlmostEqual(stratum.weight, stratum.population / stratum.allocated)
        self.assertEqual(len({p["file_name"] for p in sample.programs}), 10)

    def test_balanced_allocation_splits_labels_evenly(self):
        with formai_dataset.connect("t") as connection:
            sample = formai_sampling.draw_sample(connection, dataset="t", n=10, allocation="balanced", seed=3)
        by_label = {"VULNERABLE": 0, "SAFE": 0}
        for stratum in sample.strata:
            by_label[stratum.label] += stratum.allocated
        self.assertEqual(by_label, {"VULNERABLE": 5, "SAFE": 5})

    def test_same_seed_repeats_and_larger_samples_extend_smaller_ones(self):
        with formai_dataset.connect("t") as connection:
            first = formai_sampling.draw_sample(connection, dataset="t", n=6, stratify_by="label", seed=9)
            again = formai_sampling.draw_sample(connection, dataset="t", n=6, stratify_by="label", seed=9)
            larger = formai_sampling.draw_sample(connection, dataset="t", n=12, stratify_by="label", seed=9)
        names = [p["file_name"] for p in first.programs]
        self.assertEqual(names, [p["file_name"] for p in again.programs])
        self.assertTrue(set(names) <= {p["file_name"] for p in larger.programs})

    def test_sample_larger_than_population_is_capped(self):
        with formai_dataset.connect("t") as connection:
            sample = formai_sampling.draw_sample(connection, dataset="t", n=500, seed=1)
        self.assertEqual(sample.size, 31)  # 21 vulnerable + 10 safe eligible


def _metric_record(stratum, weight, truth, predicted, findings=(), lines=(), families=()):
    return {"file_name": f"{stratum}-{id(findings)}-{predicted}", "stratum": stratum, "weight": weight, "truth": truth,
            "predicted": predicted, "findings": list(findings), "truth_lines": list(lines),
            "truth_families": list(families),
            "truth_vulnerabilities": [{"line": line, "family": family} for line, family in zip(lines, families)],
            "predicted_line": findings[0]["line"] if findings else None}


class MetricsTests(unittest.TestCase):
    def test_inconclusive_counts_as_error_only_in_strict_views(self):
        records = [
            _metric_record("V", 1, True, True), _metric_record("V", 1, True, None),
            _metric_record("S", 1, False, False), _metric_record("S", 1, False, None),
        ]
        strata = [{"key": "V", "label": "VULNERABLE", "population": 2, "weight": 1},
                  {"key": "S", "label": "SAFE", "population": 2, "weight": 1}]
        result = formai_metrics.evaluate(records, strata=strata, bootstrap_samples=0)
        self.assertEqual(result["metrics"]["sample"]["confusion"], {"tp": 1, "fp": 1, "fn": 1, "tn": 1})
        self.assertEqual(result["metrics"]["decided"]["confusion"], {"tp": 1, "fp": 0, "fn": 0, "tn": 1})
        self.assertEqual(result["metrics"]["decided"]["precision"], 1.0)
        self.assertEqual(result["metrics"]["sample"]["recall"], 0.5)
        self.assertEqual(result["coverage"], 0.5)

    def test_population_weights_correct_precision_for_balanced_samples(self):
        # 90% of the population is vulnerable but the sample is 50/50.
        records = [_metric_record("V", 9.0, True, True) for _ in range(5)] + \
                  [_metric_record("S", 1.0, False, i == 0) for i in range(5)]
        strata = [{"key": "V", "label": "VULNERABLE", "population": 45, "weight": 9.0},
                  {"key": "S", "label": "SAFE", "population": 5, "weight": 1.0}]
        result = formai_metrics.evaluate(records, strata=strata, bootstrap_samples=0)
        self.assertAlmostEqual(result["metrics"]["sample"]["precision"], 5 / 6, places=4)
        self.assertAlmostEqual(result["metrics"]["population"]["precision"], 45 / 46, places=4)

    def test_inconclusive_reasons_and_positive_evidence_are_reported(self):
        from grading_agent.benchmark.formai_benchmark import inconclusive_reason, positive_evidence
        truncated = {"predicted_vulnerable": None, "llm_findings": [],
                     "llm_error": "Malformed final assistant JSON; finish_reason=length"}
        unconfirmed = {"predicted_vulnerable": None, "llm_findings": [{"line": 3}],
                       "llm_findings_not_confirmed_by_rsi": [{"line": 3}]}
        self.assertEqual(inconclusive_reason(truncated), "LLM output truncated")
        self.assertEqual(inconclusive_reason(unconfirmed), "LLM candidates not confirmed by RSI")
        self.assertIsNone(inconclusive_reason({"predicted_vulnerable": False}))
        tool_hit = {"predicted_vulnerable": True, "predicted_findings": [{"source": "cppcheck"}]}
        both = {"predicted_vulnerable": True, "predicted_findings": [{"source": "syntax_tree"}, {"source": "llm"}]}
        self.assertEqual(positive_evidence(tool_hit), "static tools only")
        self.assertEqual(positive_evidence(both), "static tools + LLM/RSI")

        records = [
            {**_metric_record("V", 1, True, None), "inconclusive_reason": "LLM output truncated"},
            {**_metric_record("V", 1, True, True), "positive_evidence": "static tools only"},
            {**_metric_record("S", 1, False, True), "positive_evidence": "LLM/RSI only"},
        ]
        strata = [{"key": "V", "label": "VULNERABLE", "population": 2, "weight": 1},
                  {"key": "S", "label": "SAFE", "population": 1, "weight": 1}]
        result = formai_metrics.evaluate(records, strata=strata, bootstrap_samples=0)
        self.assertEqual(result["inconclusive_reasons"], {"LLM output truncated": {"vulnerable": 1, "safe": 0}})
        self.assertEqual(result["positive_evidence"]["LLM/RSI only"], {"tp": 0, "fp": 1})

    def test_undefined_ratios_are_null(self):
        metrics = formai_metrics.binary_metrics(0, 0, 0, 3)
        self.assertIsNone(metrics["precision"])
        self.assertIsNone(metrics["recall"])
        self.assertEqual(metrics["specificity"], 1.0)

    def test_location_and_family_agreement(self):
        good = [{"line": 11, "family": "null_dereference", "source": "llm"}]
        wrong_family = [{"line": 20, "family": "out_of_bounds", "source": "llm"}]
        records = [
            _metric_record("V", 1, True, True, good, lines=(10,), families=("null_dereference",)),
            _metric_record("V", 1, True, True, wrong_family, lines=(20,), families=("null_dereference",)),
        ]
        strata = [{"key": "V", "label": "VULNERABLE", "population": 2, "weight": 1}]
        loc = formai_metrics.evaluate(records, strata=strata, line_tolerance=1, bootstrap_samples=0)["localisation"]
        self.assertEqual(loc["any_finding_line_hit"]["sample"], 1.0)
        self.assertEqual(loc["line_and_family_match"]["sample"], 0.5)

    def test_bootstrap_intervals_are_reproducible_and_bracket_the_estimate(self):
        records = [_metric_record("V", 2, True, i % 3 != 0) for i in range(30)] + \
                  [_metric_record("S", 1, False, i % 5 == 0) for i in range(30)]
        strata = [{"key": "V", "label": "VULNERABLE", "population": 60, "weight": 2},
                  {"key": "S", "label": "SAFE", "population": 30, "weight": 1}]
        first = formai_metrics.evaluate(records, strata=strata, bootstrap_samples=300, seed=5)
        second = formai_metrics.evaluate(records, strata=strata, bootstrap_samples=300, seed=5)
        low, high = first["metrics"]["population"]["ci95"]["recall"]
        self.assertEqual(first["metrics"]["population"]["ci95"], second["metrics"]["population"]["ci95"])
        self.assertLessEqual(low, first["metrics"]["population"]["recall"])
        self.assertGreaterEqual(high, first["metrics"]["population"]["recall"])


class PropertyCheckTests(unittest.TestCase):
    def findings(self, code):
        result = check_c_properties(code)
        self.assertEqual(result["status"], "completed", result.get("error"))
        return {(f["check"], f["line"]) for f in result["findings"]}

    def test_unbounded_scanf_and_gets(self):
        code = ('#include <stdio.h>\nint main(void) {\n  char a[8], b[8];\n  scanf("%s", a);\n'
                '  scanf("%7s", b);\n  gets(a);\n  return 0;\n}\n')
        found = self.findings(code)
        self.assertIn(("unbounded_input", 4), found)
        self.assertIn(("unbounded_input", 6), found)
        self.assertNotIn(("unbounded_input", 5), found)

    def test_unchecked_allocation_but_not_after_a_null_check(self):
        code = ('#include <stdlib.h>\ntypedef struct { int v; } N;\nint main(void) {\n'
                '  N *a = malloc(sizeof(N));\n  a->v = 1;\n'
                '  N *b = malloc(sizeof(N));\n  if (b == NULL) return 1;\n  b->v = 2;\n'
                '  free(a); free(b);\n  return 0;\n}\n')
        found = self.findings(code)
        self.assertIn(("unchecked_allocation", 5), found)
        self.assertNotIn(("unchecked_allocation", 8), found)

    def test_unreachable_function_is_not_reported(self):
        code = ('#include <stdlib.h>\nvoid unused(void) {\n  int *p = malloc(4);\n  *p = 1;\n  free(p);\n}\n'
                'int main(void) { return 0; }\n')
        self.assertEqual(self.findings(code), set())

    def test_loop_bound_reaching_the_array_size(self):
        code = 'int main(void) {\n  int a[10];\n  for (int i = 0; i <= 10; i++)\n    a[i] = i;\n  return 0;\n}\n'
        self.assertIn(("loop_out_of_bounds", 4), self.findings(code))

    def test_input_used_as_divisor_and_index_without_checks(self):
        code = ('#include <stdio.h>\nint main(void) {\n  int n, k, a[5] = {0};\n  scanf("%d %d", &n, &k);\n'
                '  int q = 100 / n;\n  a[k] = q;\n  return 0;\n}\n')
        found = self.findings(code)
        self.assertIn(("unchecked_input_divisor", 5), found)
        self.assertIn(("unchecked_input_index", 6), found)

    def test_checked_input_is_not_reported(self):
        code = ('#include <stdio.h>\nint main(void) {\n  int n;\n  if (scanf("%d", &n) != 1 || n == 0) return 1;\n'
                '  int q = 100 / n;\n  return q;\n}\n')
        self.assertNotIn(("unchecked_input_divisor", 5), self.findings(code))

    def test_leak_is_reported_at_the_end_of_main(self):
        code = '#include <stdlib.h>\nint main(void) {\n  int *p = malloc(8);\n  if (!p) return 1;\n  p[0] = 1;\n  return 0;\n}\n'
        self.assertIn(("memory_leak", 7), self.findings(code))


class ParserRobustnessTests(unittest.TestCase):
    def test_unknown_library_types_are_declared_and_parsing_retries(self):
        code = ('#include <pthread.h>\n#include <unistd.h>\nint main(void) {\n  pid_t pid = 0;\n  pthread_t worker;\n'
                '  socklen_t len = 0;\n  DIR *dir = 0;\n  return (int)pid + (int)len + (dir != 0);\n}\n')
        tree, error = parse_c(code)
        self.assertIsNotNone(tree, error)
        self.assertEqual(analyze_c_source(code)["status"], "completed")

    def test_macro_values_with_backslashes_are_substituted_literally(self):
        text = preprocess('#define ESC "\\x1b[0m"\nconst char *s = ESC;\n')
        self.assertIn('("\\x1b[0m")', text)


class ClaimVerificationTests(unittest.TestCase):
    SOURCE = "#include <stdlib.h>\nint main(void) {\n  int *p = malloc(4);\n  *p = 1;\n  return 0;\n}\n"
    CLAIMS = [
        {"line": 4, "category": "null_dereference", "description": "p may be NULL"},
        {"line": 5, "category": "out_of_bounds", "description": "speculative"},
    ]

    def test_iterative_passes_reject_on_any_failed_check_and_stop_when_stable(self):
        from grading_agent.llm import reviewer

        replies = [
            {"verdicts": [
                {"id": "C1", "reason": "malloc result p is never tested", "reachable": True, "guard_prevents_it": False, "concrete_trigger": True},
                {"id": "C2", "reason": "return statement has no index at all", "reachable": True, "guard_prevents_it": False, "concrete_trigger": False},
            ]},
            {"verdicts": [
                {"id": "C1", "reason": "still no NULL test before *p", "reachable": True, "guard_prevents_it": False, "concrete_trigger": True},
            ]},
        ]
        seen = []

        def fake(messages, **kwargs):
            seen.append(json.loads(messages[1]["content"]))
            return replies[len(seen) - 1], ""

        with patch.object(reviewer, "_request_json", side_effect=fake):
            result = reviewer.verify_vulnerability_claims(self.SOURCE, self.CLAIMS, max_passes=3)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["confirmed_ids"], ["C1"])
        self.assertEqual(result["rejected_ids"], ["C2"])
        self.assertEqual(len(seen), 2)  # pass 2 confirmed the same set, so pass 3 is skipped
        self.assertEqual([c["id"] for c in seen[1]["claims"]], ["C1"])
        self.assertIn("previous_assessment", seen[1]["claims"][0])

    def test_missing_verdict_makes_verification_incomplete(self):
        from grading_agent.llm import reviewer

        with patch.object(reviewer, "_request_json", return_value=({"verdicts": []}, "")):
            result = reviewer.verify_vulnerability_claims(self.SOURCE, self.CLAIMS)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["confirmed_claims"], [])

    def test_verified_fusion_is_safe_only_when_every_stage_completed(self):
        from grading_agent.benchmark.formai_predict import _select_verified_prediction

        audit = {"status": "completed", "audit": {"findings": []}}
        clean = {"status": "completed", "confirmed_claims": []}
        self.assertEqual(_select_verified_prediction(tool_status="completed", tool_findings=[], llm_audit=audit,
                                                     verification=clean), (False, None, None, "completed"))
        self.assertEqual(_select_verified_prediction(tool_status="completed", tool_findings=[], llm_audit=audit,
                                                     verification={"status": "incomplete"})[0], None)
        self.assertIsNone(_select_verified_prediction(tool_status="unavailable", tool_findings=[], llm_audit=audit,
                                                      verification=None)[0])
        confirmed = {"status": "completed", "confirmed_claims": [{"line": 9, "category": "out_of_bounds"}]}
        self.assertEqual(
            _select_verified_prediction(tool_status="completed", tool_findings=[{"line": 12, "id": "memleak"}],
                                        llm_audit=audit, verification=confirmed),
            (True, 9, "out_of_bounds", "completed"))


class _FakeSession:
    """Predicts from source text alone; records the order of calls."""
    calls: list[str] = []

    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def predict(self, item, on_stage=None):
        _FakeSession.calls.append("predict")
        if on_stage:
            on_stage("static analysis")
        vulnerable = "safe" not in item["source_code"]
        return {"file_name": item["file_name"], "predicted_vulnerable": vulnerable, "prediction_status": "completed",
                "predicted_line": 2 if vulnerable else None,
                "predicted_findings": [{"line": 2, "family": "input_buffer_overflow", "source": "fake"}] if vulnerable else [],
                "timings": {"total_s": 0.01}}


class BenchmarkRunTests(_IndexCase):
    def test_end_to_end_run_reads_labels_only_after_every_prediction(self):
        _FakeSession.calls = []
        original = formai_dataset.load_labels

        def tracked(connection, names):
            _FakeSession.calls.append("labels")
            return original(connection, names)

        config = BenchmarkConfig(dataset="t", sample_size=8, allocation="balanced", seed=4, mode="tools_only",
                                 bootstrap_samples=50)
        with patch.object(formai_dataset, "load_labels", tracked), \
             patch("grading_agent.benchmark.formai_dataset.parse_header", formai_dataset.parse_header):
            events = list(iter_benchmark(config, session_factory=_FakeSession))
        self.assertEqual(_FakeSession.calls, ["predict"] * 8 + ["labels"])
        done = events[-1]
        self.assertEqual(done["event"], "done")
        self.assertTrue(any(e["event"] == "program" and e["stage"] == "static analysis" for e in events))
        evaluation = done["evaluation"]["evaluation"]
        self.assertEqual(evaluation["programs"], 8)
        run_dir = Path(done["run_dir"])
        for name in ("config.json", "sample.json", "inputs.jsonl", "predictions.jsonl", "labels.json", "evaluation.json", "report.md"):
            self.assertTrue((run_dir / name).is_file(), name)
        inputs = [json.loads(line) for line in (run_dir / "inputs.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(set(item) == {"file_name", "source_code"} for item in inputs))  # no labels in model inputs
        self.assertIn("Dataset estimate", done["report"])


if __name__ == "__main__":
    unittest.main()
