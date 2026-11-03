from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grader.analyzers import analyze_source, originality_similarity, score_structure, strip_comments_literals
from grader.grading import _calculate_grade, _compile_failure_result, _run_cases, _score_band, _syntax_result, run_rsi_review, grade
from grader.llm import _extract_json, extract_c_code, LLMError
from grader.rules import RulesError, _normalize_metrics, load_rules
from grader.suite import _clean_tests, _compare, _generate_test_inputs, prepare_suite

ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    model = "fake-model"
    max_tokens = 2048

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def json(self, system, user, max_tokens=None):
        if '"source_chunk_with_original_line_numbers"' in user:
            return {"functions": [], "control_flow": [], "data_flow": [], "io_behavior": [],
                    "memory_behavior": [], "invariants": [], "unresolved": []}
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeSandbox:
    def __init__(self, values):
        self.values = list(values)

    def run_program(self, program_name, stdin, timeout_s=2):
        expected = self.values.pop(0)
        return {"stdout": expected, "stderr": "", "exit_code": 0, "timed_out": False, "output_truncated": False}


class FakeFullLLM:
    model = "repeatable-test-model"
    max_tokens = 4096

    def __init__(self):
        self.json_calls = 0
        self.text_calls = 0

    def json(self, system, user, max_tokens=None):
        self.json_calls += 1
        if '"source_chunk_with_original_line_numbers"' in user:
            return {"functions": [], "control_flow": [], "data_flow": [], "io_behavior": [],
                    "memory_behavior": [], "invariants": [], "unresolved": []}
        if "RSI PASS 1" in user or '"task"' in user and '"previous_suite"' not in user:
            return {"inferred_contract": {
                "input_format": "one token", "output_format": "uppercase token plus newline",
                "semantics": "uppercase conversion", "constraints": ["one token"], "formatting": "newline"
            }, "tests": [
                {"id": "core_a", "input": "a\\n", "tags": ["core"], "rationale": "ordinary case"},
                {"id": "edge_a", "input": "b\\n", "tags": ["edge"], "rationale": "boundary case"},
            ], "review_summary": "Initial core and boundary coverage."}
        if '"previous_suite"' in user:
            # First RSI recheck adds a new input; the next one confirms the set.
            if 'RSI PASS 2' in user:
                return {"additional_tests": [{"id": "edge_b", "input": "c\n", "tags": ["adversarial"], "rationale": "corner case"}], "review_summary": "Added another boundary case."}
            return {"additional_tests": [], "review_summary": "No further gaps found."}
        return {"assessment": "All deterministic evidence reviewed.", "findings": [], "changes_from_previous": []}

    def text(self, system, user, max_tokens=None):
        self.text_calls += 1
        return "#include <stdio.h>\nint main(void) { puts(\"REF\"); return 0; }\n"

    def close(self):
        pass


class FakeDockerSandbox:
    def __init__(self, image, memory_mb, cpu_limit):
        self.image_id = "sha256:fixed-test-image"
        self.current_source = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def put_text(self, path, content):
        if path.endswith("submission.c"):
            self.current_source = content

    def compile(self, source_name, output_name, flags=None, timeout=15):
        return {"stdout": "", "stderr": "", "exit_code": 0, "timed_out": False, "warnings": []}

    def lock_source(self, work_name, locked_name):
        return None

    def lock_binary(self, work_name, locked_name):
        return None

    def run_program(self, program_name, stdin, timeout_s=2):
        if program_name == "reference":
            output = "REF\n"
        elif program_name == "student_san":
            output = ""
        else:
            output = "WRONG\n"
        return {"stdout": output, "stderr": "", "exit_code": 0, "timed_out": False, "output_truncated": False}

    def clear_workspaces(self):
        return None

    def compiler_version(self):
        return "gcc (test fixture)"


class CoreTests(unittest.TestCase):
    def test_rules_auto_detect_exactly_one_c_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rules_data = json.loads((ROOT / "rules.json").read_text())
            rules_data["submission"] = {"language": "c"}
            rules_data["problem"] = "Read two integers from standard input and print their arithmetic sum."
            rules_data["metrics"] = [{"name": "correctness", "analyzer": "io_tests", "weight": 1.0, "enabled": True, "params": {}}]
            (root / "rules.json").write_text(json.dumps(rules_data))
            (root / "student.c").write_text("int main(void) { return 0; }\n")
            rules = load_rules(root / "rules.json")
            self.assertEqual(rules["source_name"], "student.c")
            (root / "second.c").write_text("int main(void) { return 0; }\n")
            with self.assertRaises(RulesError):
                load_rules(root / "rules.json")
        rules_data = json.loads((ROOT / "rules.json").read_text())
        self.assertEqual(rules_data.get("submission", {}).get("language", "c"), "c")
        supported_metrics = [{"name": "correctness", "analyzer": "io_tests", "weight": 1.0, "enabled": True, "params": {}}]
        self.assertAlmostEqual(sum(m["weight"] for m in _normalize_metrics(supported_metrics) if m["enabled"]), 1.0)

    def test_cli_source_override_works_with_multiple_c_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rules_data = json.loads((ROOT / "rules.json").read_text())
            rules_data["problem"] = "Read two integers from standard input and print their arithmetic sum."
            rules_data["metrics"] = [{"name": "correctness", "analyzer": "io_tests", "weight": 1.0, "enabled": True, "params": {}}]
            (root / "rules.json").write_text(json.dumps(rules_data))
            (root / "first.c").write_text("int main(void) { return 0; }\n")
            (root / "second.c").write_text("int main(void) { return 1; }\n")
            with self.assertRaises(RulesError):
                load_rules(root / "rules.json")
            rules = load_rules(root / "rules.json", source_override="second.c")
            self.assertEqual(rules["source_name"], "second.c")

    def test_originality_similarity_flags_renamed_code_against_optional_corpus(self):
        program = (
            "#include <stdio.h>\n"
            "int main(void) {\n"
            "    long total = 0;\n"
            "    for (int index = 0; index < 100; ++index) {\n"
            + "        total += index * 3 + 7;\n" * 16
            + "    }\n"
            "    printf(\"%ld\\n\", total);\n"
            "    return 0;\n"
            "}\n"
        )
        renamed = program.replace("total", "accumulator").replace("index", "position")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_path = root / "current.c"
            source_path.write_text(program)
            corpus = root / "submissions"
            corpus.mkdir()
            (corpus / "peer.c").write_text("/* variable names changed */\n" + renamed)
            result = originality_similarity(program, source_path, corpus, min_tokens=40, threshold=0.85)
            self.assertTrue(result["flagged"])
            self.assertEqual(result["matched_file"], "peer.c")
            self.assertGreaterEqual(result["max_similarity"], 0.85)

    def test_sandbox_clears_reference_and_student_workspaces_as_supervisor(self):
        from grader.sandbox import DockerSandbox

        sandbox = DockerSandbox("fixture")
        calls = []
        sandbox._kill_submission_processes = lambda: calls.append(("kill", None))
        sandbox.exec = lambda args, **kwargs: calls.append((args, kwargs)) or {"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False}
        sandbox.clear_workspaces()
        self.assertEqual(calls[0][0], "kill")
        self.assertEqual([entry[0][1] for entry in calls[1:]], ["/work", "/locked"])
        self.assertTrue(all(entry[1]["user"] == "0:0" for entry in calls[1:]))

    def test_sandbox_compiles_only_from_locked_source(self):
        from grader.sandbox import DockerSandbox

        sandbox = DockerSandbox("fixture")
        calls = []
        sandbox.exec = lambda args, **kwargs: calls.append((args, kwargs)) or {"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False}
        sandbox.compile("submission.c", "student")
        command = calls[0][0]
        self.assertIn("/locked/submission.c", command)
        self.assertNotIn("/work/submission.c", command)
        with self.assertRaises(Exception):
            sandbox.compile("untrusted.c", "student")

    def test_existing_legacy_disabled_metrics_are_ignored(self):
        metrics = _normalize_metrics([
            {"name": "correctness", "analyzer": "io_tests", "weight": 0.8},
            {"name": "syntax", "analyzer": "compiler_diagnostics", "weight": 0.2},
            {"name": "runtime", "analyzer": "runtime_profile", "weight": 0.0, "enabled": False},
        ])
        self.assertEqual({m["name"] for m in metrics}, {"correctness", "syntax"})

    def test_invalid_metric_penalty_is_rejected_before_grading(self):
        with self.assertRaises(RulesError):
            _normalize_metrics([
                {"name": "correctness", "analyzer": "io_tests", "weight": 0.8},
                {"name": "syntax", "analyzer": "compiler_diagnostics", "weight": 0.2, "params": {"warning_penalty": -10}},
            ])

    def test_enabled_unknown_analyzer_cannot_receive_grade_weight(self):
        with self.assertRaises(RulesError):
            _normalize_metrics([
                {"name": "correctness", "analyzer": "io_tests", "weight": 0.8},
                {"name": "runtime", "analyzer": "runtime_profile", "weight": 0.2, "enabled": True},
            ])

    def test_instructor_cases_survive_a_malformed_initial_llm_response(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rules_data = {
                "problem": "Read one integer and print that same integer followed by a newline. This is a complete test assignment.",
                "examples": [{"id": "teacher", "input": "7\n", "expected": "7\n", "tags": ["core"]}],
                "test_generation": {"enabled": True, "rsi_passes": 3, "seed": 42, "temperature": 0.0, "max_cases": 8},
                "llm": {"start_server": False},
                "metrics": [
                    {"name": "correctness", "analyzer": "io_tests", "weight": 0.8},
                    {"name": "syntax", "analyzer": "compiler_diagnostics", "weight": 0.2},
                ],
            }
            rules_path = root / "rules.json"
            rules_path.write_text(json.dumps(rules_data))
            (root / "student.c").write_text("int main(void) { return 0; }\n")
            rules = load_rules(rules_path)
            llm = FakeLLM([LLMError("invalid RSI JSON")])
            suite = prepare_suite(rules, FakeDockerSandbox("fixture", 128, 1.0), llm)
            self.assertEqual(suite["test_count"], 1)
            self.assertEqual(suite["test_cases"][0]["origin"], "instructor_example")
            self.assertEqual(suite["reference_origin"], "instructor_expected_outputs")
            self.assertTrue(suite["generation_warnings"])

    def test_test_generation_retries_schema_failure_before_failing(self):
        rules = {
            "problem": "Read one integer and print it.",
            "problem_contract": {"input_format": "one integer", "output_format": "same integer"},
            "test_generation": {"max_cases": 8, "rsi_passes": 1, "seed": 42},
        }
        llm = FakeLLM([
            {"tests": [{"id": "bad", "input": "", "tags": ["core"]}]},
            {"inferred_contract": {"input_format": "one integer", "output_format": "same integer", "semantics": "identity"},
             "tests": [{"id": "good", "input": "17\n", "tags": ["core"], "rationale": "valid input"}]},
        ])
        cases, history, meta = _generate_test_inputs(llm, rules)
        self.assertEqual(llm.calls, 2)
        self.assertEqual(cases[0]["input"], "17\n")
        self.assertEqual(history[0]["status"], "completed")


    def test_test_generation_infers_io_contract_without_contract_fields(self):
        rules = {
            "problem": "Read two integers separated by whitespace and print their sum followed by a newline.",
            "problem_contract": {},
            "test_generation": {"max_cases": 8, "rsi_passes": 1, "seed": 42},
        }
        llm = FakeLLM([{
            "inferred_contract": {
                "input_format": "two whitespace-separated integers",
                "output_format": "one integer followed by newline",
                "constraints": ["signed 32-bit inputs"],
                "semantics": "integer addition",
                "formatting": "decimal integer and newline",
                "ambiguities": []
            },
            "tests": [
                {"id": "positive", "input": "2 3\\n", "tags": ["core"], "rationale": "ordinary positive values"}
            ],
            "review_summary": "Contract inferred from task statement."
        }])
        cases, history, meta = _generate_test_inputs(llm, rules)
        self.assertEqual(cases[0]["input"], "2 3\\n")
        self.assertEqual(meta["inferred_contract"]["output_format"], "one integer followed by newline")

    def test_semantic_compressor_covers_all_source_chunks_and_validates_line_anchors(self):
        from grader.grading import _semantic_compress_source
        class Compressor:
            max_tokens = 4096
            def __init__(self): self.calls = 0
            def json(self, system, user, max_tokens=None):
                payload = json.loads(user)
                self.calls += 1
                if "source_chunk_with_original_line_numbers" in payload:
                    lines = payload["source_chunk_with_original_line_numbers"].splitlines()
                    line_no = int(lines[0][:5])
                else:
                    line_no = payload["allowed_original_line_references"][0]
                return {
                    "functions": [{"name": "main", "lines": [line_no], "purpose": "entry point"}],
                    "control_flow": [], "data_flow": [], "io_behavior": [],
                    "memory_behavior": [], "invariants": [], "unresolved": []
                }
        llm = Compressor()
        source = "\n".join(f"int value_{i} = {i};" for i in range(900))
        result = _semantic_compress_source(llm, {"problem": "Analyze this C program."}, source, chunk_chars=1200)
        self.assertGreater(result["chunk_count"], 1)
        self.assertTrue(result["coverage_complete"])
        self.assertLessEqual(len(result["compressed_chunks"]), 4)
        self.assertGreater(result["merge_levels"], 0)
        self.assertTrue(all(
            item["line_range"][0] <= item["functions"][0]["lines"][0] <= item["line_range"][1]
            for item in result["compressed_chunks"]
        ))

    def test_test_generator_does_not_require_reference_in_each_rsi_response(self):
        cases = _clean_tests({"tests": [
            {"id": "normal", "input": "2 3\n", "tags": ["core"], "rationale": "normal input"},
            {"id": "boundary", "input": "0 0\n", "tags": ["boundary"], "rationale": "zero boundary"},
        ]}, max_cases=10, pass_number=1)
        self.assertEqual(len(cases), 2)
        self.assertIn("edge", cases[1]["tags"])
        self.assertNotIn("reference_solution_c", cases[0])

    def test_json_parser_handles_code_fences_and_braces_in_strings(self):
        result = _extract_json('```json\n{"message":"text with } brace", "score": 4}\n```')
        self.assertEqual(result["score"], 4)
        self.assertEqual(result["message"], "text with } brace")

    def test_llm_base_url_without_v1_is_normalised(self):
        from grader.llm import LLMClient
        client = LLMClient({"base_url": "http://127.0.0.1:1234", "health_url": "http://127.0.0.1:1234/health", "start_server": False})
        self.assertEqual(client.base_url, "http://127.0.0.1:1234/v1")
        self.assertEqual(client.health_url, "http://127.0.0.1:1234/health")

    def test_llm_environment_overrides_rules_config(self):
        from grader.llm import LLMClient
        with patch.dict("os.environ", {"LLAMA_MODEL": "fixture-model", "LLAMA_BASE_URL": "http://127.0.0.1:9999"}):
            client = LLMClient({"model": "old-model", "base_url": "http://127.0.0.1:8080/v1", "start_server": False})
        self.assertEqual(client.model, "fixture-model")
        self.assertEqual(client.base_url, "http://127.0.0.1:9999/v1")
        self.assertEqual(client.health_url, "http://127.0.0.1:9999/health")

    def test_reference_code_extractor(self):
        code = extract_c_code("Here is the answer:\n```c\n#include <stdio.h>\nint main(void){ puts(\"ok\"); return 0; }\n```")
        self.assertIn("int main", code)

    def test_output_comparison_modes_are_explicit(self):
        self.assertFalse(_compare("2.5\n", "2.50\n", "exact"))
        self.assertTrue(_compare(" 2.50\n", "2.50\t", "tokens"))
        self.assertTrue(_compare(" 2.5 \n", "2.5", "trimmed"))

    def test_comment_markers_inside_strings_do_not_hide_code(self):
        source = '#include <stdio.h>\nint main(void) { printf("https://example.test/a//b"); return 0; }\n'
        clean = strip_comments_literals(source)
        self.assertEqual(clean.count("\n"), source.count("\n"))
        self.assertIn("int main", clean)
        self.assertTrue(analyze_source(source)["has_main"])

    def test_average_program_is_one_valid_main_and_single_main_is_accepted(self):
        source = (
            "#include <stdio.h>\n"
            "int main(void) {\n"
            "  int n, value; long long sum = 0;\n"
            "  scanf(\"%d\", &n);\n"
            "  for (int i = 0; i < n; ++i) { scanf(\"%d\", &value); sum += value; }\n"
            "  printf(\"%.2f\\n\", sum / n);\n"
            "  return 0;\n"
            "}\n"
        )
        metrics = analyze_source(source)
        self.assertTrue(metrics["has_main"])
        self.assertEqual(metrics["function_count"], 1)
        result = score_structure(metrics, {"min_functions": 1, "max_cyclomatic": 12, "max_function_length": 80, "violation_penalty": 15})
        self.assertEqual(result["score"], 100.0)

    def test_correctness_runner_returns_counterexample_evidence(self):
        cases = [{"id": "fractional", "input": "4\n1 2 2 5\n", "expected": "2.50\n", "tags": ["core"]}]
        sandbox = FakeSandbox(["2.00\n"])
        result = _run_cases(sandbox, "student", cases, {"time_limit_s": 2, "output_comparison": "exact"})
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["results"][0]["expected"], "2.50\n")
        self.assertEqual(result["results"][0]["actual"], "2.00\n")

    def test_compile_failure_still_reports_counterexample_inputs(self):
        cases = [{"id": "case_1", "input": "5\n", "expected": "5\n", "tags": ["core"]}]
        result = _compile_failure_result(cases, metric_label="correctness")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["results"][0]["input"], "5\n")
        self.assertEqual(result["results"][0]["expected"], "5\n")
        self.assertIn("not executed", result["results"][0]["actual"])

    def test_syntax_rubric_is_fixed(self):
        no_warnings = _syntax_result({"exit_code": 0, "timed_out": False, "stderr": "", "stdout": ""}, {"warning_penalty": 5})
        two_warnings = _syntax_result({"exit_code": 0, "timed_out": False, "stderr": "a.c:1: warning: x\na.c:2: warning: y", "stdout": ""}, {"warning_penalty": 5})
        self.assertEqual(no_warnings["score"], 100)
        self.assertEqual(two_warnings["score"], 90)

    def test_nonzero_application_exit_is_not_alone_a_sanitizer_finding(self):
        from grader.grading import _memory_safety

        class SanitizerSandbox:
            def compile(self, *args, **kwargs):
                return {"exit_code": 0, "timed_out": False, "stdout": "", "stderr": ""}
            def lock_source(self, *args, **kwargs):
                pass
            def lock_binary(self, *args, **kwargs):
                pass
            def run_program(self, *args, **kwargs):
                return {"exit_code": 1, "timed_out": False, "output_truncated": False, "stdout": "", "stderr": ""}

        result = _memory_safety(SanitizerSandbox(), [{"id": "case", "input": ""}], {"time_limit_s": 1}, True)
        self.assertEqual(result["score"], 100.0)

    def test_missing_metric_withholds_overall_grade_instead_of_redistributing(self):
        metrics = [
            {"name": "correctness", "enabled": True, "weight": 0.5},
            {"name": "memory_safety", "enabled": True, "weight": 0.5},
        ]
        result = _calculate_grade(metrics, {
            "correctness": {"status": "completed", "score": 100, "rationale": "pass"},
            "memory_safety": {"status": "unavailable", "score": None, "rationale": "not run"},
        })
        self.assertIsNone(result["score"])
        self.assertEqual(result["status"], "incomplete")

    def test_overall_grade_rounds_after_summing_unrounded_contributions(self):
        metrics = [
            {"name": "a", "enabled": True, "weight": 0.3333333333333},
            {"name": "b", "enabled": True, "weight": 0.3333333333333},
            {"name": "c", "enabled": True, "weight": 0.3333333333334},
        ]
        results = {name: {"status": "completed", "score": 99.99} for name in ("a", "b", "c")}
        result = _calculate_grade(metrics, results)
        self.assertEqual(result["score"], 99.99)

    def test_semantic_compressor_handles_empty_source_without_looping(self):
        from grader.grading import _semantic_compress_source
        class EmptyCompressor:
            max_tokens = 1024
            def json(self, system, user, max_tokens=None):
                return {"functions": [], "control_flow": [], "data_flow": [], "io_behavior": [],
                        "memory_behavior": [], "invariants": [], "unresolved": []}
        result = _semantic_compress_source(EmptyCompressor(), {"problem": "An empty C source file."}, "")
        self.assertEqual(result["chunk_count"], 1)
        self.assertTrue(result["coverage_complete"])

    def test_score_rubric_has_explicit_criteria_for_all_bands(self):
        from grader.grading import SCORE_RUBRIC
        self.assertEqual([item["range"] for item in SCORE_RUBRIC], ["0-39", "40-59", "60-74", "75-89", "90-100"])
        self.assertTrue(all(len(item["criteria"]) > 30 for item in SCORE_RUBRIC))

    def test_score_bands(self):
        self.assertTrue(_score_band(65).startswith("60-74"))
        self.assertTrue(_score_band(100).startswith("90-100"))

    def test_rsi_rejects_unanchored_hallucinated_line_findings(self):
        source = "#include <stdio.h>\nint main(void) {\n  int x = 1;\n  printf(\"%d\\n\", x);\n  return 0;\n}\n"
        anchored = {"title": "Unused variable", "description": "The variable is unused.", "severity": "low", "metric": "syntax", "line_start": 3, "evidence_excerpt": "int x = 1;"}
        hallucinated = {"title": "Integer division", "description": "It truncates.", "severity": "high", "metric": "correctness", "line_start": 3, "evidence_excerpt": "double average = sum / n;"}
        llm = FakeLLM([
            {"assessment": "One supported concern.", "findings": [anchored, hallucinated]},
            {"assessment": "Rechecked.", "findings": [anchored]},
            {"assessment": "Final recheck.", "findings": [anchored]},
        ])
        result = run_rsi_review(llm, {"problem": "A sufficiently detailed C task specification used for the review.", "problem_contract": {}}, source, {})
        self.assertEqual(llm.calls, 3)
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(result["findings"][0]["title"], "Unused variable")

    def test_complexity_analysis_requires_big_o_and_real_source_line_evidence(self):
        from grader.grading import _normalise_complexity
        source = ["#include <stdio.h>", "int main(void) {", "  for (int i = 0; i < n; ++i) total += i;", "  return 0;", "}"]
        valid = {
            "algorithm": "Single loop accumulation",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "explanation": "The loop executes n times and uses a fixed number of scalar variables.",
            "confidence": "high",
            "evidence": [{"line": 3, "claim": "The loop bound depends on n."}],
        }
        result = _normalise_complexity(valid, source)
        self.assertEqual(result["time_complexity"], "O(n)")
        self.assertEqual(result["space_complexity"], "O(1)")
        self.assertIsNone(_normalise_complexity({**valid, "evidence": [{"line": 99, "claim": "fake"}]}, source))
        self.assertIsNone(_normalise_complexity({**valid, "time_complexity": "linear"}, source))

    def test_rsi_passes_carry_forward_prior_understanding_and_complexity(self):
        source = "int main(void) {\n  for (int i = 0; i < 10; ++i) {}\n  return 0;\n}\n"
        complexity = {"algorithm": "loop", "time_complexity": "O(n)", "space_complexity": "O(1)", "explanation": "One loop.", "confidence": "high", "evidence": [{"line": 2, "claim": "Loop runs n times."}]}
        class SequentialLLM:
            max_tokens = 4096
            def __init__(self): self.users = []
            def json(self, system, user, max_tokens=None):
                payload = json.loads(user)
                if "source_chunk_with_original_line_numbers" in payload:
                    return {"functions": [], "control_flow": [], "data_flow": [], "io_behavior": [],
                            "memory_behavior": [], "invariants": [], "unresolved": []}
                self.users.append(payload)
                return {"assessment": f"understanding pass {payload['pass']}", "findings": [], "complexity": complexity, "changes_from_previous": []}
        llm = SequentialLLM()
        result = run_rsi_review(llm, {"problem": "Analyze a loop program.", "problem_contract": {}}, source, {})
        self.assertEqual(len(llm.users), 3)
        self.assertEqual(llm.users[1]["previous_assessment"], "understanding pass 1")
        self.assertEqual(llm.users[2]["previous_assessment"], "understanding pass 2")
        self.assertEqual(llm.users[2]["previous_complexity_analysis"]["time_complexity"], "O(n)")
        self.assertEqual(result["complexity_analysis"]["time_complexity"], "O(n)")
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["context_compression"]["source_excerpt_truncated"])
        self.assertEqual(result["context_compression"]["method"], "hierarchical_llm_semantic_compression_with_line_anchor_validation")
        self.assertTrue(result["context_compression"]["coverage_complete"])

    def test_end_to_end_grade_is_stable_and_uses_frozen_suite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "submission.c").write_text("#include <stdio.h>\nint main(void) { puts(\"WRONG\"); return 0; }\n")
            rules = {
                "problem": "Read one input token and print the exact uppercase token followed by a newline. This assignment is used for an integration regression test.",
                "submission": {"source_file": "submission.c"},
                "time_limit_s": 2,
                "memory_limit_mb": 128,
                "docker_image": "fixture",
                "problem_contract": {"input_format": "one token", "output_format": "same token uppercased", "constraints": ["one token"], "required_behavior": ["convert to uppercase"], "numeric_semantics": "not applicable", "error_handling": "valid input only"},
                "examples": [{"id": "sample", "input": "d\n", "expected": "REF\n", "tags": ["core"]}],
                "test_generation": {"enabled": True, "rsi_passes": 3, "seed": 42, "temperature": 0.0, "max_cases": 10, "cache_dir": ".grader_cache"},
                "llm": {"start_server": False},
                "metrics": [
                    {"name": "correctness", "analyzer": "io_tests", "weight": 0.5},
                    {"name": "syntax", "analyzer": "compiler_diagnostics", "weight": 0.15},
                    {"name": "structure", "analyzer": "static_analysis", "weight": 0.1},
                    {"name": "memory_safety", "analyzer": "memory_safety", "weight": 0.2},
                    {"name": "readability", "analyzer": "readability", "weight": 0.05},
                ],
            }
            (root / "rules.json").write_text(json.dumps(rules))
            first_llm = FakeFullLLM()
            with patch("grader.grading.DockerSandbox", FakeDockerSandbox), patch("grader.grading.LLMClient", return_value=first_llm):
                report1, _, _ = grade(root / "rules.json")
            self.assertEqual(report1["scoring"]["status"], "completed")
            self.assertEqual(report1["scoring"]["score"], 50.0)
            self.assertEqual(report1["metrics"]["correctness"]["score"], 0.0)
            self.assertEqual(report1["test_suite"]["test_count"], 4)
            inputs = [case["input"] for case in report1["test_suite"]["test_cases"]]
            self.assertEqual(len(inputs), len(set(inputs)), "RSI passes must not duplicate input cases")
            self.assertEqual(first_llm.text_calls, 1)
            second_llm = FakeFullLLM()
            with patch("grader.grading.DockerSandbox", FakeDockerSandbox), patch("grader.grading.LLMClient", return_value=second_llm):
                report2, _, _ = grade(root / "rules.json")
            self.assertEqual(report1["scoring"]["score"], report2["scoring"]["score"])
            self.assertEqual(report1["test_suite"]["suite_sha256"], report2["test_suite"]["suite_sha256"])
            self.assertEqual(second_llm.text_calls, 0)
            self.assertTrue((root / ".grader_cache").is_dir())


if __name__ == "__main__":
    unittest.main()
