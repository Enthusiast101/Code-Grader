import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from grading_agent.config import settings
from grading_agent.ui import app as ui_app


def _generated_suite_state(tmp_path, *, source_code, raw_rules, metric_names, count=60, focus=""):
    generated_dir = tmp_path / ".generated_tests"
    generated_dir.mkdir(exist_ok=True)
    all_path = generated_dir / "all.json"
    edge_path = generated_dir / "edges.json"
    suite_path = generated_dir / "suite.json"
    cases = [
        {"id": "instructor-1", "input": "1\n", "expected": "1\n", "origin": "instructor"},
        {"id": "generated-1", "input": "2\n", "expected": "2\n", "origin": "llm_generated_reference_oracle", "tags": ["boundary"]},
    ]
    all_path.write_text(json.dumps(cases), encoding="utf-8")
    edge_path.write_text(json.dumps(cases), encoding="utf-8")
    suite_path.write_text(json.dumps({"test_cases": cases}), encoding="utf-8")
    return {
        "status": "completed",
        "source_sha256": ui_app._sha256_text(source_code),
        "rules_sha256": ui_app._sha256_text(raw_rules),
        "requested_generated_test_count": count,
        "additional_test_focus": focus,
        "assignment_fingerprint": "fingerprint",
        "suite_path": str(suite_path),
        "metric_testset_paths": {
            metric_name: str(edge_path if metric_name == "logical" else all_path)
            for metric_name in metric_names
        },
        "instructor_test_count": 1,
        "generated_test_count": count,
        "total_test_count": count + 1,
        "edge_test_count": 2,
        "reference_solution_source": "configured",
        "oracle_kind": "instructor-reference",
        "oracle_caveat": "Trusted reference solution used.",
    }


def test_runtime_overrides_are_scoped_and_thread_isolated():
    original = settings.runtime_value("LLM_GLOBAL_RSI_MAX_TOKENS")

    def read_with_override(value):
        with settings.runtime_overrides({"LLM_GLOBAL_RSI_MAX_TOKENS": value}):
            return settings.runtime_value("LLM_GLOBAL_RSI_MAX_TOKENS")

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(read_with_override, [333, 777]))

    assert values == [333, 777]
    assert settings.runtime_value("LLM_GLOBAL_RSI_MAX_TOKENS") == original


def test_gradio_generated_case_slider_value_is_written_to_staged_rules(tmp_path, monkeypatch):
    source_path = tmp_path / "student.c"
    source_path.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    rules_path = tmp_path / "rules.json"
    rules_payload = {
        "problem": "Read an integer and print it.",
        "time_limit_s": 2,
        "memory_limit_mb": 256,
        "submission": {"language": "c", "source_file": "placeholder.c"},
        "metrics": [
            {"name": "correctness", "analyzer": "io_tests", "enabled": True,
             "weight": 1.0, "params": {"testset_path": "cases.json"}},
        ],
        "test_generation": {"enabled": True, "max_cases": 4},
    }
    rules_path.write_text(json.dumps(rules_payload), encoding="utf-8")
    raw_rules = rules_path.read_text(encoding="utf-8")
    source_code = source_path.read_text(encoding="utf-8")
    (tmp_path / "cases.json").write_text(
        json.dumps([{"input": "1\n", "expected": "1\n"}]), encoding="utf-8"
    )

    monkeypatch.setattr(ui_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(settings, "GRADIO_STAGING_DIR", tmp_path / ".gradio_runs")
    monkeypatch.setattr(settings, "GRADIO_REPORTS_DIR", tmp_path / "reports" / "gradio")
    generated_state = _generated_suite_state(
        tmp_path, source_code=source_code, raw_rules=raw_rules, metric_names=["correctness"],
        count=12, focus="boundary inputs",
    )
    captured = {}

    def fake_grade(staged_rules, *, project_root, report_dir, runtime_options):
        captured["rules"] = json.loads(Path(staged_rules).read_text(encoding="utf-8"))
        report = {
            "problem": "Read an integer and print it.",
            "configuration": {"enabled_metrics": []},
            "scoring": {"score": 80.0, "status": "completed", "warnings": []},
            "llm_rsi_evaluation": {"status": "completed"},
        }
        return report, tmp_path / "report.json", tmp_path / "report.txt"

    monkeypatch.setattr(ui_app, "grade_with_rules_file", fake_grade)
    ui_app.grade_uploaded_submission(
        source_upload=None,
        source_code_editor=source_code,
        source_editor_hash="",
        rules_upload=str(rules_path),
        rules_json_text="",
        time_limit_s=2,
        memory_limit_mb=256,
        rsi_iterations=3,
        understanding_tokens=512,
        rsi_tokens=896,
        scoring_tokens=256,
        max_findings=12,
        source_chars=12000,
        evidence_chars=4200,
        temperature=0.0,
        model_name="test-model",
        read_timeout_s=180,
        docker_cpu_limit=1.0,
        enable_thinking=False,
        run_sanitizers=False,
        max_generated_cases=12,
        additional_test_focus="boundary inputs",
        generated_suite_state=generated_state,
    )

    assert captured["rules"]["test_generation"]["enabled"] is True
    assert captured["rules"]["test_generation"]["max_cases"] == 12
    assert captured["rules"]["test_generation"]["additional_focus"] == "boundary inputs"
    assert captured["rules"]["test_generation"]["pre_generated_suite"]["status"] == "completed"
    assert captured["rules"]["metrics"][0]["params"]["testset_path"].endswith(".generated_tests/all.json")


def test_grade_button_requires_generated_suite_before_grading(tmp_path, monkeypatch):
    source_path = tmp_path / "submission.c"
    source_path.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({
        "problem": "Read an integer and print it.",
        "time_limit_s": 2,
        "memory_limit_mb": 256,
        "submission": {"language": "c", "source_file": "submission.c"},
        "metrics": [{
            "name": "correctness", "analyzer": "io_tests", "enabled": True,
            "weight": 1.0, "params": {},
        }],
        "test_generation": {"enabled": True, "max_cases": 12},
    }), encoding="utf-8")

    monkeypatch.setattr(ui_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(settings, "GRADIO_STAGING_DIR", tmp_path / ".gradio_runs")
    monkeypatch.setattr(settings, "GRADIO_REPORTS_DIR", tmp_path / "reports" / "gradio")
    monkeypatch.setattr(
        ui_app, "grade_with_rules_file",
        lambda *_args, **_kwargs: pytest.fail("grading must not start before generation"),
    )

    result = ui_app.grade_uploaded_submission(
        source_upload=str(source_path), source_code_editor="", source_editor_hash="",
        rules_upload=str(rules_path), rules_json_text="", time_limit_s=2,
        memory_limit_mb=256, rsi_iterations=3, understanding_tokens=512,
        rsi_tokens=896, scoring_tokens=256, max_findings=12,
        source_chars=12000, evidence_chars=4200, temperature=0.0,
        model_name="test-model", read_timeout_s=180, docker_cpu_limit=1.0,
        enable_thinking=False, run_sanitizers=False,
        generated_suite_state=None,
    )

    assert "Generate test cases first" in result[0]


def test_rules_upload_populates_editor_and_limits(tmp_path):
    rules_path = tmp_path / "rules.json"
    payload = {
        "time_limit_s": 17,
        "memory_limit_mb": 1024,
        "submission": {"language": "c"},
    }
    rules_path.write_text(json.dumps(payload), encoding="utf-8")

    content, time_limit, memory_limit, status, source_hash = ui_app.load_rules_editor_from_upload(str(rules_path))

    assert json.loads(content) == payload
    assert time_limit == 17.0
    assert memory_limit == 1024
    assert "Loaded rules.json" in status
    assert source_hash == ui_app._sha256_text(content)


def test_legacy_rules_upload_migrates_old_weights_and_removes_runtime_metric(tmp_path, monkeypatch):
    names_and_caps = [
        ("correctness", 20), ("syntax", 10), ("structure", 10),
        ("modularity", 10), ("complexity", 10), ("logical", 10),
        ("memory_safety", 10), ("readability", 10), ("originality", 10),
    ]
    canonical_metrics = [
        {"name": name, "analyzer": "source", "weight": cap / 100, "max_points": cap,
         "enabled": True, "params": {}}
        for name, cap in names_and_caps
    ]
    (tmp_path / "rules.json").write_text(json.dumps({"metrics": canonical_metrics}), encoding="utf-8")
    old_metrics = [
        {"name": name, "analyzer": "source", "weight": 0.1, "enabled": True, "params": {}}
        for name, _ in names_and_caps[:-1]
    ]
    old_metrics.append({
        "name": "runtime_efficiency", "analyzer": "runtime_profile", "weight": 0.05,
        "enabled": True, "params": {"testset_path": None},
    })
    old_rules_path = tmp_path / "old_rules.json"
    old_text = json.dumps({
        "problem": "task", "submission": {"language": "c", "source_file": "submission.c"},
        "metrics": old_metrics, "rubric": {"metrics": {"runtime_efficiency": "old rubric"}},
    })
    old_rules_path.write_text(old_text, encoding="utf-8")
    monkeypatch.setattr(ui_app, "PROJECT_ROOT", tmp_path)

    editor_text, _, _, status, source_hash = ui_app.load_rules_editor_from_upload(str(old_rules_path))
    migrated = json.loads(editor_text)

    assert [metric["name"] for metric in migrated["metrics"]] == [name for name, _ in names_and_caps]
    assert [metric["max_points"] for metric in migrated["metrics"]] == [cap for _, cap in names_and_caps]
    assert sum(metric["weight"] for metric in migrated["metrics"]) == 1.0
    assert "rubric" not in migrated
    assert "runtime_efficiency removed" in status
    assert source_hash == ui_app._sha256_text(old_text)


def test_gradio_grading_applies_run_controls_without_mutating_uploaded_rules(tmp_path, monkeypatch):
    source_path = tmp_path / "student.c"
    source_path.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    rules_path = tmp_path / "rules.json"
    original_rules = {
        "problem": "sample",
        "time_limit_s": 5,
        "memory_limit_mb": 512,
        "submission": {"language": "c", "source_file": "placeholder.c"},
        "metrics": [
            {
                "name": "memory_safety",
                "analyzer": "memory_safety",
                "enabled": True,
                "weight": 1.0,
                "params": {"testset_path": "tests/cases.json"},
            }
        ],
    }
    original_text = json.dumps(original_rules)
    rules_path.write_text(original_text, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "cases.json").write_text(
        json.dumps([{"input": "1\n", "expected": "1\n"}]), encoding="utf-8"
    )
    source_code = source_path.read_text(encoding="utf-8")
    generated_state = _generated_suite_state(
        tmp_path, source_code=source_code, raw_rules=original_text, metric_names=["memory_safety"],
    )

    monkeypatch.setattr(ui_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(settings, "GRADIO_STAGING_DIR", tmp_path / ".gradio_runs")
    monkeypatch.setattr(settings, "GRADIO_REPORTS_DIR", tmp_path / "reports" / "gradio")
    captured = {}

    def fake_grade(staged_rules, *, project_root, report_dir, runtime_options):
        staged = json.loads(Path(staged_rules).read_text(encoding="utf-8"))
        from grading_agent.config.rule_loader import load_rules
        normalized = load_rules(Path(staged_rules), project_root)
        captured["rules"] = staged
        captured["normalized_rules"] = normalized
        captured["runtime_options"] = runtime_options
        report = {
            "problem": "sample",
            "configuration": {"enabled_metrics": []},
            "scoring": {"score": 88.0, "status": "completed", "warnings": []},
            "llm_rsi_evaluation": {"status": "completed"},
        }
        return report, tmp_path / "report.json", tmp_path / "report.txt"

    monkeypatch.setattr(ui_app, "grade_with_rules_file", fake_grade)
    result = ui_app.grade_uploaded_submission(
        source_upload=str(source_path), source_code_editor="", source_editor_hash="",
        rules_upload=str(rules_path), rules_json_text="", time_limit_s=9,
        memory_limit_mb=1536, rsi_iterations=2, understanding_tokens=640,
        rsi_tokens=1024, scoring_tokens=320, max_findings=12,
        source_chars=12000, evidence_chars=4200, temperature=0.15,
        model_name="test-local-model", read_timeout_s=240, docker_cpu_limit=3.0,
        enable_thinking=False, run_sanitizers=False,
        generated_suite_state=generated_state,
    )

    assert result[0].startswith("## Grading result")
    assert captured["rules"]["time_limit_s"] == 9.0
    assert captured["rules"]["memory_limit_mb"] == 1536
    assert captured["rules"]["metrics"][0]["params"]["run_sanitizers"] is False
    assert captured["normalized_rules"]["metrics"][0]["params"]["run_sanitizers"] is False
    assert captured["runtime_options"]["LLM_REVIEW_MAX_ITERATIONS"] == 2
    assert captured["runtime_options"]["LLM_GLOBAL_RSI_MAX_TOKENS"] == 1024
    assert captured["runtime_options"]["LLM_GLOBAL_SOURCE_MAX_CHARS"] == 12000
    assert captured["runtime_options"]["LLM_MAX_EVIDENCE_CHARS"] == 4200
    assert captured["runtime_options"]["DOCKER_CPU_LIMIT"] == 3.0
    assert rules_path.read_text(encoding="utf-8") == original_text


def test_new_upload_wins_over_stale_editor_state(tmp_path, monkeypatch):
    source_path = tmp_path / "student.c"
    source_path.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    old_rules = {
        "problem": "Read two integers and print their sum.",
        "time_limit_s": 5, "memory_limit_mb": 512,
        "submission": {"language": "c", "source_file": "placeholder.c"},
        "metrics": [{"name": "memory_safety", "analyzer": "memory_safety", "enabled": True, "weight": 1.0, "params": {}}],
    }
    new_rules = {**old_rules, "problem": "Read a sequence and print it in ascending order."}
    old_path = tmp_path / "old_rules.json"
    new_path = tmp_path / "new_rules.json"
    old_text = json.dumps(old_rules)
    new_text = json.dumps(new_rules)
    old_path.write_text(old_text, encoding="utf-8")
    new_path.write_text(new_text, encoding="utf-8")
    _, _, _, _, old_editor_base_hash = ui_app.load_rules_editor_from_upload(str(old_path))

    monkeypatch.setattr(ui_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(settings, "GRADIO_STAGING_DIR", tmp_path / ".gradio_runs")
    monkeypatch.setattr(settings, "GRADIO_REPORTS_DIR", tmp_path / "reports" / "gradio")
    generated_state = _generated_suite_state(
        tmp_path, source_code=source_path.read_text(encoding="utf-8"),
        raw_rules=new_text, metric_names=["memory_safety"],
    )
    captured = {}

    def fake_grade(staged_rules, *, project_root, report_dir, runtime_options):
        staged = json.loads(Path(staged_rules).read_text(encoding="utf-8"))
        captured["rules"] = staged
        report = {"problem": staged["problem"], "configuration": {"enabled_metrics": []}, "scoring": {"score": 88, "status": "completed", "warnings": []}, "llm_rsi_evaluation": {"status": "completed"}}
        return report, tmp_path / "report.json", tmp_path / "report.txt"

    monkeypatch.setattr(ui_app, "grade_with_rules_file", fake_grade)
    result = ui_app.grade_uploaded_submission(
        source_upload=str(source_path), source_code_editor="", source_editor_hash="",
        rules_upload=str(new_path), rules_json_text=old_text, time_limit_s=5,
        memory_limit_mb=512, rsi_iterations=3, understanding_tokens=512,
        rsi_tokens=896, scoring_tokens=256, max_findings=12,
        source_chars=12000, evidence_chars=4200, temperature=0.0,
        model_name="test-model", read_timeout_s=180, docker_cpu_limit=1.0,
        enable_thinking=False, run_sanitizers=False,
        rules_editor_source_hash=old_editor_base_hash,
        generated_suite_state=generated_state,
    )

    assert captured["rules"]["problem"] == new_rules["problem"]
    assert captured["rules"]["_run_metadata"]["selection"]["editor_was_stale"] is True
    assert captured["rules"]["_run_metadata"]["selection"]["selected_from"] == "uploaded rules file"
    assert "ascending order" in result[0]


def test_editor_edits_are_used_when_based_on_current_upload(tmp_path, monkeypatch):
    source_path = tmp_path / "student.c"
    source_path.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    payload = {
        "problem": "Read two integers and print their sum.",
        "time_limit_s": 5, "memory_limit_mb": 512,
        "submission": {"language": "c", "source_file": "placeholder.c"},
        "metrics": [{"name": "memory_safety", "analyzer": "memory_safety", "enabled": True, "weight": 1.0, "params": {}}],
    }
    rules_path = tmp_path / "rules.json"
    uploaded_text = json.dumps(payload)
    rules_path.write_text(uploaded_text, encoding="utf-8")
    _, _, _, _, base_hash = ui_app.load_rules_editor_from_upload(str(rules_path))
    edited = {**payload, "problem": "Sort a sequence of integers."}
    edited_text = json.dumps(edited)

    monkeypatch.setattr(ui_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(settings, "GRADIO_STAGING_DIR", tmp_path / ".gradio_runs")
    monkeypatch.setattr(settings, "GRADIO_REPORTS_DIR", tmp_path / "reports" / "gradio")
    generated_state = _generated_suite_state(
        tmp_path, source_code=source_path.read_text(encoding="utf-8"),
        raw_rules=edited_text, metric_names=["memory_safety"],
    )
    captured = {}

    def fake_grade(staged_rules, *, project_root, report_dir, runtime_options):
        staged = json.loads(Path(staged_rules).read_text(encoding="utf-8"))
        captured["rules"] = staged
        report = {"problem": staged["problem"], "configuration": {"enabled_metrics": []}, "scoring": {"score": 88, "status": "completed", "warnings": []}, "llm_rsi_evaluation": {"status": "completed"}}
        return report, tmp_path / "report.json", tmp_path / "report.txt"

    monkeypatch.setattr(ui_app, "grade_with_rules_file", fake_grade)
    ui_app.grade_uploaded_submission(
        source_upload=str(source_path), source_code_editor="", source_editor_hash="",
        rules_upload=str(rules_path), rules_json_text=edited_text, time_limit_s=5,
        memory_limit_mb=512, rsi_iterations=3, understanding_tokens=512,
        rsi_tokens=896, scoring_tokens=256, max_findings=12,
        source_chars=12000, evidence_chars=4200, temperature=0.0,
        model_name="test-model", read_timeout_s=180, docker_cpu_limit=1.0,
        enable_thinking=False, run_sanitizers=False,
        rules_editor_source_hash=base_hash,
        generated_suite_state=generated_state,
    )

    assert captured["rules"]["problem"] == edited["problem"]
    assert captured["rules"]["_run_metadata"]["selection"]["editor_was_stale"] is False
    assert captured["rules"]["_run_metadata"]["selection"]["selected_from"] == "edited Gradio rules editor"


def test_llm_request_uses_runtime_model_temperature_and_timeout(monkeypatch):
    from grading_agent.llm import reviewer

    captured = {}

    class FakeResponse:
        ok = True
        reason = "OK"
        text = ""

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    def fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["body"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(reviewer.requests, "post", fake_post)
    with settings.runtime_overrides({
        "MODEL": "custom-local-model",
        "TEMPERATURE": 0.25,
        "LLAMA_REQUEST_TIMEOUT": (4.0, 321.0),
    }):
        parsed, _ = reviewer._request_json(
            [{"role": "user", "content": "return json"}],
            max_tokens=123,
            enable_thinking=False,
        )

    assert parsed == {"ok": True}
    assert captured["body"]["model"] == "custom-local-model"
    assert captured["body"]["temperature"] == 0.25
    assert captured["body"]["max_tokens"] == 123
    assert captured["timeout"] == (4.0, 321.0)


def test_gradio_app_builds_with_parameter_controls():
    app = ui_app.build_app()
    assert app is not None
    assert len(app.blocks) >= 50


def test_llama_server_startup_uses_selected_model_and_temperature(monkeypatch):
    from types import SimpleNamespace
    from grading_agent.llm import server

    captured = {}
    fake_process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(server.shutil, "which", lambda _name: "/usr/bin/llama-server")
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="version: 0.0.0 (build 12005, commit 3c809f44)", stderr=""),
    )
    def fake_popen(command):
        captured["command"] = command
        return fake_process
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(server.requests, "get", lambda *_args, **_kwargs: SimpleNamespace(status_code=200))

    with settings.runtime_overrides({"MODEL": "custom-startup-model", "TEMPERATURE": 0.3}):
        process = server.initiate_server()

    assert process is fake_process
    assert captured["command"][captured["command"].index("-hf") + 1] == "custom-startup-model"
    assert captured["command"][captured["command"].index("--temp") + 1] == "0.3"


def test_ui_summary_last_resort_formatter_keeps_grade_visible(monkeypatch):
    monkeypatch.setattr(ui_app, "_metric_summary_impl", lambda _report: (_ for _ in ()).throw(RuntimeError("formatter bug")))
    summary = ui_app._metric_summary({"scoring": {"score": 73.5, "status": "completed_with_warnings"}})
    assert "Overall grade: 73.50/100" in summary
    assert "REPORT" not in summary
    assert "formatter bug" in summary


def test_summary_discloses_exact_assignment_and_rules_source():
    report = {
        "problem": "Sort an integer sequence in ascending order.",
        "rules_provenance": {
            "rules_sha256": "a" * 64,
            "selection": {
                "selected_from": "uploaded rules file",
                "selected_rules_sha256": "a" * 64,
                "editor_was_stale": True,
            },
        },
        "configuration": {"enabled_metrics": []},
        "scoring": {"score": 91.5, "status": "completed", "warnings": []},
        "llm_rsi_evaluation": {"status": "completed"},
        "llm_rsi_findings": [],
    }
    summary = ui_app._metric_summary(report)
    assert "Sort an integer sequence in ascending order" in summary
    assert "uploaded rules file" in summary
    assert "changed after the editor was populated" in summary


def test_generate_button_analyzes_submission_then_saves_cases_without_grading(tmp_path, monkeypatch):
    source_code = (
        '#include <stdio.h>\n'
        'int main(void) { int a, b; scanf("%d%d", &a, &b); '
        'printf("%d\\n", a + b); return 0; }\n'
    )
    source_path = tmp_path / "submission.c"
    source_path.write_text(source_code, encoding="utf-8")
    reference_path = tmp_path / "reference.c"
    reference_path.write_text(
        '#include <stdio.h>\n'
        'int main(void) { int a, b; if (scanf("%d%d", &a, &b) != 2) return 1; '
        'printf("%d\\n", a + b); return 0; }\n',
        encoding="utf-8",
    )
    instructor_path = tmp_path / "instructor_cases.json"
    instructor_path.write_text(
        json.dumps([{"id": "teacher-1", "input": "1 2\n", "expected": "3\n"}]),
        encoding="utf-8",
    )
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({
        "problem": "Read two signed integers and print their sum.",
        "problem_contract": {
            "input_format": "Two signed integers separated by whitespace.",
            "output_rule": "Print their arithmetic sum followed by a newline.",
        },
        "time_limit_s": 2,
        "memory_limit_mb": 256,
        "submission": {"language": "c", "source_file": "submission.c"},
        "metrics": [{
            "name": "correctness", "analyzer": "io_tests", "enabled": True,
            "weight": 1.0, "max_points": 100, "params": {"testset_path": "instructor_cases.json"},
        }],
        "test_generation": {
            "enabled": True, "max_cases": 4, "rsi_passes": 1,
            "seed": settings.SEED, "reference_solution_path": "reference.c",
        },
    }), encoding="utf-8")

    monkeypatch.setattr(ui_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(settings, "GRADIO_STAGING_DIR", tmp_path / ".gradio_staging")
    monkeypatch.setattr(ui_app, "_generation_server_is_ready", lambda: True)
    monkeypatch.setattr(
        ui_app, "grade_with_rules_file",
        lambda *_args, **_kwargs: pytest.fail("Generate Test Cases must not start grading"),
    )

    class FakeDockerSandbox:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write_file(self, _filename, _content):
            pass

        def run(self, command, timeout_s, stdin=None):
            if command[0] == "gcc":
                return {"output": "", "exit_code": 0, "timed_out": False}
            left, right = (int(value) for value in stdin.split())
            return {
                "output": f"{left + right}\n", "exit_code": 0,
                "timed_out": False, "output_may_be_truncated": False,
            }

    import sys
    import types
    sandbox_module = types.ModuleType("grading_agent.sandbox.docker_sandbox")
    sandbox_module.DockerSandbox = FakeDockerSandbox
    monkeypatch.setitem(sys.modules, "grading_agent.sandbox.docker_sandbox", sandbox_module)

    from grading_agent.llm import reviewer

    analysis = {
        "implementation_summary": "The code parses two integers and adds them using int.",
        "input_parsing_and_validation": ["The scanf result is not checked."],
        "control_flow_and_data_types": ["Two signed int values are added."],
        "likely_weak_points": ["Addition may overflow signed int."],
        "recommended_test_dimensions": ["Large magnitudes and mixed signs."],
    }
    stages = []
    def fake_request_json(messages, *, max_tokens, enable_thinking, response_schema, schema_name):
        stages.append(schema_name)
        payload = json.loads(messages[1]["content"])
        if schema_name.startswith("assignment_submission_understanding"):
            assert payload["submission_source"] == source_code
            assert "test_inputs" not in response_schema["properties"]
            return analysis, json.dumps(analysis)

        assert payload["submitted_implementation_analysis"] == analysis
        assert schema_name == "assignment_test_generation_single_pass"
        requested = response_schema["properties"]["test_inputs"]["maxItems"]
        return ({
            "ambiguities": [],
            "test_inputs": [
                {
                    "id": f"case-{index}",
                    "input": f"{index + 10} {index + 1}\n",
                    "tags": ["boundary"],
                    "rationale": "Distinct generated input targeting integer addition.",
                }
                for index in range(requested)
            ]
        }, "{}")

    monkeypatch.setattr(reviewer, "_request_json", fake_request_json)

    status, generated_state, preview_rows, grade_button_update = ui_app.generate_test_cases_for_submission(
        None, source_code, "", str(rules_path), "", "",
        2, 256, 1.0, 0.0, "test-model", 60,
        2, "large magnitudes and mixed signs", False,
    )

    assert "Test suite ready" in status
    assert "Test-design notes from the submitted C code" in status
    assert "not grading findings" in status
    assert "Addition may overflow signed int." in status
    assert generated_state["status"] == "completed"
    assert generated_state["requested_generated_test_count"] == 2
    assert generated_state["generated_test_count"] == 2
    assert generated_state["instructor_test_count"] == 1
    assert generated_state["total_test_count"] == 3
    assert len(preview_rows) == 3
    assert preview_rows[0][1] == "Instructor"
    assert sum(row[1] == "LLM-generated" for row in preview_rows) == 2
    assert stages[0] == "assignment_submission_understanding_single_pass"
    assert stages[1:] == ["assignment_test_generation_single_pass"]
    assert Path(generated_state["suite_path"]).is_file()
    assert grade_button_update["interactive"] is True
