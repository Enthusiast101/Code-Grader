import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from grading_agent.config import settings
from grading_agent.ui import app as ui_app


def test_runtime_overrides_are_scoped_and_thread_isolated():
    original = settings.runtime_value("LLM_GLOBAL_RSI_MAX_TOKENS")

    def read_with_override(value):
        with settings.runtime_overrides({"LLM_GLOBAL_RSI_MAX_TOKENS": value}):
            return settings.runtime_value("LLM_GLOBAL_RSI_MAX_TOKENS")

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(read_with_override, [333, 777]))

    assert values == [333, 777]
    assert settings.runtime_value("LLM_GLOBAL_RSI_MAX_TOKENS") == original


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
        str(source_path),
        str(rules_path),
        "",
        9,
        1536,
        2,
        640,
        1024,
        320,
        12,
        1,
        12000,
        4200,
        0.15,
        "test-local-model",
        240,
        3.0,
        False,
        False,
    )

    assert result[0].startswith("## Final grading result")
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
    captured = {}

    def fake_grade(staged_rules, *, project_root, report_dir, runtime_options):
        staged = json.loads(Path(staged_rules).read_text(encoding="utf-8"))
        captured["rules"] = staged
        report = {"problem": staged["problem"], "configuration": {"enabled_metrics": []}, "scoring": {"score": 88, "status": "completed", "warnings": []}, "llm_rsi_evaluation": {"status": "completed"}}
        return report, tmp_path / "report.json", tmp_path / "report.txt"

    monkeypatch.setattr(ui_app, "grade_with_rules_file", fake_grade)
    result = ui_app.grade_uploaded_submission(
        str(source_path), str(new_path), old_text, 5, 512, 3, 512, 896, 256,
        12, 2, 12000, 4200, 0.0, "test-model", 180, 1.0, False, False,
        True, "http://127.0.0.1:11434", "qwen2.5:3b", 1800, old_editor_base_hash,
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

    monkeypatch.setattr(ui_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(settings, "GRADIO_STAGING_DIR", tmp_path / ".gradio_runs")
    monkeypatch.setattr(settings, "GRADIO_REPORTS_DIR", tmp_path / "reports" / "gradio")
    captured = {}

    def fake_grade(staged_rules, *, project_root, report_dir, runtime_options):
        staged = json.loads(Path(staged_rules).read_text(encoding="utf-8"))
        captured["rules"] = staged
        report = {"problem": staged["problem"], "configuration": {"enabled_metrics": []}, "scoring": {"score": 88, "status": "completed", "warnings": []}, "llm_rsi_evaluation": {"status": "completed"}}
        return report, tmp_path / "report.json", tmp_path / "report.txt"

    monkeypatch.setattr(ui_app, "grade_with_rules_file", fake_grade)
    ui_app.grade_uploaded_submission(
        str(source_path), str(rules_path), json.dumps(edited), 5, 512, 3, 512, 896, 256,
        12, 2, 12000, 4200, 0.0, "test-model", 180, 1.0, False, False,
        True, "http://127.0.0.1:11434", "qwen2.5:3b", 1800, base_hash,
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
