#!/usr/bin/env python3
"""Wire automatic assignment-level test generation into the existing pipeline.

Run from the repository root after copying the testing package. Generation is
performed inside the already-created Docker sandbox and before preflight or
student analyzers. A generation failure clears execution test paths so scoring
fails closed rather than silently grading with an incomplete suite.
"""
from pathlib import Path

path = Path("src/grading_agent/main.py")
s = path.read_text(encoding="utf-8")

def replace_once(old: str, new: str, label: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"Refusing automatic-test integration ({label}): expected 1 match, got {count}.")
    s = s.replace(old, new, 1)

replace_once(
'''from grading_agent.preflight import run_preflight
''',
'''from grading_agent.preflight import run_preflight
from grading_agent.testing.test_generation import TestGenerationError, prepare_assignment_test_suite
''', "imports")
replace_once(
'''EXECUTION_ANALYZERS = {"io_tests", "edge_case_runner", "runtime_profile", "memory_safety"}


def _unavailable_metric(error: str) -> dict[str, Any]:
''',
'''EXECUTION_ANALYZERS = {"io_tests", "edge_case_runner", "runtime_profile", "memory_safety"}


def _prepare_assignment_tests(sandbox: Any, rules: dict[str, Any]) -> dict[str, Any]:
    """Generate/load the frozen assignment suite before student analyzers run."""
    from grading_agent.llm import reviewer

    process = None
    def request_json(messages, response_schema, stage_name):
        nonlocal process
        if not _is_llm_server_ready():
            process = initiate_server()
        parsed, raw = reviewer._request_json(
            messages,
            max_tokens=max(
                2048,
                int(settings.runtime_value("LLM_PROGRAM_UNDERSTANDING_MAX_TOKENS")),
                int(settings.runtime_value("LLM_GLOBAL_RSI_MAX_TOKENS")),
            ),
            enable_thinking=False,
            response_schema=response_schema,
            schema_name=stage_name,
        )
        if not isinstance(parsed, dict):
            raise TestGenerationError(
                "llm_generation_invalid",
                f"{stage_name} returned no valid structured response: {str(raw)[:800]}",
            )
        return parsed

    try:
        return prepare_assignment_test_suite(
            rules,
            sandbox=sandbox,
            request_json=request_json,
            project_root=Path(rules.get("project_root", PROJECT_ROOT)),
            model_id=str(settings.runtime_value("MODEL")),
            generation_seed=int(settings.SEED),
            temperature=float(settings.runtime_value("TEMPERATURE")),
        )
    finally:
        _stop_llm_server(process)


def _clear_execution_test_paths(rules: dict[str, Any]) -> None:
    """Prevent incomplete test generation from being mistaken for measured tests."""
    for metric in rules.get("metrics", []):
        if metric.get("enabled", True) and metric.get("analyzer") in EXECUTION_ANALYZERS:
            metric.setdefault("params", {}).pop("resolved_testset_path", None)
            metric.pop("resolved_testset_path", None)


def _unavailable_metric(error: str) -> dict[str, Any]:
''', "generation helpers")
# Insert after Docker container startup, before the preflight try block.
replace_once(
'''            sandbox = sandbox_manager.__enter__()
            report["sandbox"] = {"status": "available", "image": sandbox_manager.image}
        except Exception as exc:
''',
'''            sandbox = sandbox_manager.__enter__()
            report["sandbox"] = {"status": "available", "image": sandbox_manager.image}
            if any(metric.get("enabled", True) and metric.get("analyzer") in EXECUTION_ANALYZERS for metric in rules.get("metrics", [])):
                try:
                    report["test_generation"] = _prepare_assignment_tests(sandbox, rules)
                except Exception as generation_exc:
                    _clear_execution_test_paths(rules)
                    report["test_generation"] = {
                        "status": "failed",
                        "error_code": getattr(generation_exc, "code", type(generation_exc).__name__),
                        "error": f"{type(generation_exc).__name__}: {generation_exc}",
                    }
                    logger.error("Assignment test generation failed; execution-based metrics will be inconclusive: %s", generation_exc)
        except Exception as exc:
''', "run generator before preflight")
# If Docker itself is missing, expose why tests could not be generated.
replace_once(
'''            report["preflight"] = {
                "status": "unavailable", "compiled": None,
                "smoke_tested": False,
                "error": sandbox_error or "Docker sandbox unavailable.",
            }
''',
'''            report["preflight"] = {
                "status": "unavailable", "compiled": None,
                "smoke_tested": False,
                "error": sandbox_error or "Docker sandbox unavailable.",
            }
            if any(metric.get("enabled", True) and metric.get("analyzer") in EXECUTION_ANALYZERS for metric in rules.get("metrics", [])):
                report["test_generation"] = {"status": "failed", "error_code": "sandbox_unavailable", "error": sandbox_error or "Docker sandbox unavailable."}
''', "sandbox unavailable generation status")
# Surface test suite provenance in the report configuration and make its artifact hash auditable.
replace_once(
'''    report["analyzer_pipeline_elapsed_seconds"] = round(
''',
'''    if isinstance(report.get("test_generation"), dict) and report["test_generation"].get("status") == "completed":
        generated_meta = report["test_generation"]
        generated_suite_path = generated_meta.get("suite_path")
        if generated_suite_path and Path(generated_suite_path).is_file():
            generated_meta["suite_sha256"] = hashlib.sha256(Path(generated_suite_path).read_bytes()).hexdigest()

    report["analyzer_pipeline_elapsed_seconds"] = round(
''', "test artifact fingerprint")
path.write_text(s, encoding="utf-8")
print("Integrated automatic RSI test generation before preflight and analyzers.")
