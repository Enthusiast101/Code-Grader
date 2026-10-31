"""Run instructor-provided edge and corner cases as a separate metric."""
from __future__ import annotations
from typing import Any
from grading_agent.analyzers.io_tests import run_io_tests
from grading_agent.sandbox.docker_sandbox import DockerSandbox
from grading_agent.config import settings


def run_edge_case_runner(
    sandbox: DockerSandbox,
    source_code: str,
    test_cases: list[dict[str, Any]],
    filename: str = settings.DEFAULT_SOURCE_FILENAME,
    compile_timeout_s: int = settings.ANALYZER_COMPILE_TIMEOUT_S,
    run_timeout_s: float = settings.ANALYZER_EDGE_RUN_TIMEOUT_S,
) -> dict[str, Any]:
    result = run_io_tests(
        sandbox=sandbox,
        source_code=source_code,
        test_cases=test_cases,
        filename=filename,
        compile_timeout_s=compile_timeout_s,
        run_timeout_s=run_timeout_s,
    )
    result["analyzer"] = "edge_case_runner"
    result["score_basis"] = "fraction of configured edge/corner cases passed"
    return result
