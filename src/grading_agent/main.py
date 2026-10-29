import json
import logging
from pathlib import Path

from grading_agent.analyzers.io_tests import run_io_tests
from grading_agent.analyzers.memory_safety import run_memory_safety
from grading_agent.llm.reviewer import review_evidence
from grading_agent.sandbox.docker_sandbox import DockerSandbox


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    source_path = PROJECT_ROOT / "examples" / "sum" / "main.c"
    tests_path = PROJECT_ROOT / "tests" / "sum" / "testcases.json"
    report_path = PROJECT_ROOT / "reports" / "sum.json"

    source_code = source_path.read_text(encoding="utf-8")

    with tests_path.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    try:
        with DockerSandbox() as sandbox:
            logger.info("Docker sandbox started.")

            io_report = run_io_tests(
                sandbox=sandbox,
                source_code=source_code,
                test_cases=test_cases,
            )

            memory_report = run_memory_safety(
                sandbox=sandbox,
                source_code=source_code,
                test_cases=test_cases,
            )

            report = {
                "submission": source_path.name,
                "io_tests": io_report,
                "memory_safety": memory_report,
            }

        # Run LLM review after the sandbox has been cleaned up.
        try:
            report["llm_review"] = review_evidence(report)
            logger.info("LLM evidence review completed.")
        except Exception as exc:
            # Do not discard deterministic analyzer results if the
            # model server is down or its response is invalid.
            logger.warning("LLM review unavailable: %s", exc)
            report["llm_review"] = {
                "status": "unavailable",
                "error": str(exc),
            }

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info("Report saved to %s", report_path)

    except Exception:
        logger.exception("Grading failed.")
        raise


if __name__ == "__main__":
    main()
