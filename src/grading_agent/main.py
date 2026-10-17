
import json
import logging
from pathlib import Path

from grading_agent.analyzers.io_tests import run_io_tests
from grading_agent.sandbox.docker_sandbox import DockerSandbox


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root, regardless of the directory used to launch this file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_test_cases(path: Path) -> list[dict[str, str]]:
    """Load and validate externally defined input/output tests."""
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("Test cases must be a non-empty JSON array.")

    for index, case in enumerate(cases, start=1):
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("input"), str)
            or not isinstance(case.get("expected"), str)
        ):
            raise ValueError(
                f"Test case {index} must contain string fields "
                "'input' and 'expected'."
            )

    return cases


def main():
    source_path = PROJECT_ROOT / "examples" / "sum" / "main.c"
    tests_path = PROJECT_ROOT / "tests" / "sum" / "testcases.json"

    source_code = source_path.read_text(encoding="utf-8")
    test_cases = load_test_cases(tests_path)

    try:
        with DockerSandbox() as sandbox:
            logger.info("Docker sandbox started.")

            report = run_io_tests(
                sandbox=sandbox,
                source_code=source_code,
                test_cases=test_cases,
            )

            print(json.dumps(report, indent=2))

            if not report["compiled"]:
                logger.error("Compilation failed.")
            elif report["tests_passed"] == report["tests_total"]:
                logger.info("All tests passed.")
            else:
                logger.warning(
                    "%s/%s tests passed.",
                    report["tests_passed"],
                    report["tests_total"],
                )

    except Exception:
        logger.exception("Grading failed.")
        raise

    logger.info("Docker sandbox stopped and cleaned up.")


if __name__ == "__main__":
    main()
