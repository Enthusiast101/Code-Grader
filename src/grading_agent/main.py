
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


def load_rules(path: Path) -> dict:
    """Load the instructor's grading rules from a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        rules = json.load(file)

    if not isinstance(rules, dict):
        raise ValueError("Rules must be a JSON object.")

    if not isinstance(rules.get("metrics"), list) or not rules["metrics"]:
        raise ValueError("Rules must contain a non-empty metrics list.")

    return rules


def main():
    # Open the instructor's rules file.
    rules_path = PROJECT_ROOT / "rules.json"
    rules = load_rules(rules_path)

    # Get the first grading metric from the rules.
    metric = rules["metrics"][0]

    if len(rules["metrics"]) != 1:
        raise NotImplementedError(
            "Only one metric is currently supported."
        )

    if not isinstance(metric, dict):
        raise ValueError("Each metric must be a JSON object.")

    weight = metric.get("weight")

    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not 0 <= weight <= 1
    ):
        raise ValueError("Metric weight must be between 0 and 1.")

    if abs(weight - 1.0) > 1e-9:
        raise ValueError(
            "The single supported metric must have weight 1.0."
        )


    # Read the test-file location from the rules.
    testset_path = metric["params"]["testset_path"]

    # Locate the source code and test cases.
    source_path = PROJECT_ROOT / "examples" / "sum" / "main.c"
    tests_path = PROJECT_ROOT / testset_path

    source_code = source_path.read_text(encoding="utf-8")
    test_cases = load_test_cases(tests_path)


    try:
        with DockerSandbox(
            memory_limit=f"{rules['memory_limit_mb']}m"
        ) as sandbox:
            logger.info("Docker sandbox started.")

            report = run_io_tests(
                sandbox=sandbox,
                source_code=source_code,
                test_cases=test_cases,
                compile_timeout_s=metric["params"].get("compile_timeout_s", 10),
                run_timeout_s=rules["time_limit_s"],
            )

            #scoring code that will assign scores
            metric_score = report.get("score", 0.0)
            contribution = round(metric_score * weight * 100, 2)

            report["problem"] = rules["problem"]
            report["metrics"] = [
                {
                    "name": metric.get("name", "correctness"),
                    "analyzer": metric["analyzer"],
                    "weight": weight,
                    "score": metric_score,
                    "contribution": contribution,
                }
            ]
            report["overall_grade"] = contribution

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
