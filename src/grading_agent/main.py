import math
import json
import logging
from pathlib import Path

from grading_agent.analyzers.io_tests import run_io_tests
from grading_agent.analyzers.static_analysis import run_static_analysis
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

#changing load_rules for better grading
def load_rules(path: Path) -> dict:
    """Load and validate the instructor's grading rules."""
    with path.open("r", encoding="utf-8") as file:
        rules = json.load(file)

    if not isinstance(rules, dict):
        raise ValueError("Rules must be a JSON object.")

    if not isinstance(rules.get("problem"), str) or not rules["problem"].strip():
        raise ValueError("Rules must contain a non-empty problem name.")

    for key in ("time_limit_s", "memory_limit_mb"):
        value = rules.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    metrics = rules.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("Rules must contain a non-empty metrics list.")

    weights = []

    for index, metric in enumerate(metrics, start=1):
        if not isinstance(metric, dict):
            raise ValueError(f"Metric {index} must be a JSON object.")

        if not isinstance(metric.get("name"), str) or not metric["name"].strip():
            raise ValueError(f"Metric {index} must have a non-empty name.")

        if not isinstance(metric.get("analyzer"), str) or not metric["analyzer"].strip():
            raise ValueError(f"Metric {index} must have an analyzer name.")

        weight = metric.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or not 0 <= weight <= 1
        ):
            raise ValueError(
                f"Metric {index} weight must be between 0 and 1."
            )

        if not isinstance(metric.get("params"), dict):
            raise ValueError(f"Metric {index} params must be a JSON object.")

        weights.append(weight)

    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Metric weights must add up to 1.0.")

    return rules


def main():
    # 1. Load the instructor's grading rules.
    rules_path = PROJECT_ROOT / "rules.json"
    rules = load_rules(rules_path)

    # The source file is currently fixed to the sum example.
    source_path = PROJECT_ROOT / "examples" / "sum" / "main.c"
    source_code = source_path.read_text(encoding="utf-8")

    # Store the results from every configured metric.
    metric_results = []

    try:
        # 2. Start one isolated Docker sandbox.
        with DockerSandbox(
            memory_limit=f"{rules['memory_limit_mb']}m"
        ) as sandbox:
            logger.info("Docker sandbox started.")

            # 3. Process every metric in rules.json.
            for metric in rules["metrics"]:
                analyzer = metric["analyzer"]
                params = metric["params"]

                if analyzer == "io_tests":
                    testset_path = params.get("testset_path")

                    if not isinstance(testset_path, str) or not testset_path:
                        raise ValueError(
                            f"Metric {metric['name']!r} requires "
                            "params.testset_path."
                        )

                    test_cases = load_test_cases(
                        PROJECT_ROOT / testset_path
                    )

                    details = run_io_tests(
                        sandbox=sandbox,
                        source_code=source_code,
                        test_cases=test_cases,
                        compile_timeout_s=params.get("compile_timeout_s", 10),
                        run_timeout_s=rules["time_limit_s"],
                    )

                elif analyzer == "static_analysis":
                    details = run_static_analysis(
                        sandbox=sandbox,
                        source_code=source_code,
                        timeout_s=params.get("timeout_s", 10),
                        warning_penalty=params.get("warning_penalty", 0.1),
                    )

                else:
                    raise ValueError(
                        f"Unsupported analyzer: {analyzer!r}. "
                        "Implement this analyzer before using it."
                    )

                # 5. Validate the score returned by the analyzer.
                score = details.get("score", 0.0)

                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                ):
                    raise ValueError(
                        f"Analyzer {analyzer!r} returned "
                        "an invalid score."
                    )

                weight = metric["weight"]

                # Convert this metric's score into grade points.
                contribution = round(score * weight * 100, 2)

                metric_results.append({
                    "name": metric["name"],
                    "analyzer": analyzer,
                    "weight": weight,
                    "score": score,
                    "contribution": contribution,
                    "details": details,
                })

            # 6. Calculate the overall grade using all metrics.
            overall_grade = round(
                sum(
                    result["score"] * result["weight"] * 100
                    for result in metric_results
                ),
                2,
            )

            # 7. Build the final report.
            report = {}

            # Preserve the existing top-level correctness fields
            # when exactly one metric is configured.
            if len(metric_results) == 1:
                report.update(metric_results[0]["details"])

            report.update({
                "problem": rules["problem"],
                "metrics": metric_results,
                "overall_grade": overall_grade,
            })

            print(json.dumps(report, indent=2))

            # 8. Report outcomes appropriately for each analyzer.
            for result in metric_results:
                details = result["details"]
                analyzer = result["analyzer"]

                if not details.get("compiled", False):
                    logger.error(
                        "Analysis or compilation failed for metric %s.",
                        result["name"],
                    )

                elif analyzer == "io_tests":
                    if (
                        details["tests_passed"]
                        == details["tests_total"]
                    ):
                        logger.info(
                            "All tests passed for metric %s.",
                            result["name"],
                        )
                    else:
                        logger.warning(
                            "%s/%s tests passed for metric %s.",
                            details["tests_passed"],
                            details["tests_total"],
                            result["name"],
                        )

                elif analyzer == "static_analysis":
                    logger.info(
                        "Static analysis completed for metric %s: "
                        "%s warning(s).",
                        result["name"],
                        details["warning_count"],
                    )

    except Exception:
        logger.exception("Grading failed.")
        raise

    logger.info("Docker sandbox stopped and cleaned up.")

if __name__ == "__main__":
    main()
