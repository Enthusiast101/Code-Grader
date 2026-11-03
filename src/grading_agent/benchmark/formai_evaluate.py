"""Evaluate FormAI-v2 vulnerability predictions against held-out labels.

The benchmark labels are consumed only after prediction files have been
created. Ground truth is program-level: a file is vulnerable when FormAI has
at least one confirmed vulnerable record. Multiple vulnerable locations and
error types are retained for localization/type evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from grading_agent.config import settings



def _iter_ground_truth(path: Path) -> Iterable[dict[str, Any]]:
    """Read either JSON-array or JSONL ground-truth files."""

    with path.open("r", encoding="utf-8") as stream:
        first_non_whitespace = ""

        while True:
            char = stream.read(1)

            if not char or not char.isspace():
                first_non_whitespace = char
                break

        stream.seek(0)

        # Existing FormAI-v2 format: a JSON array.
        if first_non_whitespace == "[":
            try:
                import ijson
            except ImportError:
                data = json.load(stream)

                if not isinstance(data, list):
                    raise ValueError(
                        "Ground-truth JSON must contain a top-level array."
                    )

                for item in data:
                    if isinstance(item, dict):
                        yield item

            else:
                with path.open("rb") as binary_stream:
                    yield from ijson.items(binary_stream, "item")

        # New FormAI-v1 format: one JSON object per line.
        else:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid ground-truth JSONL at line "
                        f"{line_number}: {exc}"
                    ) from exc

                if not isinstance(item, dict):
                    raise ValueError(
                        f"Ground-truth line {line_number} "
                        "must contain a JSON object."
                    )

                yield item


def _is_vulnerable(category: Any) -> bool | None:
    if not isinstance(category, str):
        return None
    normalized = category.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"vulnerable", "unsafe", "positive", "true"}:
        return True
    if normalized in {"safe", "safe_within_bound", "non_vulnerable", "not_vulnerable", "clean", "negative", "false"}:
        return False
    return None


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prediction JSONL at line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Prediction at line {line_number} must be a JSON object.")
            name = item.get("file_name") or item.get("filename")
            if not isinstance(name, str) or not name:
                raise ValueError(f"Prediction at line {line_number} has no file_name.")
            if name in predictions:
                raise ValueError(f"Duplicate prediction for file_name {name!r} in {path}.")
            predictions[name] = item
    return predictions


def _prediction_is_inconclusive(
    prediction: dict[str, Any]
) -> bool:
    """Identify predictions that cannot safely be scored as binary."""

    if prediction.get("llm_unresolved_positive_summary") is True:
        return True

    if prediction.get("llm_unresolved_invalid_candidates") is True:
        return True

    status = prediction.get("prediction_status")

    if isinstance(status, str) and status.casefold() in {
        "failed",
        "error",
        "inconclusive",
        "skipped",
        "timeout"
    }:
        return True

    if prediction.get("llm_error"):
        return True

    return False


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, int) and value > 0:
        return [value]
    if isinstance(value, list):
        return sorted({item for item in value if isinstance(item, int) and item > 0})
    return []


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return sorted({item.strip() for item in value if isinstance(item, str) and item.strip()})
    return []


def _normalize_type(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _calculate_binary_metrics(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> dict[str, Any]:
    """Calculate binary classification metrics.

    Undefined metrics are represented as None rather than silently
    assigning zero when their denominator is zero.
    """

    total = tp + fp + fn + tn
    positive_support = tp + fn
    negative_support = tn + fp

    def ratio(numerator: float, denominator: float):
        if denominator == 0:
            return None
        return round(numerator / denominator, 6)

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)

    accuracy = ratio(tp + tn, total)

    positive_f1 = ratio(
        2 * tp,
        2 * tp + fp + fn
    )

    negative_f1 = ratio(
        2 * tn,
        2 * tn + fp + fn
    )

    # Macro-F1 is undefined if either class has undefined F1.
    macro_f1 = (
        round((positive_f1 + negative_f1) / 2, 6)
        if positive_f1 is not None and negative_f1 is not None
        else None
    )

    # Weighted-F1 uses the actual number of examples in each class.
    weighted_f1 = None

    if total > 0:
        weighted_sum = 0.0

        if positive_support:
            weighted_sum += positive_f1 * positive_support

        if negative_support:
            weighted_sum += negative_f1 * negative_support

        weighted_f1 = round(weighted_sum / total, 6)

    balanced_accuracy = (
        round((recall + specificity) / 2, 6)
        if recall is not None and specificity is not None
        else None
    )

    mcc_denominator = (
        (tp + fp) *
        (tp + fn) *
        (tn + fp) *
        (tn + fn)
    )

    if mcc_denominator:
        import math

        mcc = round(
            (tp * tn - fp * fn) /
            math.sqrt(mcc_denominator),
            6
        )
    else:
        mcc = None

    return {
        "sample_count": total,
        "class_support": {
            "vulnerable": positive_support,
            "bounded_negative": negative_support
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": positive_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {
            "vulnerable": positive_f1,
            "bounded_negative": negative_f1
        },
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "false_positive_rate": ratio(fp, fp + tn),
        "false_negative_rate": ratio(fn, fn + tp),
        "zero_denominator_policy": (
            "Undefined metrics are reported as null; "
            "they are never silently replaced with zero."
        )
    }


def evaluate_mode(
    ground_truth_path: Path,
    predictions_path: Path,
    *,
    line_tolerance: int = settings.FORM_AI_LINE_TOLERANCE,
) -> dict[str, Any]:
    """Evaluate only records with actual boolean predictions.

    Missing and inconclusive predictions are never silently converted to safe.
    A complete benchmark exposes headline metrics; incomplete runs expose
    observed-subset metrics separately and set headline metrics to null.
    """
    predictions = _load_predictions(predictions_path)
    tp = fp = fn = tn = 0
    line_checked = line_exact = line_within = 0
    type_checked = type_correct = 0
    evaluated = missing = skipped = inconclusive = matched = 0
    ground_truth_names: set[str] = set()
    valid_ground_truth = 0

    for record in _iter_ground_truth(ground_truth_path):
        filename = record.get("file_name")
        # The derived program-level category is authoritative over raw record fields.
        truth = _is_vulnerable(record.get("ground_truth_category", record.get("category")))
        if not isinstance(filename, str) or not filename or truth is None:
            skipped += 1
            continue
        if filename in ground_truth_names:
            raise ValueError(f"Duplicate ground-truth filename {filename!r}; aggregate labels to one row per program first.")
        ground_truth_names.add(filename)
        valid_ground_truth += 1

        vulnerable_lines = _as_int_list(record.get("vulnerable_lines"))
        if not vulnerable_lines:
            vulnerable_lines = _as_int_list(record.get("vulnerable_line"))
        error_types = _as_string_list(record.get("error_types"))
        if not error_types:
            error_types = _as_string_list(record.get("error_type"))

        prediction = predictions.get(filename)
        if prediction is None:
            missing += 1
            continue
        matched += 1
        if _prediction_is_inconclusive(prediction):
            inconclusive += 1
            continue
        raw_prediction = prediction.get("predicted_vulnerable")
        if isinstance(raw_prediction, bool):
            predicted = raw_prediction
        elif isinstance(raw_prediction, str) and raw_prediction.casefold() in {"vulnerable", "unsafe", "true", "positive"}:
            predicted = True
        elif isinstance(raw_prediction, str) and raw_prediction.casefold() in {"safe", "non_vulnerable", "false", "negative"}:
            predicted = False
        else:
            inconclusive += 1
            continue

        evaluated += 1
        if truth and predicted:
            tp += 1
            predicted_line = prediction.get("predicted_line")
            if vulnerable_lines and isinstance(predicted_line, int) and not isinstance(predicted_line, bool):
                line_checked += 1
                line_exact += int(predicted_line in vulnerable_lines)
                line_within += int(any(abs(predicted_line - line) <= line_tolerance for line in vulnerable_lines))
            predicted_type = prediction.get("predicted_error_type")
            if error_types and isinstance(predicted_type, str):
                type_checked += 1
                normalized_prediction = _normalize_type(predicted_type)
                type_correct += int(any(_normalize_type(kind) == normalized_prediction for kind in error_types))
        elif not truth and predicted:
            fp += 1
        elif truth and not predicted:
            fn += 1
        else:
            tn += 1

    # Calculate metrics after processing every program.
    extra_predictions = sorted(set(predictions) - ground_truth_names)

    partial_metrics = _calculate_binary_metrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn
    )

    complete = (
        valid_ground_truth > 0
        and missing == 0
        and inconclusive == 0
        and skipped == 0
        and not extra_predictions
        and evaluated == valid_ground_truth
    )
    localization = {
        "line_predictions_evaluated": line_checked,
        "exact_line_accuracy": round(line_exact / line_checked, 6) if line_checked else None,
        "within_tolerance_accuracy": round(line_within / line_checked, 6) if line_checked else None,
        "line_tolerance": line_tolerance,
        "note": "A predicted line is correct when it matches any confirmed vulnerable line for that source file.",
    }
    error_type_accuracy = round(type_correct / type_checked, 6) if type_checked else None
    return {
        "prediction_file": str(predictions_path),
        "evaluation_status": "completed" if complete else "incomplete",
        "metrics_finalized": complete,
        "ground_truth_records": valid_ground_truth,
        "evaluated_records": evaluated,
        "matched_prediction_records": matched,
        "missing_predictions": missing,
        "extra_predictions_not_in_ground_truth": len(extra_predictions),
        "extra_prediction_examples": extra_predictions[:10],
        "inconclusive_predictions": inconclusive,
        "skipped_records": skipped,
        "prediction_coverage": round(matched / valid_ground_truth, 6) if valid_ground_truth else None,
        "classification_coverage": round(evaluated / valid_ground_truth, 6) if valid_ground_truth else None,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        # Headline metrics are withheld for an incomplete test-set run.
        "precision": partial_metrics["precision"] if complete else None,
        "recall": partial_metrics["recall"] if complete else None,
        "f1_score": partial_metrics["f1_score"] if complete else None,
        "accuracy": partial_metrics["accuracy"] if complete else None,
        "specificity": partial_metrics["specificity"] if complete else None,
        "false_positive_rate": partial_metrics["false_positive_rate"] if complete else None,
        "false_negative_rate": partial_metrics["false_negative_rate"] if complete else None,
        "balanced_accuracy": partial_metrics["balanced_accuracy"] if complete else None,
        "macro_f1": partial_metrics["macro_f1"] if complete else None,
        "weighted_f1": partial_metrics["weighted_f1"] if complete else None,
        "mcc": partial_metrics["mcc"] if complete else None,
        "per_class_f1": partial_metrics["per_class_f1"] if complete else None,
        "observed_subset_metrics": partial_metrics,
        "localization": localization,
        "error_type_accuracy": error_type_accuracy,
        "error_type_predictions_evaluated": type_checked,
        "note": (
            "Complete metrics use every valid ground-truth program exactly once. "
            "Missing and inconclusive predictions are not treated as safe. "
            "For incomplete runs, observed_subset_metrics are diagnostic only; headline metrics are withheld."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path, help="Program-level FormAI labels JSON.")
    parser.add_argument(
    "--dataset-name",
    default="FormAI-v2",
    help="Dataset name for the evaluation report."
)
    parser.add_argument("--predictions", required=True, action="append", metavar="MODE=JSONL", help="Repeat, e.g. tools_only=tools.jsonl")
    parser.add_argument("--line-tolerance", type=int, default=settings.FORM_AI_LINE_TOLERANCE)
    parser.add_argument("--output", type=Path, default=Path(settings.FORM_AI_EVALUATION_OUTPUT))
    args = parser.parse_args()

    if args.line_tolerance < 0:
        parser.error("--line-tolerance cannot be negative")

    modes: dict[str, Any] = {}
    for item in args.predictions:
        if "=" not in item:
            parser.error("Each --predictions value must be MODE=JSONL")
        name, filename = item.split("=", 1)
        if name in modes:
            parser.error(f"Duplicate prediction mode: {name}")
        modes[name] = evaluate_mode(args.ground_truth, Path(filename), line_tolerance=args.line_tolerance)

    result = {
        "dataset": args.dataset_name,
        "ground_truth_file": str(args.ground_truth),
        "modes": modes,
        "comparison_note": "Compare tools-only, LLM-only, hybrid, and hybrid+RSI on the same held-out program split. Keep labels out of prediction-time runs and do not tune on the final test split.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
