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


def _iter_ground_truth(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import ijson  # type: ignore
    except ImportError:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, list):
            raise ValueError("FormAI ground-truth JSON must be a top-level array.")
        yield from (item for item in data if isinstance(item, dict))
        return

    with path.open("rb") as stream:
        yield from ijson.items(stream, "item")


def _is_vulnerable(category: Any) -> bool | None:
    if not isinstance(category, str):
        return None
    normalized = category.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"vulnerable", "unsafe", "positive", "true"}:
        return True
    if normalized in {"safe", "non_vulnerable", "not_vulnerable", "clean", "negative", "false"}:
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


def evaluate_mode(
    ground_truth_path: Path,
    predictions_path: Path,
    *,
    line_tolerance: int = 0,
) -> dict[str, Any]:
    predictions = _load_predictions(predictions_path)
    tp = fp = fn = tn = 0
    line_checked = line_exact = line_within = 0
    type_checked = type_correct = 0
    evaluated = missing = skipped = inconclusive = 0
    matched_predictions: set[str] = set()

    for record in _iter_ground_truth(ground_truth_path):
        filename = record.get("file_name")
        truth = _is_vulnerable(record.get("category", record.get("ground_truth_category")))
        if not isinstance(filename, str) or truth is None:
            skipped += 1
            continue
        vulnerable_lines = _as_int_list(record.get("vulnerable_lines"))
        if not vulnerable_lines:
            vulnerable_lines = _as_int_list(record.get("vulnerable_line"))
        error_types = _as_string_list(record.get("error_types"))
        if not error_types:
            error_types = _as_string_list(record.get("error_type"))

        prediction = predictions.get(filename)
        if prediction is None:
            missing += 1
            predicted = False
            prediction = {}
        else:
            matched_predictions.add(filename)
            raw_prediction = prediction.get("predicted_vulnerable")
            if isinstance(raw_prediction, bool):
                predicted = raw_prediction
            elif isinstance(raw_prediction, str):
                predicted = raw_prediction.casefold() in {"vulnerable", "unsafe", "true", "positive"}
            else:
                skipped += 1
                inconclusive += 1
                continue
        evaluated += 1

        if truth and predicted:
            tp += 1
            predicted_line = prediction.get("predicted_line")
            if vulnerable_lines and isinstance(predicted_line, int):
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

    extra_predictions = sorted(set(predictions) - matched_predictions)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "prediction_file": str(predictions_path),
        "evaluated_records": evaluated,
        "missing_predictions": missing,
        "extra_predictions_not_in_ground_truth": len(extra_predictions),
        "extra_prediction_examples": extra_predictions[:10],
        "inconclusive_predictions": inconclusive,
        "skipped_records": skipped,
        "prediction_coverage": round(evaluated / (evaluated + inconclusive), 6) if evaluated + inconclusive else None,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "accuracy": round((tp + tn) / evaluated, 6) if evaluated else None,
        "specificity": round(specificity, 6),
        "false_positive_rate": round(fp / (fp + tn), 6) if fp + tn else 0.0,
        "false_negative_rate": round(fn / (fn + tp), 6) if fn + tp else 0.0,
        "balanced_accuracy": round((recall + specificity) / 2, 6),
        "localization": {
            "line_predictions_evaluated": line_checked,
            "exact_line_accuracy": round(line_exact / line_checked, 6) if line_checked else None,
            "within_tolerance_accuracy": round(line_within / line_checked, 6) if line_checked else None,
            "line_tolerance": line_tolerance,
            "note": "A predicted line is correct when it matches any confirmed vulnerable line for that source file.",
        },
        "error_type_accuracy": round(type_correct / type_checked, 6) if type_checked else None,
        "error_type_predictions_evaluated": type_checked,
        "note": "One record per source program. Missing predictions count as negative predictions; inconclusive predictions are counted separately. Localization/type metrics use all aggregated ground-truth locations/types for each vulnerable program.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path, help="Program-level FormAI labels JSON.")
    parser.add_argument("--predictions", required=True, action="append", metavar="MODE=JSONL", help="Repeat, e.g. tools_only=tools.jsonl")
    parser.add_argument("--line-tolerance", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("formai-evaluation.json"))
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
        "dataset": "FormAI-v2",
        "ground_truth_file": str(args.ground_truth),
        "modes": modes,
        "comparison_note": "Compare tools-only, LLM-only, hybrid, and hybrid+RSI on the same held-out program split. Keep labels out of prediction-time runs and do not tune on the final test split.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
