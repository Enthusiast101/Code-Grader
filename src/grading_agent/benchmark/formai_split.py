"""Create deterministic stratified FormAI splits with labels kept separate."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from grading_agent.config import settings


def _read_source_inputs(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            filename = item.get("file_name")
            source = item.get("source_code")
            if not isinstance(filename, str) or not isinstance(source, str):
                raise ValueError(f"Invalid source JSONL record at line {line_number}.")
            if filename in records:
                raise ValueError(f"Duplicate FormAI filename in source inputs: {filename}")
            records[filename] = item
    return records


def _read_labels(path: Path) -> dict[str, dict[str, Any]]:
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, list):
        raise ValueError("Ground-truth labels must be a JSON array.")
    by_name: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(labels, 1):
        if not isinstance(item, dict) or not isinstance(item.get("file_name"), str):
            raise ValueError(f"Invalid label record at index {index}.")
        filename = item["file_name"]
        if filename in by_name:
            raise ValueError(f"Duplicate FormAI filename in labels: {filename}")
        category = str(item.get("ground_truth_category", item.get("category", ""))).upper()
        if category not in {"VULNERABLE", "SAFE"}:
            raise ValueError(f"Unsupported ground-truth category for {filename}: {category!r}")
        by_name[filename] = {**item, "ground_truth_category": category}
    return by_name


def _split_names(
    names: list[str],
    *,
    validation_fraction: float,
    test_fraction: float,
    rng: random.Random,
) -> dict[str, list[str]]:
    shuffled = list(names)
    rng.shuffle(shuffled)
    count = len(shuffled)
    validation_count = int(count * validation_fraction)
    test_count = int(count * test_fraction)

    # For non-trivial strata, maintain at least one example in each requested
    # split while retaining at least one training example.
    if count >= 3 and validation_fraction > 0:
        validation_count = max(1, validation_count)
    if count >= 3 and test_fraction > 0:
        test_count = max(1, test_count)
    while validation_count + test_count >= count and count > 1:
        if test_count >= validation_count and test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
        else:
            break

    return {
        "train": shuffled[: count - validation_count - test_count],
        "validation": shuffled[count - validation_count - test_count : count - test_count],
        "test": shuffled[count - test_count :],
    }


def split_formai(
    source_path: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    validation_fraction: float = settings.FORM_AI_SPLIT_VALIDATION_FRACTION,
    test_fraction: float = settings.FORM_AI_SPLIT_TEST_FRACTION,
    seed: int = settings.FORM_AI_SPLIT_SEED,
) -> dict[str, Any]:
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("Split fractions must be in [0, 1).")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1.")

    source_records = _read_source_inputs(source_path)
    labels = _read_labels(labels_path)
    missing_source = sorted(set(labels) - set(source_records))
    missing_labels = sorted(set(source_records) - set(labels))
    if missing_source or missing_labels:
        raise ValueError(
            "Source and label files must contain the same filenames. "
            f"Missing source={len(missing_source)}, missing labels={len(missing_labels)}."
        )

    names_by_class = {
        category: sorted(
            name for name, label in labels.items()
            if label["ground_truth_category"] == category
        )
        for category in ("VULNERABLE", "SAFE")
    }
    rng = random.Random(seed)
    assignments: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for category, names in names_by_class.items():
        category_splits = _split_names(
            names,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            rng=rng,
        )
        for split_name, split_names in category_splits.items():
            assignments[split_name].extend(split_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "splits": {},
        "ground_truth_separate_from_source_inputs": True,
    }

    for split_name, filenames in assignments.items():
        filenames.sort()
        source_output = output_dir / f"{split_name}_inputs.jsonl"
        label_output = output_dir / f"{split_name}_labels.json"
        with source_output.open("w", encoding="utf-8") as stream:
            for filename in filenames:
                stream.write(json.dumps(source_records[filename], ensure_ascii=False) + "\n")
        label_output.write_text(
            json.dumps([labels[name] for name in filenames], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary["splits"][split_name] = {
            "count": len(filenames),
            "vulnerable": sum(labels[name]["ground_truth_category"] == "VULNERABLE" for name in filenames),
            "safe": sum(labels[name]["ground_truth_category"] == "SAFE" for name in filenames),
            "source_inputs": str(source_output),
            "labels": str(label_output),
        }

    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, help="Source-only FormAI JSONL.")
    parser.add_argument("--labels", required=True, type=Path, help="Separate FormAI ground-truth JSON.")
    parser.add_argument("--output-dir", type=Path, default=Path(settings.FORM_AI_SPLIT_OUTPUT_DIR))
    parser.add_argument("--validation-fraction", type=float, default=settings.FORM_AI_SPLIT_VALIDATION_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=settings.FORM_AI_SPLIT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=settings.FORM_AI_SPLIT_SEED)
    args = parser.parse_args()
    try:
        result = split_formai(
            args.inputs, args.labels, args.output_dir,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
