"""Strict filename/count checks for source-only FormAI benchmark splits.

The labels are used only to verify split alignment and later evaluate results;
label categories are never passed to the prediction pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_source_manifest(path: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid source JSONL at line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Source JSONL line {line_number} must be an object.")
            name, source = item.get("file_name"), item.get("source_code")
            if not isinstance(name, str) or not name.strip() or not isinstance(source, str) or not source.strip():
                raise ValueError(f"Source JSONL line {line_number} must contain non-empty file_name and source_code strings.")
            if name in seen:
                raise ValueError(f"Duplicate source program {name!r} at JSONL line {line_number}.")
            seen.add(name)
            names.append(name)
    if not names:
        raise ValueError(f"Source input file contains no programs: {path}")
    return names


def read_label_names(path: Path) -> list[str]:
    """Read only filenames; do not inspect ground-truth classes before prediction."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid ground-truth JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("Ground-truth labels must be a JSON array.")
    names: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(payload, 1):
        if not isinstance(item, dict) or not isinstance(item.get("file_name"), str) or not item["file_name"].strip():
            raise ValueError(f"Ground-truth record {index} has no valid file_name.")
        name = item["file_name"]
        if name in seen:
            raise ValueError(f"Duplicate ground-truth program {name!r} at record {index}.")
        seen.add(name)
        names.append(name)
    if not names:
        raise ValueError(f"Ground-truth file contains no programs: {path}")
    return names


def validate_benchmark_split(inputs_path: Path, labels_path: Path) -> dict[str, Any]:
    """Require exact one-to-one filename alignment before a full Gradio run."""
    input_names = read_source_manifest(inputs_path)
    label_names = read_label_names(labels_path)
    input_set, label_set = set(input_names), set(label_names)
    missing_sources = sorted(label_set - input_set)
    missing_labels = sorted(input_set - label_set)
    if missing_sources or missing_labels:
        raise ValueError(
            "The source inputs and labels are not the same split. "
            f"Source programs={len(input_names)}, label programs={len(label_names)}, "
            f"missing source files={len(missing_sources)} {missing_sources[:5]}, "
            f"source files without labels={len(missing_labels)} {missing_labels[:5]}. "
            "Upload matching test_inputs.jsonl and test_labels.json from the same split."
        )
    return {
        "programs": len(input_names),
        "filenames_match": True,
    }
