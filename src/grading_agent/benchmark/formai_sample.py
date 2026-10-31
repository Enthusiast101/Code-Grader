"""Create a deterministic, class-stratified FormAI-v2 sample by unique program.

FormAI-v2 may contain multiple verification records for one source file. This
sampler merges those records by ``file_name`` before reservoir sampling. A file
is VULNERABLE if any record proves a vulnerability; it is SAFE only when all
its records are non-vulnerable; files with no vulnerability and any timeout /
unknown record are excluded from the binary benchmark.

Source inputs and labels are written separately to prevent label leakage.
Requires the optional ``benchmark`` dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import ijson  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Install benchmark dependencies with `python -m pip install -e '.[benchmark]'` "
            "to stream the multi-GB FormAI JSON."
        ) from exc
    with path.open("rb") as stream:
        yield from ijson.items(stream, "item")


def _normalise_category(value: Any) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    category = value.strip().upper().replace(" ", "_").replace("-", "_")
    if category == "VULNERABLE":
        return "VULNERABLE"
    if category in {"NON_VULNERABLE", "NOT_VULNERABLE", "SAFE", "CLEAN"}:
        return "SAFE"
    # FormAI uses values such as "UNKNOWN (time out)" for unresolved runs.
    if category.startswith("UNKNOWN") or "TIME_OUT" in category or "TIMEOUT" in category:
        return "UNKNOWN"
    return "UNKNOWN"


def _create_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE programs (
            file_name TEXT PRIMARY KEY,
            source_code TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            has_vulnerable INTEGER NOT NULL DEFAULT 0,
            has_safe INTEGER NOT NULL DEFAULT 0,
            has_unknown INTEGER NOT NULL DEFAULT 0,
            record_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE vulnerability_records (
            file_name TEXT NOT NULL,
            vulnerable_line INTEGER,
            error_type TEXT,
            PRIMARY KEY (file_name, vulnerable_line, error_type),
            FOREIGN KEY (file_name) REFERENCES programs(file_name)
        );
        """
    )


def _aggregate_records(dataset_path: Path, connection: sqlite3.Connection) -> dict[str, int]:
    counts = {"records_seen": 0, "invalid_records": 0, "vulnerable_records": 0,
              "safe_records": 0, "unknown_records": 0, "unique_programs": 0}
    insert_program = """INSERT INTO programs
        (file_name, source_code, source_sha256, has_vulnerable, has_safe, has_unknown, record_count)
        VALUES (?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(file_name) DO UPDATE SET
          has_vulnerable = MAX(programs.has_vulnerable, excluded.has_vulnerable),
          has_safe = MAX(programs.has_safe, excluded.has_safe),
          has_unknown = MAX(programs.has_unknown, excluded.has_unknown),
          record_count = programs.record_count + 1
    """

    with connection:
        for record in iter_records(dataset_path):
            counts["records_seen"] += 1
            file_name = record.get("file_name")
            source_code = record.get("source_code")
            if not isinstance(file_name, str) or not file_name or not isinstance(source_code, str) or not source_code:
                counts["invalid_records"] += 1
                continue

            category = _normalise_category(record.get("category"))
            is_vulnerable = int(category == "VULNERABLE")
            is_safe = int(category == "SAFE")
            is_unknown = int(category == "UNKNOWN")
            counts[{"VULNERABLE": "vulnerable_records", "SAFE": "safe_records", "UNKNOWN": "unknown_records"}[category]] += 1

            digest = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
            existing = connection.execute(
                "SELECT source_sha256 FROM programs WHERE file_name = ?", (file_name,)
            ).fetchone()
            if existing and existing[0] != digest:
                raise ValueError(
                    f"FormAI file_name {file_name!r} maps to different source_code records; "
                    "refusing to merge inconsistent samples."
                )

            connection.execute(
                insert_program,
                (file_name, source_code, digest, is_vulnerable, is_safe, is_unknown),
            )

            if is_vulnerable:
                raw_line = record.get("vulnerable_line")
                line = raw_line if isinstance(raw_line, int) and raw_line > 0 else None
                raw_type = record.get("error_type")
                error_type = raw_type.strip() if isinstance(raw_type, str) and raw_type.strip() else None
                connection.execute(
                    "INSERT OR IGNORE INTO vulnerability_records (file_name, vulnerable_line, error_type) VALUES (?, ?, ?)",
                    (file_name, line, error_type),
                )

    counts["unique_programs"] = connection.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
    return counts


def _program_class(has_vulnerable: int, has_safe: int, has_unknown: int) -> str | None:
    # A confirmed vulnerability wins even if other verification properties timed out.
    if has_vulnerable:
        return "VULNERABLE"
    # Do not call a program safe when one or more verification attempts timed out.
    if has_unknown:
        return None
    if has_safe:
        return "SAFE"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-output", type=Path, default=Path("benchmark/formai_inputs.jsonl"))
    parser.add_argument("--labels-output", type=Path, default=Path("benchmark/formai_ground_truth.json"))
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    if not args.dataset.is_file():
        parser.error(f"Dataset file does not exist: {args.dataset}")

    args.source_output.parent.mkdir(parents=True, exist_ok=True)
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # SQLite keeps the 2.5+ GB source JSON out of RAM while deduplicating records.
    with tempfile.TemporaryDirectory(prefix="formai_sample_") as temp_dir:
        db_path = Path(temp_dir) / "formai.sqlite3"
        connection = sqlite3.connect(db_path)
        try:
            _create_database(connection)
            counts = _aggregate_records(args.dataset, connection)
            reservoirs: dict[str, list[str]] = {"VULNERABLE": [], "SAFE": []}
            eligible_counts = {"VULNERABLE": 0, "SAFE": 0}
            excluded_unknown_only = 0

            cursor = connection.execute(
                "SELECT file_name, has_vulnerable, has_safe, has_unknown "
                "FROM programs ORDER BY file_name"
            )
            for file_name, has_vulnerable, has_safe, has_unknown in cursor:
                category = _program_class(has_vulnerable, has_safe, has_unknown)
                if category is None:
                    excluded_unknown_only += 1
                    continue
                eligible_counts[category] += 1
                bucket = reservoirs[category]
                seen = eligible_counts[category]
                if len(bucket) < args.per_class:
                    bucket.append(file_name)
                else:
                    slot = rng.randrange(seen)
                    if slot < args.per_class:
                        bucket[slot] = file_name

            selected = {key: len(value) for key, value in reservoirs.items()}
            if not selected["VULNERABLE"] or not selected["SAFE"]:
                raise RuntimeError(
                    "Could not construct both benchmark classes. "
                    f"Selected={selected}; eligible={eligible_counts}; "
                    f"excluded_unknown_only={excluded_unknown_only}."
                )

            labels: list[dict[str, Any]] = []
            with args.source_output.open("w", encoding="utf-8") as source_file:
                for category in ("VULNERABLE", "SAFE"):
                    for file_name in sorted(reservoirs[category]):
                        row = connection.execute(
                            "SELECT source_code FROM programs WHERE file_name = ?", (file_name,)
                        ).fetchone()
                        if row is None:
                            raise RuntimeError(f"Internal sampling error: missing source for {file_name}")
                        source_file.write(json.dumps(
                            {"file_name": file_name, "source_code": row[0]}, ensure_ascii=False
                        ) + "\n")

                        vulnerability_rows = connection.execute(
                            "SELECT vulnerable_line, error_type FROM vulnerability_records "
                            "WHERE file_name = ? ORDER BY vulnerable_line, error_type",
                            (file_name,),
                        ).fetchall()
                        lines = sorted({line for line, _ in vulnerability_rows if isinstance(line, int)})
                        error_types = sorted({kind for _, kind in vulnerability_rows if isinstance(kind, str)})
                        labels.append({
                            "file_name": file_name,
                            "ground_truth_category": category,
                            "vulnerable_line": lines[0] if lines else None,
                            "vulnerable_lines": lines,
                            "error_type": error_types[0] if error_types else None,
                            "error_types": error_types,
                        })

            args.labels_output.write_text(
                json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            summary = {
                "source_inputs": str(args.source_output),
                "ground_truth_labels": str(args.labels_output),
                "selected_unique_programs": selected,
                "eligible_unique_programs": eligible_counts,
                "excluded_unknown_only_programs": excluded_unknown_only,
                "record_counts": counts,
                "seed": args.seed,
                "sampling_unit": "unique file_name; all vulnerability locations/types aggregated",
                "ground_truth_separate_from_source_inputs": True,
            }
            print(json.dumps(summary, indent=2))
        finally:
            connection.close()


if __name__ == "__main__":
    main()
