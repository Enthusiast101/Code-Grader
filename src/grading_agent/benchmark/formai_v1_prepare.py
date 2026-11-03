
"""Prepare a reproducible FormAI-v1 vulnerability-detection pilot.

Reads the two FormAI ZIP archives and writes model inputs separately from labels.
It never compiles or executes a C program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import sqlite3
import statistics
import tempfile
import zipfile

from collections import Counter
from pathlib import Path, PurePosixPath


VULNERABLE = "VULNERABLE"
SAFE = "SAFE_WITHIN_BOUND"
UNKNOWN = "UNKNOWN"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalise_source(source: str) -> str:
    """Normalize BOM, line endings and known FormAI header punctuation."""

    source = source.lstrip("\ufeff")
    source = source.replace("\r\n", "\n").replace("\r", "\n")

    lines = source.splitlines(keepends=True)

    if lines:
        lines[0] = re.sub(
            r"(?i)(FormAI\s+DATASET\s+v1\.0)\.(?=\s|$)",
            r"\1",
            lines[0],
            count=1
        )

    return "".join(lines)


def source_hash(source: str) -> str:
    normalized = normalise_source(source)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def normalise_label(value: str) -> str:
    value = (value or "").strip().upper()

    if value == "VULNERABLE":
        return VULNERABLE

    if value == "NOT VULNERABLE UP TO BOUND K":
        return SAFE

    return UNKNOWN


def parse_line(value: str) -> int | None:
    value = (value or "").strip()

    if not value or value.upper() in {"N/A", "NA", "NONE"}:
        return None

    try:
        number = float(value)

        if number >= 1 and number.is_integer():
            return int(number)

    except ValueError:
        pass

    return None


def write_jsonl(stream, record: dict) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_sources(archive: zipfile.ZipFile):
    """Index source ZIP files and calculate dataset statistics."""

    sources = {}
    sizes = []
    hash_counts = Counter()

    empty = 0
    invalid_utf8 = 0
    total_bytes = 0

    infos = [
        item for item in archive.infolist()
        if not item.is_dir() and item.filename.lower().endswith(".c")
    ]

    for index, info in enumerate(infos, 1):

        name = PurePosixPath(info.filename).name

        if name in sources:
            raise ValueError(
                f"Duplicate C filename in source ZIP: {name}"
            )

        raw = archive.read(info)

        raw_digest = hashlib.sha256(raw).hexdigest()
        hash_counts[raw_digest] += 1

        try:
            source = raw.decode("utf-8-sig")
            normalized_digest = source_hash(source)
            valid_utf8 = True

        except UnicodeDecodeError:
            normalized_digest = raw_digest
            valid_utf8 = False
            invalid_utf8 += 1

        if not raw:
            empty += 1

        sizes.append(len(raw))
        total_bytes += len(raw)

        sources[name] = {
            "member": info.filename,
            "sha256": raw_digest,
            "normalized_sha256": normalized_digest,
            "size_bytes": len(raw),
            "utf8_valid": valid_utf8
        }

        if index % 20000 == 0:
            print(
                f"Indexed {index:,}/{len(infos):,} source files",
                flush=True
            )

    stats = {
        "source_c_files": len(infos),
        "empty_files": empty,
        "invalid_utf8_files": invalid_utf8,
        "source_bytes_total": total_bytes,
        "mean_size_bytes": round(statistics.mean(sizes), 2) if sizes else 0,
        "median_size_bytes": statistics.median(sizes) if sizes else 0,
        "maximum_size_bytes": max(sizes, default=0),
        "exact_duplicate_hash_groups": sum(
            value > 1 for value in hash_counts.values()
        ),
        "duplicate_files_beyond_first": sum(
            value - 1 for value in hash_counts.values() if value > 1
        )
    }

    return sources, stats


def prepare_labels(
    zip_path: Path,
    sources: dict,
    db: sqlite3.Connection
):
    """Aggregate classification CSV records by unique filename."""

    counts = Counter()

    # The supplied FormAI CSV has been audited for repeated source
    # consistency by filename. Therefore, hash the first occurrence
    # of each filename rather than hashing repeated copies.

    checked_sources = {}

    sql = """
        INSERT INTO programs
        VALUES (?, ?, ?, ?, 1, ?)

        ON CONFLICT(file_name) DO UPDATE SET
            has_vulnerable = MAX(
                programs.has_vulnerable,
                excluded.has_vulnerable
            ),

            has_safe = MAX(
                programs.has_safe,
                excluded.has_safe
            ),

            has_unknown = MAX(
                programs.has_unknown,
                excluded.has_unknown
            ),

            record_count = programs.record_count + 1,

            source_match = MIN(
                programs.source_match,
                excluded.source_match
            )
    """

    with zipfile.ZipFile(zip_path) as archive:

        csv_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if len(csv_files) != 1:
            raise ValueError(
                f"Expected one CSV in classification ZIP; found {csv_files}"
            )

        with archive.open(csv_files[0]) as raw:

            with io.TextIOWrapper(
                raw,
                encoding="utf-8-sig",
                newline=""
            ) as stream, db:

                reader = csv.DictReader(stream)

                required = {
                    "Filename",
                    "Vulnerability type",
                    "Source code",
                    "Function name",
                    "Line",
                    "Error type"
                }

                if not reader.fieldnames or not required.issubset(
                    reader.fieldnames
                ):
                    raise ValueError(
                        f"Unexpected classification CSV columns: {reader.fieldnames}"
                    )

                for row_number, row in enumerate(reader, 1):

                    counts["classification_rows"] += 1

                    name = PurePosixPath(
                        (row.get("Filename") or "").strip()
                    ).name

                    source = row.get("Source code") or ""

                    if not name or not source:
                        counts["invalid_rows"] += 1
                        continue

                    label = normalise_label(
                        row.get("Vulnerability type")
                    )

                    label_counter = {
                        VULNERABLE: "vulnerable_rows",
                        SAFE: "bounded_safe_rows",
                        UNKNOWN: "unknown_rows"
                    }

                    counts[label_counter[label]] += 1

                    if name not in checked_sources:

                        digest = source_hash(source)
                        metadata = sources.get(name)

                        matched = bool(
                            metadata
                            and digest == metadata["normalized_sha256"]
                        )

                        checked_sources[name] = (digest, matched)

                        if not matched:
                            counts["source_mismatch_files"] += 1

                    else:
                        digest, matched = checked_sources[name]

                        counts["repeated_source_fields_not_rehashed"] += 1

                    if not matched:
                        counts["source_mismatch_rows"] += 1

                    db.execute(
                        sql,
                        (
                            name,
                            int(label == VULNERABLE),
                            int(label == SAFE),
                            int(label == UNKNOWN),
                            int(matched)
                        )
                    )

                    if label == VULNERABLE:

                        function = (
                            row.get("Function name") or ""
                        ).strip()

                        error = (
                            row.get("Error type") or ""
                        ).strip()

                        if function.upper() in {"", "N/A", "NA", "NONE"}:
                            function = None

                        if error.upper() in {"", "N/A", "NA", "NONE"}:
                            error = None

                        db.execute(
                            """
                            INSERT OR IGNORE INTO findings
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                name,
                                function,
                                parse_line(row.get("Line")),
                                error
                            )
                        )

                    if row_number % 25000 == 0:
                        print(
                            f"Read {row_number:,} classification rows",
                            flush=True
                        )

    counts["unique_classification_filenames"] = db.execute(
        "SELECT COUNT(*) FROM programs"
    ).fetchone()[0]

    return dict(counts)


def final_label(
    vulnerable: int,
    safe: int,
    unknown: int
) -> str | None:

    if vulnerable:
        return VULNERABLE

    if unknown:
        return None

    if safe:
        return SAFE

    return None


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--sources-zip",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--classification-zip",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resources/formai_v1/processed")
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=50
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = parser.parse_args()

    if args.per_class < 1:
        parser.error("--per-class must be positive")

    for path in (args.sources_zip, args.classification_zip):

        if not path.is_file():
            parser.error(f"Archive not found: {path}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    output_names = [
        "source_inputs.jsonl",
        "ground_truth_labels.jsonl",
        "inventory.jsonl",
        "exclusions.jsonl",
        "unmatched_classification.jsonl",
        "summary.json"
    ]

    existing = [
        out / name for name in output_names
        if (out / name).exists()
    ]

    if existing and not args.overwrite:
        parser.error(
            f"Output files already exist: {existing}. "
            "Use --overwrite or another output directory."
        )

    for path in existing:
        path.unlink()

    summary = {
        "dataset": "FormAI-v1",
        "seed": args.seed,
        "per_class_requested": args.per_class,

        "sampling_unit": (
            "unique source filename; all findings aggregated"
        ),

        "source_archive_sha256": sha256_file(args.sources_zip),

        "classification_archive_sha256": sha256_file(
            args.classification_zip
        ),

        "ground_truth_exposed_to_model": False,
        "C_programs_executed": False
    }

    # SQLite stores aggregated classification records on disk.
    # The temporary database is removed automatically after preparation.

    with tempfile.TemporaryDirectory(prefix="formai_v1_") as temp_dir:

        db = sqlite3.connect(
            Path(temp_dir) / "formai.sqlite3"
        )

        try:

            db.executescript("""
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;

                CREATE TABLE programs (
                    file_name TEXT PRIMARY KEY,
                    has_vulnerable INTEGER,
                    has_safe INTEGER,
                    has_unknown INTEGER,
                    record_count INTEGER,
                    source_match INTEGER
                );

                CREATE TABLE findings (
                    file_name TEXT,
                    function_name TEXT,
                    line INTEGER,
                    error_type TEXT,
                    UNIQUE(
                        file_name,
                        function_name,
                        line,
                        error_type
                    )
                );
            """)

            with zipfile.ZipFile(args.sources_zip) as source_zip:

                sources, summary["source_inventory"] = prepare_sources(
                    source_zip
                )

            summary["classification_inventory"] = prepare_labels(
                args.classification_zip,
                sources,
                db
            )

            program_rows = {
                row[0]: row[1:]
                for row in db.execute("""
                    SELECT
                        file_name,
                        has_vulnerable,
                        has_safe,
                        has_unknown,
                        record_count,
                        source_match
                    FROM programs
                """)
            }

            class_files = {
                VULNERABLE: [],
                SAFE: []
            }

            excluded_counts = Counter()

            with (out / "inventory.jsonl").open(
                "w", encoding="utf-8"
            ) as inventory:

                with (out / "exclusions.jsonl").open(
                    "w", encoding="utf-8"
                ) as exclusions:

                    for name in sorted(sources):

                        metadata = sources[name]
                        row = program_rows.get(name)

                        label = None
                        reason = None

                        if row is None:

                            status = "UNLABELLED"
                            reason = (
                                "No classification record for this source file."
                            )

                        else:

                            vulnerable, safe, unknown, record_count, matched = row

                            if not metadata["utf8_valid"]:

                                status = "INVALID_SOURCE_ENCODING"
                                reason = "Source is not valid UTF-8."

                            elif not matched:

                                status = "SOURCE_MISMATCH"
                                reason = "CSV source did not match source archive."

                            else:

                                label = final_label(
                                    vulnerable,
                                    safe,
                                    unknown
                                )

                                if label == VULNERABLE:
                                    status = VULNERABLE

                                elif label == SAFE:
                                    status = SAFE

                                else:
                                    status = UNKNOWN
                                    reason = (
                                        "Unknown/timeout is not evidence of safety."
                                    )

                        write_jsonl(
                            inventory,
                            {
                                "file_name": name,
                                "source_sha256": metadata["sha256"],
                                "size_bytes": metadata["size_bytes"],
                                "classification_records": row[3] if row else 0,
                                "status": status,
                                "ground_truth_category": label
                            }
                        )

                        if label:

                            class_files[label].append(name)

                        else:

                            excluded_counts[status] += 1

                            write_jsonl(
                                exclusions,
                                {
                                    "file_name": name,
                                    "reason_code": status,
                                    "reason": reason
                                }
                            )

            unmatched = sorted(
                set(program_rows) - set(sources)
            )

            with (out / "unmatched_classification.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:

                for name in unmatched:

                    write_jsonl(
                        stream,
                        {
                            "file_name": name,
                            "reason": "Not present in source archive."
                        }
                    )

            # Select an equal number of programs from each class.
            rng = random.Random(args.seed)

            selected = {}

            for label in (VULNERABLE, SAFE):

                candidates = sorted(class_files[label])

                if len(candidates) < args.per_class:

                    raise RuntimeError(
                        f"Only {len(candidates)} {label} programs available; "
                        f"requested {args.per_class}."
                    )

                selected[label] = sorted(
                    rng.sample(candidates, args.per_class)
                )

            # Source inputs and reference labels are separate files.
            with zipfile.ZipFile(args.sources_zip) as source_zip:

                with (out / "source_inputs.jsonl").open(
                    "w", encoding="utf-8"
                ) as source_out:

                    with (out / "ground_truth_labels.jsonl").open(
                        "w", encoding="utf-8"
                    ) as label_out:

                        members = {
                            PurePosixPath(i.filename).name: i.filename
                            for i in source_zip.infolist()
                            if not i.is_dir()
                            and i.filename.lower().endswith(".c")
                        }

                        for label in (VULNERABLE, SAFE):

                            for name in selected[label]:

                                raw = source_zip.read(members[name])

                                write_jsonl(
                                    source_out,
                                    {
                                        "file_name": name,
                                        "source_sha256": hashlib.sha256(
                                            raw
                                        ).hexdigest(),
                                        "source_code": raw.decode(
                                            "utf-8-sig"
                                        )
                                    }
                                )

                                findings = [
                                    {
                                        "function_name": fn,
                                        "line": line,
                                        "error_type": error
                                    }
                                    for fn, line, error in db.execute(
                                        """
                                        SELECT
                                            function_name,
                                            line,
                                            error_type
                                        FROM findings
                                        WHERE file_name = ?
                                        ORDER BY line, function_name, error_type
                                        """,
                                        (name,)
                                    )
                                ]

                                write_jsonl(
                                    label_out,
                                    {
                                        "file_name": name,
                                        "ground_truth_category": label,
                                        "findings": findings,

                                        "vulnerable_lines": sorted({
                                            f["line"]
                                            for f in findings
                                            if f["line"]
                                        }),

                                        "function_names": sorted({
                                            f["function_name"]
                                            for f in findings
                                            if f["function_name"]
                                        }),

                                        "error_types": sorted({
                                            f["error_type"]
                                            for f in findings
                                            if f["error_type"]
                                        })
                                    }
                                )

            summary.update({
                "source_files_accounted_for": len(sources),

                "eligible_unique_programs": {
                    label: len(class_files[label])
                    for label in (VULNERABLE, SAFE)
                },

                "selected_unique_programs": {
                    label: len(selected[label])
                    for label in selected
                },

                "selected_total": sum(
                    len(items) for items in selected.values()
                ),

                "excluded_source_programs_by_reason": dict(
                    excluded_counts
                ),

                "classification_filenames_not_in_source_archive": len(
                    unmatched
                ),

                "selected_filenames": selected,
                "outputs": output_names,

                "safety_note": (
                    "Preparation only reads archives; it does not compile "
                    "or execute C programs. Ground-truth labels are separate "
                    "from model inputs."
                )
            })

            (out / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

            print(json.dumps(summary, indent=2, ensure_ascii=False))

        finally:
            db.close()


if __name__ == "__main__":
    main()
