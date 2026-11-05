"""FormAI dataset: download, verify, extract and index by program.

FormAI (https://github.com/FormAI-Dataset/FormAI-dataset) labels AI-generated C
programs with the ESBMC bounded model checker. The published files hold one
record per verified property, so a program can appear many times. This module
turns them into one SQLite index with one row per program:

* ``programs``: source, generating model, task header and the program label:
  ``VULNERABLE`` when any record has an ESBMC counterexample, ``SAFE`` when
  every record finished as non-vulnerable within the bound, otherwise
  ``UNKNOWN`` (time-outs and parse errors are not evidence of safety and are
  excluded from benchmarks).
* ``vulnerabilities``: every confirmed vulnerable line with its ESBMC error type
  and a coarse ``family`` shared with the predictor (see ``FAMILIES``).

Building the index only reads archives; no FormAI program is compiled or run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grading_agent.config import settings

INDEX_SCHEMA_VERSION = 2
BASE_URL = "https://github.com/FormAI-Dataset/FormAI-dataset/raw/main/"

VULNERABLE = "VULNERABLE"
SAFE = "SAFE"
UNKNOWN = "UNKNOWN"

# Coarse vulnerability families. ESBMC error types, cppcheck ids and LLM
# categories all map onto these so detection type and location can be scored.
FAMILIES: dict[str, str] = {
    "input_buffer_overflow": "Buffer overflow from unbounded input (scanf/fscanf/sscanf %s, gets)",
    "null_dereference": "Dereference of a possibly NULL pointer (e.g. unchecked malloc result)",
    "out_of_bounds": "Array or buffer access outside its bounds",
    "invalid_pointer": "Invalid, freed, dangling or mis-typed pointer use, invalid free",
    "memory_leak": "Allocated memory never freed",
    "integer_overflow": "Signed integer arithmetic overflow",
    "float_overflow": "Floating-point overflow",
    "division_by_zero": "Division or modulus by zero",
    "other": "Other safety property violation",
}

_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    # Ordered: the first matching rule wins.
    ("input_buffer_overflow", r"buffer overflow on (?:f|s)?scanf|buffer overflow on gets|\binvalidscanf\b|"
                              r"\b(?:f|s)?scanf\b.*(?:overflow|width|unbounded|%s)|\bgets\b|unbounded (?:input|read)"),
    ("float_overflow", r"floating[- ]point|ieee_"),
    ("division_by_zero", r"division by zero|divide by zero|\bzerodiv|modulo by zero|divisor (?:is |may be |could be )?zero"),
    ("integer_overflow", r"arithmetic overflow|integer overflow|signed overflow|integeroverflow|overflow on (?:add|sub|mul|shl|neg)|wrap ?around"),
    ("memory_leak", r"forgotten memory|memory leak|\bmemleak|never freed|not freed|leaks? memory"),
    ("null_dereference", r"null pointer|nullpointer|\bnull dereference|dereference failure: null|unchecked (?:malloc|calloc|realloc|allocation)|"
                         r"allocation (?:result|failure) (?:is )?(?:not|never) checked"),
    ("out_of_bounds", r"array bounds|out[- ]of[- ]bounds|outofbounds|arrayindex|bufferaccess|index out of|vla array size|"
                      r"access to object out of bounds|buffer overflow|buffer overrun|overrun|off[- ]by[- ]one"),
    ("invalid_pointer", r"invalid pointer|invalidated dynamic object|expired variable|string constant|incompatible base type|"
                        r"non-dynamic memory|oversized field|code type|operand of free|same object violation|file object|"
                        r"use[- ]after[- ]free|double[- ]free|dangling|invalid free|deallocuse|doublefree|danglingpointer"),
)
_FAMILY_PATTERNS = tuple((family, re.compile(pattern, re.IGNORECASE)) for family, pattern in _FAMILY_RULES)


def error_family(text: Any) -> str:
    """Map an ESBMC error type, cppcheck id or free-text category to a family."""
    value = str(text or "").strip()
    if not value:
        return "other"
    if value in FAMILIES:
        return value
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(value):
            return family
    return "other"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    directory: str
    downloads: tuple[tuple[str, str], ...]  # (file name, sha256)
    kind: str  # "json_7z" or "csv_zip"
    default_generator: str | None = None
    notes: str = ""
    extracted_member: str | None = None


CATALOG: dict[str, DatasetSpec] = {
    "v2": DatasetSpec(
        key="v2",
        name="FormAI-v2",
        directory="formai_v2",
        downloads=(
            ("FormAI-v2-classification.7z.001", "512a344251c1b86ecef85ef8e47d16cc7fec259be10c578af21cd6444561c9c0"),
            ("FormAI-v2-classification.7z.002", "dd5c7901641e1d0262c37f23700bdb20a4896ceeec4d8e0e9ab8a1499008d4ab"),
        ),
        kind="json_7z",
        extracted_member="FormAI-v2.json",
        notes="Programs written by nine LLMs, verified with ESBMC 7.5.",
    ),
    "v1": DatasetSpec(
        key="v1",
        name="FormAI-v1",
        directory="formai_v1",
        downloads=(
            ("FormAI_dataset_classification-V1.zip", "82b00611d71ff992d85b7bba20ebece580bfd055939b0142861326dff17ede74"),
        ),
        kind="csv_zip",
        default_generator="gpt35",
        notes="Programs written by GPT-3.5-turbo, verified with ESBMC 7.3.",
    ),
}


def dataset_dir(version: str) -> Path:
    return Path(settings.FORM_AI_DATA_ROOT) / CATALOG[version].directory


def index_path(version: str) -> Path:
    return dataset_dir(version) / "index.sqlite3"


Progress = Callable[[str], None]


def _say(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(version: str, *, progress: Progress | None = None) -> list[Path]:
    """Download the archives for ``version`` (if missing) and verify SHA-256."""
    import requests

    spec = CATALOG[version]
    folder = dataset_dir(version)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, expected in spec.downloads:
        path = folder / name
        if not path.is_file() or sha256_file(path) != expected:
            _say(progress, f"Downloading {name}")
            partial = path.with_suffix(path.suffix + ".part")
            with requests.get(BASE_URL + name, stream=True, timeout=60) as response:
                response.raise_for_status()
                with partial.open("wb") as out:
                    for chunk in response.iter_content(1 << 20):
                        out.write(chunk)
            actual = sha256_file(partial)
            if actual != expected:
                partial.unlink(missing_ok=True)
                raise ValueError(f"{name}: SHA-256 {actual} does not match the published {expected}.")
            partial.replace(path)
        paths.append(path)
    return paths


def _seven_zip() -> str | None:
    found = shutil.which("7z") or shutil.which("7za")
    if found:
        return found
    for candidate in (Path(r"C:\Program Files\7-Zip\7z.exe"), Path(r"C:\Program Files (x86)\7-Zip\7z.exe")):
        if candidate.is_file():
            return str(candidate)
    return None


def _extract_json(version: str, *, progress: Progress | None = None) -> Path:
    spec = CATALOG[version]
    folder = dataset_dir(version)
    target = folder / "extracted" / str(spec.extracted_member)
    if target.is_file():
        return target
    first = folder / spec.downloads[0][0]
    _say(progress, f"Extracting {spec.extracted_member}")
    tool = _seven_zip()
    if tool:
        subprocess.run([tool, "x", "-y", str(first), f"-o{target.parent}"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    else:
        try:
            import multivolumefile  # type: ignore
            import py7zr  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Extracting FormAI-v2 needs 7-Zip on PATH or `pip install py7zr multivolumefile`.") from exc
        with multivolumefile.open(str(first)[:-4], mode="rb") as volume, py7zr.SevenZipFile(volume, "r") as archive:
            archive.extractall(path=target.parent)
    if not target.is_file():
        raise RuntimeError(f"Extraction finished but {target} is missing.")
    return target


# ---------------------------------------------------------------------------
# Record readers: both yield normalised verification records.
# ---------------------------------------------------------------------------

def _category(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text == "VULNERABLE":
        return VULNERABLE
    if text in {"NON-VULNERABLE", "NON_VULNERABLE", "NOT VULNERABLE", "NOT VULNERABLE UP TO BOUND K", "SAFE"}:
        return SAFE
    return UNKNOWN


def _positive_int(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number >= 1 and number.is_integer() else None


def _text_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return None if not text or text.upper() in {"N/A", "NA", "NONE"} else text


def _iter_v2(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import ijson  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install ijson (`pip install ijson`) to stream the multi-GB FormAI-v2 JSON.") from exc
    with path.open("rb") as stream:
        for record in ijson.items(stream, "item", use_float=True):
            yield {
                "file_name": record.get("file_name"),
                "category": _category(record.get("category")),
                "raw_category": record.get("category"),
                "source_code": record.get("source_code"),
                "line": _positive_int(record.get("vulnerable_line")),
                "function": _text_or_none(record.get("function")),
                "error_type": _text_or_none(record.get("error_type")),
                "num_lines": _positive_int(record.get("num_lines")),
                "cyclomatic_complexity": record.get("cyclomatic_complexity")
                if isinstance(record.get("cyclomatic_complexity"), (int, float)) else None,
            }


def _iter_v1(path: Path) -> Iterator[dict[str, Any]]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in {path.name}; found {members}.")
        with archive.open(members[0]) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                source = row.get("Source code") or ""
                yield {
                    "file_name": Path((row.get("Filename") or "").strip()).name,
                    "category": _category(row.get("Vulnerability type")),
                    "raw_category": row.get("Vulnerability type"),
                    "source_code": source,
                    "line": _positive_int(row.get("Line")),
                    "function": _text_or_none(row.get("Function name")),
                    "error_type": _text_or_none(row.get("Error type")),
                    "num_lines": source.count("\n") + 1 if source else None,
                    "cyclomatic_complexity": None,
                }


def iter_records(version: str, *, progress: Progress | None = None) -> Iterator[dict[str, Any]]:
    spec = CATALOG[version]
    if spec.kind == "json_7z":
        yield from _iter_v2(_extract_json(version, progress=progress))
    else:
        yield from _iter_v1(dataset_dir(version) / spec.downloads[0][0])


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

_HEADER = re.compile(r"^\s*//\s*(?P<model>.*?)\s+DATASET\s+v[\d.]+\.?\s*Category:\s*(?P<task>.*?)\s*;\s*Style:\s*(?P<style>.*?)\s*$", re.I)


def parse_header(source: str) -> dict[str, str | None]:
    """Read the FormAI first-line header: generator, task category and style."""
    first = source.lstrip("\ufeff").split("\n", 1)[0]
    match = _HEADER.match(first)
    if not match:
        return {"header_model": None, "task": None, "style": None}
    return {"header_model": match.group("model"), "task": match.group("task"), "style": match.group("style")}


def generator_from_name(file_name: str, default: str | None = None) -> str:
    """``gpt4o_mini-31631.c`` -> ``gpt4o_mini``; FormAI-v1 names use ``default``."""
    stem = Path(file_name).stem
    if "-" in stem and not stem.startswith("FormAI_"):
        return stem.rsplit("-", 1)[0]
    return default or "unknown"


_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE programs (
    file_name TEXT PRIMARY KEY,
    generator TEXT NOT NULL,
    task TEXT,
    style TEXT,
    source_sha256 TEXT NOT NULL,
    num_lines INTEGER,
    cyclomatic_complexity REAL,
    record_count INTEGER NOT NULL,
    vulnerable_records INTEGER NOT NULL,
    safe_records INTEGER NOT NULL,
    unknown_records INTEGER NOT NULL,
    label TEXT NOT NULL,
    verification_complete INTEGER NOT NULL,
    primary_family TEXT,
    family_count INTEGER NOT NULL DEFAULT 0
);
-- Source lives apart from the labels so sampling scans stay small.
CREATE TABLE sources (file_name TEXT PRIMARY KEY, source_code TEXT NOT NULL);
CREATE TABLE vulnerabilities (
    file_name TEXT NOT NULL,
    line INTEGER,
    function TEXT,
    error_type TEXT,
    family TEXT NOT NULL,
    UNIQUE (file_name, line, function, error_type)
);
"""


def build_index(version: str, *, progress: Progress | None = None, overwrite: bool = False) -> dict[str, Any]:
    """Aggregate verification records into one row per program."""
    spec = CATALOG[version]
    target = index_path(version)
    if target.is_file() and not overwrite and is_ready(version):
        return index_summary(version)
    archives = [dataset_dir(version) / name for name, _ in spec.downloads]
    if not all(path.is_file() for path in archives):
        download(version, progress=progress)
    archive_hashes = {}
    for path, (name, expected) in zip(archives, spec.downloads):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{name}: SHA-256 {actual} does not match the published {expected}; re-download it.")
        archive_hashes[name] = actual

    started = time.time()
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.with_suffix(".building")
    building.unlink(missing_ok=True)
    connection = sqlite3.connect(building)
    try:
        connection.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;" + _SCHEMA)
        # name -> [sha, records, vulnerable, safe, unknown, conflict]
        state: dict[str, list[Any]] = {}
        counts = {"records": 0, "invalid_records": 0, "source_conflict_records": 0}
        pending_programs: list[tuple] = []
        pending_sources: list[tuple] = []
        pending_vulns: list[tuple] = []

        def flush() -> None:
            connection.executemany(
                "INSERT INTO programs (file_name, generator, task, style, source_sha256, num_lines, "
                "cyclomatic_complexity, record_count, vulnerable_records, safe_records, unknown_records, label, "
                "verification_complete) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 'UNKNOWN', 0)", pending_programs)
            connection.executemany("INSERT INTO sources (file_name, source_code) VALUES (?, ?)", pending_sources)
            connection.executemany(
                "INSERT OR IGNORE INTO vulnerabilities (file_name, line, function, error_type, family) VALUES (?, ?, ?, ?, ?)",
                pending_vulns)
            pending_programs.clear()
            pending_sources.clear()
            pending_vulns.clear()

        for record in iter_records(version, progress=progress):
            counts["records"] += 1
            name, source = record.get("file_name"), record.get("source_code")
            if not isinstance(name, str) or not name or not isinstance(source, str) or not source.strip():
                counts["invalid_records"] += 1
                continue
            source = source.replace("\r\n", "\n").replace("\r", "\n")
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            entry = state.get(name)
            if entry is None:
                header = parse_header(source)
                entry = state[name] = [digest, 0, 0, 0, 0, 0]
                pending_programs.append((
                    name, generator_from_name(name, spec.default_generator), header["task"], header["style"],
                    digest, record.get("num_lines") or source.count("\n") + 1,
                    record.get("cyclomatic_complexity"),
                ))
                pending_sources.append((name, source))
            elif entry[0] != digest:
                # One file name with two different sources cannot be labelled reliably.
                entry[5] = 1
                counts["source_conflict_records"] += 1
            entry[1] += 1
            category = record["category"]
            entry[{VULNERABLE: 2, SAFE: 3, UNKNOWN: 4}[category]] += 1
            if category == VULNERABLE:
                pending_vulns.append((name, record.get("line"), record.get("function"), record.get("error_type"),
                                      error_family(record.get("error_type"))))
            if len(pending_programs) + len(pending_vulns) >= 20_000:
                flush()
            if counts["records"] % 100_000 == 0:
                _say(progress, f"Indexed {counts['records']:,} verification records ({len(state):,} programs)")
        flush()

        _say(progress, "Labelling programs")
        updates = []
        for name, (_, records, vulnerable, safe, unknown, conflict) in state.items():
            if conflict:
                label = UNKNOWN
            elif vulnerable:
                label = VULNERABLE
            elif safe and not unknown:
                label = SAFE
            else:
                label = UNKNOWN
            updates.append((records, vulnerable, safe, unknown, label, int(not unknown and not conflict), name))
        connection.executemany(
            "UPDATE programs SET record_count=?, vulnerable_records=?, safe_records=?, unknown_records=?, label=?, "
            "verification_complete=? WHERE file_name=?", updates)
        connection.execute("CREATE INDEX vulnerabilities_file ON vulnerabilities(file_name)")
        connection.execute("CREATE INDEX programs_label ON programs(label, primary_family, generator)")
        # Primary family: the family of the earliest vulnerable line (ties -> FAMILIES order).
        order = {family: index for index, family in enumerate(FAMILIES)}
        primary = []
        families_by_program: dict[str, list[tuple[int, int, str]]] = {}
        for name, line, family in connection.execute("SELECT file_name, line, family FROM vulnerabilities"):
            families_by_program.setdefault(name, []).append((line if line is not None else 10**9, order.get(family, 99), family))
        for name, items in families_by_program.items():
            items.sort()
            primary.append((items[0][2], len({item[2] for item in items}), name))
        connection.executemany("UPDATE programs SET primary_family=?, family_count=? WHERE file_name=?", primary)

        # Line-alignment audit: an input-overflow line should contain the input call.
        alignment_rows = connection.execute(
            "SELECT v.line, s.source_code FROM vulnerabilities v JOIN sources s USING(file_name) "
            "WHERE v.family='input_buffer_overflow' AND v.line IS NOT NULL LIMIT 5000").fetchall()
        aligned = sum(1 for line, source in alignment_rows
                      if re.search(r"\b(?:f|s)?scanf\b|\bgets\b", (source.split("\n") + [""] * line)[line - 1]))
        label_counts = dict(connection.execute("SELECT label, COUNT(*) FROM programs GROUP BY label").fetchall())
        meta = {
            "dataset": spec.name,
            "version": version,
            "index_schema": INDEX_SCHEMA_VERSION,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "build_seconds": round(time.time() - started, 1),
            "archive_sha256": archive_hashes,
            "verification_records": counts["records"],
            "invalid_records": counts["invalid_records"],
            "source_conflict_records": counts["source_conflict_records"],
            "programs": len(state),
            "labels": label_counts,
            "line_alignment_check": {
                "input_overflow_lines_checked": len(alignment_rows),
                "line_contains_input_call": round(aligned / len(alignment_rows), 4) if alignment_rows else None,
            },
            "label_policy": (
                "VULNERABLE: at least one ESBMC counterexample. SAFE: every property finished without a violation "
                "within the bound. UNKNOWN (excluded): time-outs, parse errors, or one file name with two sources."
            ),
        }
        connection.executemany("INSERT INTO meta (key, value) VALUES (?, ?)",
                               [(key, json.dumps(value)) for key, value in meta.items()])
        connection.commit()
    finally:
        connection.close()
    building.replace(target)
    _say(progress, f"Index ready: {len(state):,} programs")
    return index_summary(version)


# ---------------------------------------------------------------------------
# Reading the index
# ---------------------------------------------------------------------------

class _ClosingConnection(sqlite3.Connection):
    """``with connect(...) as c:`` closes the file (sqlite3's default only commits)."""

    def __exit__(self, *exc_info: Any) -> bool:
        super().__exit__(*exc_info)
        self.close()
        return False


def connect(version: str) -> sqlite3.Connection:
    path = index_path(version)
    if not path.is_file():
        raise FileNotFoundError(
            f"{CATALOG[version].name} is not indexed yet. Run "
            f"`python -m grading_agent.benchmark.formai_dataset prepare --dataset {version}` "
            "or use 'Prepare dataset' in the app."
        )
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False,
                                 factory=_ClosingConnection)
    connection.row_factory = sqlite3.Row
    return connection


def is_ready(version: str) -> bool:
    """True when the index exists and was built by this schema version."""
    path = index_path(version)
    if not path.is_file():
        return False
    try:
        with connect(version) as connection:
            row = connection.execute("SELECT value FROM meta WHERE key='index_schema'").fetchone()
        return row is not None and json.loads(row[0]) == INDEX_SCHEMA_VERSION
    except sqlite3.Error:
        return False


def index_summary(version: str) -> dict[str, Any]:
    with connect(version) as connection:
        meta = {row["key"]: json.loads(row["value"]) for row in connection.execute("SELECT key, value FROM meta")}
        generators = {
            row["generator"]: {"VULNERABLE": row["v"], "SAFE": row["s"], "UNKNOWN": row["u"]}
            for row in connection.execute(
                "SELECT generator, SUM(label='VULNERABLE') v, SUM(label='SAFE') s, SUM(label='UNKNOWN') u "
                "FROM programs GROUP BY generator ORDER BY generator")
        }
        families = dict(connection.execute(
            "SELECT primary_family, COUNT(*) FROM programs WHERE label='VULNERABLE' GROUP BY primary_family "
            "ORDER BY COUNT(*) DESC").fetchall())
    meta["generators"] = generators
    meta["primary_families"] = families
    meta["eligible_programs"] = meta["labels"].get(VULNERABLE, 0) + meta["labels"].get(SAFE, 0)
    return meta


@dataclass
class ProgramLabel:
    file_name: str
    label: str
    verification_complete: bool
    lines: list[int] = field(default_factory=list)
    error_types: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    vulnerabilities: list[dict[str, Any]] = field(default_factory=list)


def load_sources(connection: sqlite3.Connection, names: list[str]) -> dict[str, str]:
    """Source only: the model-facing view of a program never includes its label."""
    out: dict[str, str] = {}
    for start in range(0, len(names), 500):
        chunk = names[start:start + 500]
        marks = ",".join("?" * len(chunk))
        for row in connection.execute(f"SELECT file_name, source_code FROM sources WHERE file_name IN ({marks})", chunk):
            out[row["file_name"]] = row["source_code"]
    return out


def load_labels(connection: sqlite3.Connection, names: list[str]) -> dict[str, ProgramLabel]:
    labels: dict[str, ProgramLabel] = {}
    for start in range(0, len(names), 500):
        chunk = names[start:start + 500]
        marks = ",".join("?" * len(chunk))
        for row in connection.execute(
                f"SELECT file_name, label, verification_complete FROM programs WHERE file_name IN ({marks})", chunk):
            labels[row["file_name"]] = ProgramLabel(row["file_name"], row["label"], bool(row["verification_complete"]))
        for row in connection.execute(
                f"SELECT file_name, line, function, error_type, family FROM vulnerabilities WHERE file_name IN ({marks}) "
                "ORDER BY file_name, line", chunk):
            label = labels[row["file_name"]]
            label.vulnerabilities.append(dict(row))
    for label in labels.values():
        label.lines = sorted({v["line"] for v in label.vulnerabilities if v["line"]})
        label.error_types = sorted({v["error_type"] for v in label.vulnerabilities if v["error_type"]})
        label.families = sorted({v["family"] for v in label.vulnerabilities})
    return labels


def prepare(version: str, *, progress: Progress | None = None, overwrite: bool = False) -> dict[str, Any]:
    """Download (if needed), verify, extract and index one dataset version."""
    download(version, progress=progress)
    return build_index(version, progress=progress, overwrite=overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Download, verify, extract and index a FormAI version.")
    prep.add_argument("--dataset", choices=sorted(CATALOG), default="v2")
    prep.add_argument("--overwrite", action="store_true", help="Rebuild the index even if it exists.")
    info = sub.add_parser("info", help="Print the index summary.")
    info.add_argument("--dataset", choices=sorted(CATALOG), default="v2")
    args = parser.parse_args()
    if args.command == "prepare":
        summary = prepare(args.dataset, progress=lambda message: print(message, flush=True), overwrite=args.overwrite)
    else:
        summary = index_summary(args.dataset)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
