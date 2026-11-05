"""Similarity between the submission and other submissions in a directory.

Code is compared as normalized token shingles: comments and literals are
removed and every identifier becomes ``IDENT``, so renaming variables or
reformatting does not hide copying. A high score is a review signal, not a
plagiarism verdict.

Scoring (0-100, higher means more original):

* no comparison directory configured, missing, or without C files -> 100;
* otherwise the submission's highest similarity ``s`` to any other file maps
  linearly from 100 at ``low_threshold`` down to 0 at ``high_threshold``.

The analyzer also reports the full pairwise similarity matrix of every file
in the directory, so an instructor can review the whole cohort.
"""
from __future__ import annotations

import hashlib
import re
from itertools import combinations
from pathlib import Path
from typing import Any

from grading_agent.analyzers.source_metrics import strip_comments_and_literals
from grading_agent.config import settings

_C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline",
    "int", "long", "register", "restrict", "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile",
    "while", "_Bool", "_Complex", "_Imaginary",
}
_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|->|\+\+|--|&&|\|\||[^\s]")
_MATRIX_FILE_LIMIT = 200


def _tokens(source: str) -> list[str]:
    clean = strip_comments_and_literals(source)
    result = []
    for token in _TOKEN.findall(clean):
        if token in _C_KEYWORDS:
            result.append(token)
        elif re.fullmatch(r"[A-Za-z_]\w*", token):
            result.append("IDENT")
        elif token.isdigit():
            result.append("NUMBER")
        else:
            result.append(token)
    return result


def _shingles(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}


def _similarity(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    # Jaccard penalizes unrelated extra code; containment helps detect copied fragments.
    jaccard = intersection / union if union else 0.0
    containment = intersection / min(len(left), len(right))
    return max(jaccard, containment)


def _fingerprint(source: str, shingle_size: int) -> tuple[str, set[tuple[str, ...]]]:
    tokens = _tokens(source)
    return hashlib.sha256(" ".join(tokens).encode()).hexdigest(), _shingles(tokens, shingle_size)


def _pair_similarity(left: tuple[str, set], right: tuple[str, set]) -> float:
    return 1.0 if left[0] == right[0] else _similarity(left[1], right[1])


def similarity_score(max_similarity: float, low_threshold: float, high_threshold: float) -> float:
    """Map the highest similarity to a 0-100 originality score."""
    if max_similarity <= low_threshold:
        return 100.0
    if max_similarity >= high_threshold:
        return 0.0
    return round(100.0 * (high_threshold - max_similarity) / (high_threshold - low_threshold), 2)


def _full_marks(reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "status": "completed",
        "score": 100.0,
        "score_scale": "0-100",
        "comparison_performed": False,
        "max_similarity": None,
        "files_compared": 0,
        "flagged_for_human_review": False,
        "summary": reason + " Similarity is therefore graded at full marks.",
        **extra,
    }


def run_similarity_flag(
    source_code: str,
    corpus_dir: str | Path | None,
    flag_threshold: float = settings.ANALYZER_SIMILARITY_THRESHOLD,
    max_files: int = settings.ANALYZER_SIMILARITY_MAX_FILES,
    shingle_size: int = settings.ANALYZER_SIMILARITY_SHINGLE_SIZE,
    max_file_bytes: int = settings.ANALYZER_SIMILARITY_MAX_FILE_BYTES,
    exclude_path: str | Path | None = None,
    low_threshold: float = settings.ANALYZER_SIMILARITY_LOW_THRESHOLD,
    submission_name: str | None = None,
) -> dict[str, Any]:
    """Score the submission's originality against every file in ``corpus_dir``."""
    if not 0 <= low_threshold < flag_threshold <= 1 or shingle_size < 2 or max_files < 1:
        raise ValueError("Invalid similarity analyzer parameters: need 0 <= low_threshold < flag_threshold <= 1.")
    if corpus_dir in (None, ""):
        return _full_marks("No comparison directory was configured.")
    root = Path(corpus_dir)
    if not root.is_dir():
        return _full_marks(f"The comparison directory '{corpus_dir}' does not exist.", configured_dir=str(corpus_dir))

    excluded = Path(exclude_path).resolve() if exclude_path else None
    own_name = submission_name or (excluded.name if excluded else None)
    own_bytes = source_code.encode("utf-8")
    candidates = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".c")

    others: list[tuple[str, tuple[str, set]]] = []
    skipped: list[str] = []
    self_matches: list[str] = []
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        if excluded is not None and path.resolve() == excluded:
            self_matches.append(relative)
            continue
        if len(others) >= max_files:
            skipped.append(relative)
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                skipped.append(relative)
                continue
            raw = path.read_bytes()
        except OSError:
            skipped.append(relative)
            continue
        # The submission itself may sit in the directory under its own name
        # (for example when Gradio grades a staged copy of one cohort file).
        if own_name and path.name == own_name and raw.replace(b"\r\n", b"\n") == own_bytes.replace(b"\r\n", b"\n"):
            self_matches.append(relative)
            continue
        others.append((relative, _fingerprint(raw.decode("utf-8", errors="replace"), shingle_size)))

    if not others:
        return _full_marks(
            f"The comparison directory '{root.name}' contains no other C submissions.",
            configured_dir=str(corpus_dir), files_skipped=skipped, self_matches_excluded=self_matches,
        )

    submission_print = _fingerprint(source_code, shingle_size)
    per_file = sorted(
        (
            {
                "file": name,
                "similarity": round(_pair_similarity(submission_print, fingerprint), 4),
                "exact_normalized_match": submission_print[0] == fingerprint[0],
            }
            for name, fingerprint in others
        ),
        key=lambda item: item["similarity"],
        reverse=True,
    )
    max_similarity = per_file[0]["similarity"]
    score = similarity_score(max_similarity, low_threshold, flag_threshold)

    # Every file against every other file, for cohort-level review.
    matrix_files = others[:_MATRIX_FILE_LIMIT]
    names = [name for name, _ in matrix_files]
    matrix = [[1.0 if i == j else 0.0 for j in range(len(names))] for i in range(len(names))]
    flagged_pairs: list[dict[str, Any]] = []
    for i, j in combinations(range(len(matrix_files)), 2):
        value = round(_pair_similarity(matrix_files[i][1], matrix_files[j][1]), 4)
        matrix[i][j] = matrix[j][i] = value
        if value >= flag_threshold:
            flagged_pairs.append({"files": [names[i], names[j]], "similarity": value})
    flagged_pairs.sort(key=lambda item: item["similarity"], reverse=True)

    closest = per_file[0]["file"]
    flagged = max_similarity >= flag_threshold
    summary = (
        f"Compared with {len(others)} other file(s); highest similarity {max_similarity:.2f} with '{closest}'. "
        + (
            "This is at or above the review threshold and scores 0 originality points; review the pair manually."
            if flagged else
            "This is at or below the low threshold, so originality is full." if max_similarity <= low_threshold else
            f"Partial originality credit: similarity between {low_threshold:.2f} (full credit) and {flag_threshold:.2f} (no credit)."
        )
    )
    return {
        "status": "completed",
        "score": score,
        "score_scale": "0-100",
        "comparison_performed": True,
        "configured_dir": str(corpus_dir),
        "files_compared": len(others),
        "files_skipped": skipped,
        "self_matches_excluded": self_matches,
        "low_threshold": low_threshold,
        "flag_threshold": flag_threshold,
        "shingle_size": shingle_size,
        "max_similarity": max_similarity,
        "most_similar_file": closest,
        "flagged_for_human_review": flagged,
        "per_file_similarity": per_file[:50],
        "pairwise_files": names,
        "pairwise_matrix": matrix,
        "cohort_flagged_pairs": flagged_pairs[:50],
        "summary": summary,
        "warning": "Similarity is a triage signal; it does not establish plagiarism or intent.",
    }
