"""Token-shingle similarity against an instructor-provided source corpus.

A similarity flag is a review signal, not a plagiarism verdict.
"""
from __future__ import annotations

import hashlib
import re
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


def run_similarity_flag(
    source_code: str,
    corpus_dir: str | Path,
    flag_threshold: float = settings.ANALYZER_SIMILARITY_THRESHOLD,
    max_files: int = settings.ANALYZER_SIMILARITY_MAX_FILES,
    shingle_size: int = settings.ANALYZER_SIMILARITY_SHINGLE_SIZE,
    max_file_bytes: int = settings.ANALYZER_SIMILARITY_MAX_FILE_BYTES,
    exclude_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(corpus_dir)
    if not root.is_dir():
        return {"status": "inconclusive", "score": None, "error": f"Similarity corpus directory not found: {root}"}
    if not 0 <= flag_threshold <= 1 or shingle_size < 2 or max_files < 1:
        raise ValueError("Invalid similarity analyzer parameters.")

    candidate_tokens = _tokens(source_code)
    candidate_shingles = _shingles(candidate_tokens, shingle_size)
    candidate_hash = hashlib.sha256(" ".join(candidate_tokens).encode()).hexdigest()
    matches: list[dict[str, Any]] = []
    compared = skipped = 0

    excluded = Path(exclude_path).resolve() if exclude_path else None
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".c", ".h"})
    for path in files:
        if excluded is not None and path.resolve() == excluded:
            continue
        if compared >= max_files:
            break
        try:
            if path.stat().st_size > max_file_bytes:
                skipped += 1
                continue
            other_source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            skipped += 1
            continue
        other_tokens = _tokens(other_source)
        other_hash = hashlib.sha256(" ".join(other_tokens).encode()).hexdigest()
        other_shingles = _shingles(other_tokens, shingle_size)
        score = 1.0 if candidate_hash == other_hash else _similarity(candidate_shingles, other_shingles)
        compared += 1
        if score >= flag_threshold:
            matches.append({"file": str(path.relative_to(root)), "similarity": round(score, 4), "exact_normalized_match": candidate_hash == other_hash})

    matches.sort(key=lambda item: item["similarity"], reverse=True)
    if compared == 0:
        return {"status": "inconclusive", "score": None, "corpus_files_compared": 0, "files_skipped": skipped, "error": "No comparable C or header files found in the similarity corpus."}
    max_similarity = max((item["similarity"] for item in matches), default=0.0)
    flagged = max_similarity >= flag_threshold
    return {
        "status": "completed",
        "corpus_files_compared": compared,
        "corpus_files_available": len(files),
        "scan_limit_reached": len(files) > compared,
        "files_skipped": skipped,
        "flag_threshold": flag_threshold,
        "max_similarity": round(max_similarity, 4),
        "flagged_for_human_review": flagged,
        "matches": matches[:20],
        "score": round(max(0.0, 100.0 * (1.0 - max_similarity)), 2),
        "score_scale": "0-100",
        "warning": "Similarity is a triage signal; it does not establish plagiarism or intent.",
    }
