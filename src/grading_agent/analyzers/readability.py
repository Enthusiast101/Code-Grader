"""Deterministic source-readability heuristics."""
from __future__ import annotations
from typing import Any
from grading_agent.analyzers.source_metrics import analyze_source


def run_readability(
    source_code: str,
    max_line_length: int = 100,
    long_line_penalty: float = 1.0,
    single_char_identifier_penalty: float = 1.0,
    missing_comment_penalty: float = 5.0,
) -> dict[str, Any]:
    metrics = analyze_source(source_code)
    long_lines = [i for i, line in enumerate(source_code.splitlines(), 1) if len(line) > max_line_length]
    identifiers = metrics["identifier_count"]
    single_char = metrics["single_character_identifier_count"]
    penalty = len(long_lines) * long_line_penalty + single_char * single_char_identifier_penalty
    if metrics["nonblank_line_count"] >= 8 and metrics["comment_line_count"] == 0:
        penalty += missing_comment_penalty
    score = max(0.0, 100.0 - penalty)
    return {
        "status": "completed",
        "long_lines": long_lines,
        "comment_line_count": metrics["comment_line_count"],
        "identifier_count": identifiers,
        "single_character_identifier_count": single_char,
        "average_line_length": metrics["average_line_length"],
        "score": round(score, 2),
        "score_scale": "0-100",
        "measurement_note": "Deterministic style heuristics; naming and comments require contextual human/LLM judgment.",
    }
