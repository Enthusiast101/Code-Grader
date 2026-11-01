"""Deterministic assignment-mandated usefulness and limitation examples."""
from __future__ import annotations

from typing import Any


def build_usefulness_and_limitations() -> dict[str, list[dict[str, str]]]:
    """Return concrete, appropriately qualified examples for the final report."""
    return {
        "usefulness": [
            {
                "use_case": "Course assignment triage",
                "example": "An instructor evaluates a class set of C submissions against the same versioned rules and hidden tests, then reviews the failing inputs and per-metric evidence.",
                "benefit": "Reduces repetitive compile-and-test work while keeping grades reproducible from the source hash and rules hash.",
            },
            {
                "use_case": "Fast formative feedback",
                "example": "A student submits a program before the deadline and receives exact failing inputs, expected outputs, compiler diagnostics, and bounded runtime findings.",
                "benefit": "Helps students reproduce and fix defects before manual assessment.",
            },
            {
                "use_case": "Security and memory-risk screening",
                "example": "A course uses Valgrind, ASan/UBSan, static diagnostics, and source review to prioritize submissions that need a closer manual check.",
                "benefit": "Combines independent runtime and source-level signals instead of relying on output correctness alone.",
            },
            {
                "use_case": "Similarity triage across a course corpus",
                "example": "With an instructor-authorized corpus populated, the similarity analyzer flags submissions whose token-shingle overlap exceeds the configured threshold.",
                "benefit": "Narrows the set of submissions an instructor needs to inspect; it does not determine plagiarism or authorship.",
            },
        ],
        "limitations": [
            {
                "limitation": "Finite tests cannot prove correctness",
                "example": "A sum program passes every supplied case but receives 2147483647 and 1 when no overflow case is configured; the program may invoke signed integer overflow.",
                "mitigation": "Define the numeric contract in rules, include boundary/counterexample tests, and state exactly which inputs were exercised.",
            },
            {
                "limitation": "Dynamic tools only observe executed paths",
                "example": "A memory leak occurs only after a rare error branch; if none of the configured tests enters that branch, Valgrind may report a clean run.",
                "mitigation": "Expand tests to exercise error paths and retain source-level review as a complementary, non-conclusive signal.",
            },
            {
                "limitation": "Heuristic and LLM judgments can be subjective",
                "example": "A short, clear program uses single-letter loop indices; a style heuristic may penalize it even when the instructor rubric does not require longer names or comments.",
                "mitigation": "Keep the instructor rubric authoritative, classify non-rubric suggestions as optional, and require evidence IDs for factual claims.",
            },
            {
                "limitation": "Similarity is not proof of copying or LLM use",
                "example": "Two students independently write the same short scanf/printf solution and exceed a similarity threshold because the task has very little possible code structure.",
                "mitigation": "Use similarity only to flag candidates for instructor review; compare distinctive logic and context before drawing conclusions.",
            },
        ],
    }
