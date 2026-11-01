"""Compact, consistent one-to-two sentence score explanations."""
from __future__ import annotations

import re
from typing import Any


def compact_rationale(value: Any, *, max_words: int = 18, max_chars: int = 120) -> str:
    """Flatten a score explanation to at most two short sentences."""
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(sentences[:2]).strip()
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(" ,;:") + "…"
    if len(text) > max_chars:
        text = text[: max_chars - 1].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return text
