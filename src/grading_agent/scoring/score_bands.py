"""Calibrated score bands for 0..10 LLM ratings and 0..100 final scores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoreBand:
    label: str
    lower: float
    upper: float
    midpoint: float
    descriptor: str


_DESCRIPTORS = (
    "No usable evidence of the required behavior; essentially absent or wholly invalid.",
    "Almost entirely unsuccessful; only a negligible fragment is demonstrated.",
    "Severe failure; a few isolated elements work, but the requirement is largely unmet.",
    "Very poor; major defects dominate and only limited behavior is correct.",
    "Poor; substantial defects remain across most of the criterion.",
    "Weak; some relevant behavior exists, but important requirements are missed.",
    "Below basic; partial success with several major omissions or failures.",
    "Limited; core behavior is recognizable but materially incomplete or fragile.",
    "Low pass; essential parts work, with significant gaps still present.",
    "Marginal; more than half the criterion is demonstrated, but key weaknesses remain.",
    "Adequate; the main requirement is met, though notable weaknesses remain.",
    "Satisfactory; most requirements are met, with several non-trivial defects.",
    "Fairly good; generally correct, with clear but bounded shortcomings.",
    "Good; requirements are substantially met and remaining issues are moderate.",
    "Strong; only a small number of meaningful issues remain.",
    "Very good; nearly all requirements are satisfied with minor shortcomings.",
    "Excellent; complete in the important areas with only very small issues.",
    "Very excellent; robust and well-supported, with negligible gaps.",
    "Near exemplary; virtually all requirements are met and evidence is comprehensive.",
    "Exemplary; fully satisfies the criterion with complete, consistent evidence.",
)
SCORE_BANDS = tuple(
    ScoreBand(f"{lower}-{lower + 5}", lower, lower + 5, lower + 2.5, _DESCRIPTORS[index])
    for index, lower in enumerate(range(0, 100, 5))
)
BAND_BY_LABEL = {band.label: band for band in SCORE_BANDS}
BAND_LABELS = tuple(BAND_BY_LABEL)

LLM_SCORE_BANDS = tuple(
    ScoreBand(
        f"{lower:g}-{lower + 0.5:g}", lower, lower + 0.5,
        lower + 0.25, _DESCRIPTORS[index],
    )
    for index, lower in enumerate(i * 0.5 for i in range(20))
)
LLM_BAND_BY_LABEL = {band.label: band for band in LLM_SCORE_BANDS}


def score_from_band(value: Any) -> float | None:
    """Return the fixed midpoint for a recognized five-point band."""
    if not isinstance(value, str):
        return None
    band = BAND_BY_LABEL.get(value.strip())
    return band.midpoint if band else None


def band_for_score(score: float) -> str:
    """Map a finite 0..100 score to a display band; 100 maps to 95-100."""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("score must be numeric")
    numeric = float(score)
    if not 0.0 <= numeric <= 100.0:
        raise ValueError("score must be between 0 and 100")
    lower = min(95, int(numeric // 5) * 5)
    return f"{lower}-{lower + 5}"


def llm_score_from_band(value: Any) -> float | None:
    """Return a fixed midpoint for a recognized 0..10 LLM rating band."""
    if not isinstance(value, str):
        return None
    band = LLM_BAND_BY_LABEL.get(value.strip())
    return band.midpoint if band else None


def llm_band_for_score(score: float) -> str:
    """Map a finite 0..10 LLM score to a half-point descriptor band."""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("score must be numeric")
    numeric = float(score)
    if not 0.0 <= numeric <= 10.0:
        raise ValueError("LLM score must be between 0 and 10")
    lower = min(9.5, int(numeric * 2) / 2)
    return f"{lower:g}-{lower + 0.5:g}"


def rubric_prompt_block(metric_rubrics: dict[str, Any] | None = None) -> str:
    """Build the calibrated scale and metric criteria for scoring prompts."""
    lines = [
        "CALIBRATED LLM METRIC RATING: score every metric from 0 to 10, allowing decimals.",
        "Return a numeric score from 0 to 10; do not return a band label.",
        "Choose scores consistently with the nearest band descriptor and metric-specific rubric.",
        "Provide a 0-10 assessment for every enabled metric. Executed correctness/logical test pass rates remain authoritative for the final score when available; retain the LLM rating as a separate assessment.",
        "Missing measurements are not zeros. Never invent evidence or task requirements.",
        "GENERAL 0-10 ANCHORS (each band is a half point):",
    ]
    lines.extend(f"{band.label}/10: {band.descriptor}" for band in LLM_SCORE_BANDS)
    if metric_rubrics:
        criteria = metric_rubrics.get("metrics", metric_rubrics)
        if isinstance(criteria, dict) and criteria:
            lines.append("METRIC-SPECIFIC CRITERIA:")
            lines.extend(f"{name}: {criteria[name]}" for name in sorted(criteria))
    return "\n".join(lines)
