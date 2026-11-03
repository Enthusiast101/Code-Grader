"""Five-point anchored advisory score bands for LLM calibration.

These bands constrain LLM review recommendations only. Deterministic analyzer
scores remain the sole authority for the final grade.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ScoreBand:
    label: str
    lower: int
    upper: int
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

def score_from_band(value: Any) -> float | None:
    """Return fixed midpoint for a valid half-open 5-point band."""
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

def rubric_prompt_block(metric_rubrics: dict[str, Any] | None = None) -> str:
    """Build the fixed rubric text embedded in every LLM metric review prompt."""
    lines = [
        "ADVISORY SCORE SCALE — choose exactly one 5-point band per metric.",
        "Bands are half-open [lower, upper), except the final band includes 100.",
        "The only allowed labels are: " + ", ".join(BAND_LABELS) + ".",
        "Do not invent a number inside a band. Software converts the chosen band",
        "to its fixed midpoint (2.5, 7.5, ..., 97.5). This is advisory only;",
        "deterministic analyzers calculate the actual grade. Apply metric-specific",
        "criteria and cite evidence IDs. Missing measurements are not zeros.",
        "GENERAL BAND DESCRIPTORS:",
    ]
    lines.extend(f"{b.label}: {b.descriptor}" for b in SCORE_BANDS)
    if metric_rubrics:
        lines.append("METRIC-SPECIFIC CRITERIA:")
        lines.extend(f"{name}: {metric_rubrics[name]}" for name in sorted(metric_rubrics))
    return "\n".join(lines)
