"""Quality anchors for mapping metric evidence to configured point caps."""
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


def rubric_prompt_block(
    metric_rubrics: dict[str, Any] | None = None,
    metric_caps: dict[str, float] | None = None,
) -> str:
    """Build a rubric prompt for consolidated, metric-specific point scores."""
    lines = [
        "CONSOLIDATED METRIC POINTS: return the earned points directly for each metric, using that metric's maximum below.",
        "Do not return a normalized rating or a separate LLM rating. A returned metric score is already points earned out of its configured maximum.",
        "The overall grade is the sum of earned points and is out of 100. Diagnostic-only metrics are excluded.",
        "Executed correctness/logical test results and completed deterministic analyzer results remain authoritative when available; model points are used for other metrics when evidence supports them.",
        "Missing measurements are not zeros. Never invent evidence or task requirements.",
        "GENERAL QUALITY ANCHORS (bands are percentages of each metric's point maximum):",
    ]
    lines.extend(f"{band.label}%: {band.descriptor}" for band in SCORE_BANDS)
    if metric_caps:
        lines.append("MAXIMUM POINTS BY METRIC:")
        lines.extend(f"{name}: 0 to {points:g} points" for name, points in metric_caps.items())
    if metric_rubrics:
        criteria = metric_rubrics.get("metrics", metric_rubrics)
        if isinstance(criteria, dict) and criteria:
            lines.append("METRIC-SPECIFIC CRITERIA:")
            lines.extend(f"{name}: {criteria[name]}" for name in sorted(criteria))
    return "\n".join(lines)
