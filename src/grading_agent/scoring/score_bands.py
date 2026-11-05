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


# Few, clearly separated anchors. A small model cannot reliably tell twenty
# near-synonymous bands ("excellent", "very excellent", "near exemplary") apart,
# which shows up as arbitrary, ceiling-heavy scores.
GENERIC_LEVEL_ANCHORS = (
    ("10", "Fully meets the criterion; no defect found after checking the source and evidence."),
    ("8", "Meets the criterion with one or two minor, low-impact issues."),
    ("6", "Mostly meets it; at least one moderate issue a reviewer would ask to fix."),
    ("4", "Partly meets it; a major issue or several moderate ones."),
    ("2", "Largely unmet; only fragments satisfy the criterion."),
    ("0", "Not met at all, demonstrated by a specific failure (give zero_score_reason)."),
)


def rubric_prompt_block(
    metric_rubrics: dict[str, Any] | None = None,
    metric_caps: dict[str, float] | None = None,
    *,
    levels: dict[str, dict[str, str] | None] | None = None,
) -> str:
    """Build the rating instructions: one integer level 0-10 per metric.

    ``levels`` optionally supplies metric-specific anchors (e.g. {"10": ...,
    "6": ...}) that take precedence over the generic anchors for that metric.
    Odd levels sit between the neighbouring anchors.
    """
    lines = [
        "RATING SCALE: rate each metric with one integer from 0 to 10. Python converts the level to points.",
        "Start from 10 and deduct only for issues you can point to in the source, validated findings, or evidence.",
        "Missing measurements are not failures. Never invent evidence or task requirements.",
        "GENERIC LEVEL ANCHORS:",
    ]
    lines.extend(f"{level}: {text}" for level, text in GENERIC_LEVEL_ANCHORS)
    if metric_caps:
        lines.append("MAXIMUM POINTS BY METRIC (for reference only; still return 0-10 levels):")
        lines.extend(f"{name}: {points:g} points" for name, points in metric_caps.items())
    if metric_rubrics:
        criteria = metric_rubrics.get("metrics", metric_rubrics)
        if isinstance(criteria, dict) and criteria:
            lines.append("METRIC-SPECIFIC CRITERIA:")
            lines.extend(
                f"{name}: {criteria[name]}" for name in sorted(criteria)
                if isinstance(criteria[name], str) and criteria[name].strip()
            )
    specific = {name: value for name, value in (levels or {}).items() if isinstance(value, dict) and value}
    if specific:
        lines.append("METRIC-SPECIFIC LEVEL ANCHORS (use these instead of the generic anchors for that metric):")
        for name in sorted(specific):
            ordered = sorted(specific[name].items(), key=lambda item: -float(item[0]) if item[0].replace(".", "", 1).isdigit() else 0)
            lines.append(f"{name}: " + " | ".join(f"{level} = {text}" for level, text in ordered))
    return "\n".join(lines)
