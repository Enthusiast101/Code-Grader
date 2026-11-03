from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grading_agent.scoring.score_bands import BAND_LABELS, band_for_score, score_from_band


def test_twenty_five_point_bands_are_discrete_and_midpoint_anchored():
    assert len(BAND_LABELS) == 20
    assert BAND_LABELS[0] == "0-5"
    assert BAND_LABELS[-1] == "95-100"
    assert score_from_band("0-5") == 2.5
    assert score_from_band("5-10") == 7.5
    assert score_from_band("95-100") == 97.5
    assert score_from_band("0-6") is None


def test_continuous_deterministic_score_maps_to_single_display_band():
    assert band_for_score(0) == "0-5"
    assert band_for_score(4.999) == "0-5"
    assert band_for_score(5) == "5-10"
    assert band_for_score(99.99) == "95-100"
    assert band_for_score(100) == "95-100"
