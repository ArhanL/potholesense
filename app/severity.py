"""Severity scoring for a detected pothole.

Severity is banded on the defect's **measured width in metres**, recovered by
projecting its bounding box onto the road plane, not on how large it happens
to look in the frame.

That distinction is the whole point. Apparent size in pixels varies by a
factor of roughly 200 between a defect 5 m ahead and the same defect 30 m
ahead, so banding on it produces a ranking of how close the car got, not of
how bad the road is. Projected width is stable to a few centimetres across
the usable range, which makes it comparable between sightings, between
drives, and against a council's own published intervention criteria.

What this still cannot do is measure depth, which a single monocular frame
does not contain. Width is therefore a partial criterion, not the whole one:
a wide shallow patch and a narrow deep hole are not distinguished. That
limitation is stated rather than hidden, because a council prioritising work
has to be able to justify the ordering.
"""
from __future__ import annotations
from config import SEVERITY_WIDTH_BANDS_M, SEVERITY_MAX_RANGE_M


def bbox_area_fraction(bbox, frame_w: int, frame_h: int) -> float:
    """Fraction of the frame covered by the box.

    Retained as diagnostic metadata only - it is recorded alongside each
    detection but deliberately no longer drives severity.
    """
    x1, y1, x2, y2 = bbox
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    return area / float(max(frame_w * frame_h, 1))


def classify(width_m: float | None, length_m: float | None = None,
             distance_m: float | None = None, reliable: bool = True) -> str:
    """Band a defect by its measured width. Returns 'low' | 'medium' | 'high'.

    An unmeasurable or too-distant sighting returns 'unknown' rather than a
    guess: it still counts as a sighting and still refines the position, but
    it must not be allowed to invent a severity.
    """
    if not reliable or width_m is None or width_m <= 0.0:
        return "unknown"
    if distance_m is not None and distance_m > SEVERITY_MAX_RANGE_M:
        return "unknown"

    # A defect longer than it is wide is at least that bad; take the larger
    # horizontal dimension so an elongated crack is not under-rated.
    extent = max(width_m, length_m or 0.0)
    if extent >= SEVERITY_WIDTH_BANDS_M["high"]:
        return "high"
    if extent >= SEVERITY_WIDTH_BANDS_M["medium"]:
        return "medium"
    return "low"


# Ordering for the council report. Severity comes from the physical
# measurement; confidence and corroboration affect only how far up the list a
# defect sits within its band, never what it is classified as.
#
# The bands are spaced 100 apart and the modifiers are capped at 25 in total,
# so no amount of corroboration can lift a defect past one that is physically
# worse. That property is what makes the ordering explainable to a council:
# the band always dominates, and the modifiers only break ties inside it.
_BASE = {"high": 300.0, "medium": 200.0, "low": 100.0, "unknown": 0.0}
_MAX_CORROBORATION = 20.0
_MAX_CONFIDENCE_BONUS = 5.0


def priority_rank(severity: str, sightings: int, confidence: float = 1.0) -> float:
    """Higher = fix sooner."""
    base = _BASE.get(severity, 0.0)
    corroboration = min(max(sightings, 0), 10) * (_MAX_CORROBORATION / 10.0)
    certainty = _MAX_CONFIDENCE_BONUS * max(0.0, min(confidence, 1.0))
    return base + corroboration + certainty
