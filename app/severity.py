"""Severity scoring for a detected pothole.

We cannot measure true depth from a single monocular frame, so severity is a
*proxy*: how large the pothole appears relative to the frame, weighted by
detector confidence and by how many independent times we have seen it.

This is deliberately explainable - a council needs to justify prioritisation.
"""
from config import SEVERITY_BANDS


def bbox_area_fraction(bbox, frame_w: int, frame_h: int) -> float:
    x1, y1, x2, y2 = bbox
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    return area / float(max(frame_w * frame_h, 1))


def classify(area_fraction: float, confidence: float, sightings: int = 1) -> str:
    """Return 'low' | 'medium' | 'high'."""
    score = area_fraction * (0.6 + 0.4 * confidence)
    # Repeated independent sightings increase our certainty, nudging severity up.
    if sightings >= 3:
        score *= 1.15
    if score >= SEVERITY_BANDS["high"]:
        return "high"
    if score >= SEVERITY_BANDS["medium"]:
        return "medium"
    return "low"


def priority_rank(severity: str, sightings: int) -> float:
    """Higher = fix sooner. Used to order the council report."""
    base = {"high": 100.0, "medium": 50.0, "low": 10.0}[severity]
    return base + min(sightings, 10) * 2.0
