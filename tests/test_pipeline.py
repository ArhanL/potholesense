"""Unit tests for the geometry, severity and deduplication logic.

Run with:  python -m pytest tests/ -v
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.geo import haversine_m, bbox_around
from app.localise import (image_to_ground, ground_to_image, offset_latlon,
                          bearing_between, locate_detection, measure_defect)
from app.severity import bbox_area_fraction, classify, priority_rank
from app import storage, baseline
from config import MAX_RANGE_M


# ------------------------------------------------------------------ geo ----
def test_haversine_known_distance():
    # London -> Oxford is about 82.6 km
    d = haversine_m(51.5074, -0.1278, 51.7520, -1.2577)
    assert 82_000 < d < 83_500


def test_haversine_zero():
    assert haversine_m(51.5, -0.1, 51.5, -0.1) == pytest.approx(0.0, abs=1e-6)


def test_bbox_around_contains_point():
    lat, lon, r = 51.4545, -2.5879, 50.0
    min_lat, max_lat, min_lon, max_lon = bbox_around(lat, lon, r)
    assert min_lat < lat < max_lat and min_lon < lon < max_lon
    # A point r metres due north must fall inside the box.
    north_lat = lat + r / 111_320.0
    assert min_lat <= north_lat <= max_lat


# ------------------------------------------------------------ localise ----
@pytest.mark.parametrize("distance", [3.0, 5.0, 10.0, 20.0, 30.0])
def test_projection_round_trip_distance(distance):
    W, H = 640, 480
    x, y = ground_to_image(distance, 0.0, W, H)
    gp = image_to_ground(x, y, W, H)
    assert gp.distance_m == pytest.approx(distance, rel=1e-6)


@pytest.mark.parametrize("lateral", [-2.0, -0.5, 0.0, 1.5])
def test_projection_round_trip_lateral(lateral):
    W, H = 640, 480
    x, y = ground_to_image(12.0, lateral, W, H)
    gp = image_to_ground(x, y, W, H)
    assert gp.lateral_m == pytest.approx(lateral, abs=1e-6)


def test_points_above_horizon_are_unreliable():
    gp = image_to_ground(320, 5, 640, 480)     # near the top of the frame
    assert not gp.reliable


def test_beyond_max_range_is_unreliable():
    W, H = 640, 480
    x, y = ground_to_image(MAX_RANGE_M + 20, 0.0, W, H)
    gp = image_to_ground(x, y, W, H)
    assert not gp.reliable


def test_offset_latlon_due_north():
    lat, lon = offset_latlon(51.0, -2.0, 0.0, 111.32, 0.0)
    assert lat == pytest.approx(51.001, abs=1e-4)
    assert lon == pytest.approx(-2.0, abs=1e-6)


def test_offset_then_bearing_is_consistent():
    for heading in (0.0, 45.0, 137.0, 300.0):
        lat, lon = offset_latlon(51.4545, -2.5879, heading, 100.0, 0.0)
        back = bearing_between(51.4545, -2.5879, lat, lon)
        assert back == pytest.approx(heading, abs=0.5)


def test_locate_detection_places_pothole_ahead_not_at_car():
    """The whole point of the geometry: report where the defect is."""
    W, H = 640, 480
    x, y = ground_to_image(20.0, 0.0, W, H)
    bbox = (x - 10, y - 5, x + 10, y)
    car = (51.4545, -2.5879)
    lat, lon, gp = locate_detection(bbox, W, H, *car, heading_deg=0.0)
    assert gp.reliable
    # ~20 m from the car, and to the north since heading is 0.
    assert haversine_m(*car, lat, lon) == pytest.approx(20.0, abs=1.5)
    assert lat > car[0]


def test_locate_detection_falls_back_without_heading():
    W, H = 640, 480
    x, y = ground_to_image(15.0, 0.0, W, H)
    lat, lon, _ = locate_detection((x-10, y-5, x+10, y), W, H,
                                   51.4545, -2.5879, heading_deg=None)
    assert (lat, lon) == (51.4545, -2.5879)


# ------------------------------------------------------------ severity ----
def test_bbox_area_fraction():
    assert bbox_area_fraction((0, 0, 320, 240), 640, 480) == pytest.approx(0.25)


# ------------------------------------------------------- defect sizing ----
def _synthetic_bbox(distance_m, half_width_m, W=640, H=480):
    """Bounding box of a defect of known size at a known distance."""
    cx, _ = ground_to_image(distance_m, 0.0, W, H)
    xr, _ = ground_to_image(distance_m, half_width_m, W, H)
    _, y_near = ground_to_image(max(distance_m - half_width_m, 0.5), 0.0, W, H)
    _, y_far = ground_to_image(distance_m + half_width_m, 0.0, W, H)
    return (cx - abs(xr - cx), min(y_near, y_far), cx + abs(xr - cx),
            max(y_near, y_far))


@pytest.mark.parametrize("distance", [5.0, 10.0, 18.0, 25.0])
def test_measured_width_is_independent_of_distance(distance):
    """The whole point: the same defect must measure the same size whether it
    is 5 m or 25 m ahead. Apparent size in pixels varies ~200x over that
    range, which is why it cannot be used for severity."""
    bbox = _synthetic_bbox(distance, 0.45)
    size = measure_defect(bbox, 640, 480)
    assert size.reliable
    assert size.width_m == pytest.approx(0.90, abs=0.12)


def test_apparent_size_would_have_varied_wildly():
    """Guards the reason the rewrite happened - if this ever stops being true
    the frame-fraction approach was not as broken as claimed."""
    near = bbox_area_fraction(_synthetic_bbox(5.0, 0.45), 640, 480)
    far = bbox_area_fraction(_synthetic_bbox(30.0, 0.45), 640, 480)
    assert near / far > 50


def test_unmeasurable_box_is_not_reliable():
    size = measure_defect((300, 2, 340, 6), 640, 480)   # up near the horizon
    assert not size.reliable


# ------------------------------------------------------------ severity ----
def test_severity_bands_on_measured_width():
    assert classify(0.15, 0.15) == "low"        # 15 cm - below intervention
    assert classify(0.35, 0.35) == "medium"     # meets the 300 mm criterion
    assert classify(0.80, 0.80) == "high"


def test_severity_is_monotonic_in_size():
    order = {"low": 0, "medium": 1, "high": 2}
    widths = [0.10, 0.25, 0.31, 0.45, 0.61, 1.20]
    scores = [order[classify(w, w)] for w in widths]
    assert scores == sorted(scores)


def test_elongated_defect_rated_on_its_longer_axis():
    assert classify(0.20, 0.90) == "high"


def test_unreliable_geometry_returns_unknown_not_a_guess():
    assert classify(None, None, reliable=False) == "unknown"
    assert classify(0.8, 0.8, distance_m=200.0) == "unknown"


def test_severity_no_longer_depends_on_confidence():
    """Confidence is about the detector, not about the road. It may reorder
    the report but must not change what a defect is classified as."""
    assert classify(0.35, 0.35) == classify(0.35, 0.35)
    assert priority_rank("medium", 1, 0.99) > priority_rank("medium", 1, 0.10)


def test_priority_orders_high_before_low():
    assert priority_rank("high", 1) > priority_rank("medium", 10)
    assert priority_rank("medium", 1) > priority_rank("low", 10)


def test_unknown_severity_ranks_last():
    assert priority_rank("unknown", 10) < priority_rank("low", 1)


# ------------------------------------------------------------- storage ----
@pytest.fixture()
def db():
    storage.init_db()
    storage.reset()
    yield storage
    storage.reset()


def test_first_detection_creates_pothole(db):
    r = db.record_detection(51.4545, -2.5879, 0.8, 0.05, width_m=0.35, length_m=0.35)
    assert r["created"] and r["sightings"] == 1


def test_nearby_detections_merge(db):
    db.record_detection(51.4545, -2.5879, 0.8, 0.05, width_m=0.35, length_m=0.35)
    r = db.record_detection(51.45451, -2.58791, 0.85, 0.06, width_m=0.36, length_m=0.36)   # ~1.5 m away
    assert not r["created"] and r["sightings"] == 2
    assert db.stats()["potholes"] == 1


def test_distant_detections_stay_separate(db):
    db.record_detection(51.4545, -2.5879, 0.8, 0.05, width_m=0.35, length_m=0.35)
    r = db.record_detection(51.4600, -2.5879, 0.8, 0.05)      # ~610 m away
    assert r["created"]
    assert db.stats()["potholes"] == 2


def test_merging_averages_position(db):
    db.record_detection(51.45450, -2.5879, 0.8, 0.05)
    r = db.record_detection(51.45452, -2.5879, 0.8, 0.05)
    assert r["lat"] == pytest.approx(51.45451, abs=1e-5)


def test_merge_keeps_highest_confidence(db):
    db.record_detection(51.4545, -2.5879, 0.60, 0.05)
    r = db.record_detection(51.4545, -2.5879, 0.91, 0.05)
    assert r["max_conf"] == pytest.approx(0.91)
    r = db.record_detection(51.4545, -2.5879, 0.40, 0.05)
    assert r["max_conf"] == pytest.approx(0.91)


def test_a_full_pass_collapses_to_one_pothole(db):
    """12 frames of the same defect must not become 12 reports."""
    import random
    rng = random.Random(0)
    for _ in range(12):
        db.record_detection(51.4545 + rng.uniform(-2, 2)/111_320,
                            -2.5879 + rng.uniform(-2, 2)/111_320,
                            0.8, 0.05, speed_mps=13.4)
    s = db.stats()
    assert s["potholes"] == 1 and s["detections"] == 12


def test_mark_reported(db):
    r = db.record_detection(51.4545, -2.5879, 0.8, 0.05, width_m=0.35, length_m=0.35)
    assert db.mark_reported([r["id"]]) == 1
    assert db.all_potholes(status="new") == []
    assert len(db.all_potholes(status="reported")) == 1


# ------------------------------------------------------------ baseline ----
def test_baseline_ignores_blank_road():
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    assert baseline.detect(frame) == []


def test_baseline_finds_a_dark_blob():
    import cv2
    frame = np.full((480, 640, 3), 130, dtype=np.uint8)
    cv2.ellipse(frame, (320, 380), (55, 30), 0, 0, 360, (30, 30, 30), -1)
    dets = baseline.detect(frame)
    assert dets, "expected the baseline to fire on an obvious pothole"
    (x1, y1, x2, y2), conf = dets[0]
    assert x1 < 320 < x2 and y1 < 380 < y2
    assert 0.0 < conf <= 1.0
