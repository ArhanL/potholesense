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
                          bearing_between, locate_detection)
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


def test_severity_is_monotonic_in_size():
    assert classify(0.001, 0.9) == "low"
    assert classify(0.05, 0.9) == "medium"
    assert classify(0.15, 0.9) == "high"


def test_repeat_sightings_can_raise_severity():
    a = classify(0.080, 0.9, sightings=1)
    b = classify(0.080, 0.9, sightings=5)
    assert (a, b) == ("medium", "high")


def test_priority_orders_high_before_low():
    assert priority_rank("high", 1) > priority_rank("medium", 10)
    assert priority_rank("medium", 1) > priority_rank("low", 10)


# ------------------------------------------------------------- storage ----
@pytest.fixture()
def db():
    storage.init_db()
    storage.reset()
    yield storage
    storage.reset()


def test_first_detection_creates_pothole(db):
    r = db.record_detection(51.4545, -2.5879, 0.8, 0.05)
    assert r["created"] and r["sightings"] == 1


def test_nearby_detections_merge(db):
    db.record_detection(51.4545, -2.5879, 0.8, 0.05)
    r = db.record_detection(51.45451, -2.58791, 0.85, 0.06)   # ~1.5 m away
    assert not r["created"] and r["sightings"] == 2
    assert db.stats()["potholes"] == 1


def test_distant_detections_stay_separate(db):
    db.record_detection(51.4545, -2.5879, 0.8, 0.05)
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
    r = db.record_detection(51.4545, -2.5879, 0.8, 0.05)
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
