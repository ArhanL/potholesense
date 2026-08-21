"""Estimate a pothole's real-world position from its position in the frame.

Why this matters
----------------
Tagging a detection with the *car's* GPS fix is wrong: the pothole is 5-30 m
ahead of the car, and the fix changes every frame as the car approaches. A
single defect then smears across ~35 m of GPS trace, which either splits into
several phantom potholes or, if you widen the merge radius to compensate,
swallows genuinely distinct defects nearby.

Instead we invert the camera projection under a flat-ground assumption
(inverse perspective mapping). Every sighting of the same pothole then maps to
approximately the same world point, so a tight merge radius works and the
reported coordinate is the defect's location rather than the observer's.

Model
-----
Pinhole camera at height `h` above the road, pitched `p` degrees below
horizontal, no roll. For an image row y:

    theta = p + atan((y - cy) / fy)      # depression angle of the ray
    d     = h / tan(theta)               # ground distance ahead

and for column x, lateral offset (positive = right):

    lat_off = d * (x - cx) / fx

The bottom edge of the bounding box is used, since that is where the defect
meets the road plane. Accuracy degrades with distance (rays near the horizon
are ill-conditioned), so estimates beyond MAX_RANGE_M are rejected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from config import (CAMERA_HEIGHT_M, CAMERA_PITCH_DEG, CAMERA_VFOV_DEG,
                    CAMERA_HFOV_DEG, MAX_RANGE_M, MIN_RANGE_M)

EARTH_RADIUS_M = 6_371_000.0


@dataclass
class GroundPoint:
    distance_m: float      # along the vehicle's heading
    lateral_m: float       # positive = right of centreline
    range_m: float         # straight-line distance from camera
    reliable: bool         # False if near the horizon / out of range


def _focal_lengths(frame_w: int, frame_h: int) -> tuple[float, float]:
    fy = (frame_h / 2.0) / math.tan(math.radians(CAMERA_VFOV_DEG) / 2.0)
    fx = (frame_w / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG) / 2.0)
    return fx, fy


def image_to_ground(x: float, y: float, frame_w: int, frame_h: int,
                    camera_height_m: float = CAMERA_HEIGHT_M,
                    pitch_deg: float = CAMERA_PITCH_DEG) -> GroundPoint:
    """Project an image point onto the road plane."""
    fx, fy = _focal_lengths(frame_w, frame_h)
    cx, cy = frame_w / 2.0, frame_h / 2.0

    theta = math.radians(pitch_deg) + math.atan2(y - cy, fy)
    if theta <= 1e-4:                       # at or above the horizon
        return GroundPoint(float("inf"), 0.0, float("inf"), False)

    d = camera_height_m / math.tan(theta)
    lateral = d * (x - cx) / fx
    rng = math.hypot(d, lateral)
    reliable = MIN_RANGE_M <= d <= MAX_RANGE_M
    return GroundPoint(d, lateral, rng, reliable)


def ground_to_image(distance_m: float, lateral_m: float,
                    frame_w: int, frame_h: int,
                    camera_height_m: float = CAMERA_HEIGHT_M,
                    pitch_deg: float = CAMERA_PITCH_DEG) -> tuple[float, float]:
    """Inverse of image_to_ground - used by the simulator to render a defect
    at a known world position, giving a closed-loop test of the geometry."""
    fx, fy = _focal_lengths(frame_w, frame_h)
    cx, cy = frame_w / 2.0, frame_h / 2.0
    if distance_m <= 0:
        return cx, float(frame_h)
    theta = math.atan2(camera_height_m, distance_m)
    y = cy + fy * math.tan(theta - math.radians(pitch_deg))
    x = cx + fx * lateral_m / distance_m
    return x, y


def offset_latlon(lat: float, lon: float, heading_deg: float,
                  forward_m: float, right_m: float) -> tuple[float, float]:
    """Move a WGS84 point `forward_m` along `heading_deg` and `right_m` to its right."""
    h = math.radians(heading_deg)
    # North/East components: forward along heading, right is heading + 90 deg.
    north = forward_m * math.cos(h) + right_m * math.cos(h + math.pi / 2)
    east = forward_m * math.sin(h) + right_m * math.sin(h + math.pi / 2)
    dlat = math.degrees(north / EARTH_RADIUS_M)
    dlon = math.degrees(east / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2, in degrees from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def locate_detection(bbox, frame_w: int, frame_h: int,
                     car_lat: float, car_lon: float, heading_deg: float | None):
    """Full pipeline: bbox -> estimated pothole (lat, lon, distance, reliable).

    Returns (lat, lon, GroundPoint). If the geometry is unreliable or no
    heading is available, falls back to the car's own position.
    """
    x1, y1, x2, y2 = bbox
    foot_x = (x1 + x2) / 2.0
    foot_y = y2                       # bottom edge sits on the road plane
    gp = image_to_ground(foot_x, foot_y, frame_w, frame_h)

    if not gp.reliable or heading_deg is None:
        return car_lat, car_lon, gp

    lat, lon = offset_latlon(car_lat, car_lon, heading_deg,
                             gp.distance_m, gp.lateral_m)
    return lat, lon, gp
