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


@dataclass
class DefectSize:
    """Physical extent of a defect on the road plane, in metres."""
    width_m: float         # across the carriageway
    length_m: float        # along the direction of travel
    distance_m: float      # to the near edge
    reliable: bool         # False if the geometry could not be trusted


def measure_defect(bbox, frame_w: int, frame_h: int,
                   camera_height_m: float = CAMERA_HEIGHT_M,
                   pitch_deg: float = CAMERA_PITCH_DEG) -> DefectSize:
    """Recover a defect's real-world size from its bounding box.

    Apparent size in pixels is useless as a severity measure: the same 0.9 m
    pothole covers 0.7% of the frame at 5 m and 0.003% at 30 m - a factor of
    200. Ranking on it ranks how close the car happened to get.

    Projecting the box onto the road plane removes the dependence. The bottom
    corners give the two ends of the defect's leading edge, so their lateral
    separation is its width across the carriageway; the bottom and top edges
    give the near and far ground distances, whose difference is its length
    along the direction of travel.

    Length is the weaker of the two. Beyond ~20 m a defect spans only one or
    two pixels vertically, so quantisation dominates; width stays good to
    within a few centimetres out to MAX_RANGE_M. When the far edge cannot be
    trusted we fall back to assuming the defect is roughly as long as it is
    wide, which is true of most potholes.
    """
    x1, y1, x2, y2 = bbox
    near_l = image_to_ground(x1, y2, frame_w, frame_h, camera_height_m, pitch_deg)
    near_r = image_to_ground(x2, y2, frame_w, frame_h, camera_height_m, pitch_deg)
    far_c = image_to_ground((x1 + x2) / 2.0, y1, frame_w, frame_h,
                            camera_height_m, pitch_deg)

    if not (near_l.reliable and near_r.reliable):
        return DefectSize(0.0, 0.0, near_l.distance_m, False)

    width = abs(near_r.lateral_m - near_l.lateral_m)
    near_d = (near_l.distance_m + near_r.distance_m) / 2.0

    if far_c.reliable and far_c.distance_m > near_d:
        length = far_c.distance_m - near_d
    else:
        length = width                      # assume roughly circular
    return DefectSize(width, length, near_d, True)


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
