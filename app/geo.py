"""Geospatial helpers."""
from math import radians, sin, cos, asin, sqrt

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points, in metres."""
    p1, p2 = radians(lat1), radians(lat2)
    dphi = p2 - p1
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(a))


def bbox_around(lat: float, lon: float, radius_m: float):
    """Cheap lat/lon bounding box for pre-filtering DB queries."""
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * max(cos(radians(lat)), 1e-6))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon
