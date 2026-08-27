"""Reverse geocoding: turn a coordinate into a road name.

A council report that says "51.454502, -2.587903" makes a highways officer do
the work of finding the street. One that says "Whiteladies Road, Bristol" can
be actioned. This is the cheapest credibility improvement in the whole
pipeline.

Design constraints, in order of importance:

* **Never block a survey.** Lookups happen after the drive, on demand, not on
  the frame path. A pothole with no road name is still a valid pothole.
* **Respect the service.** Nominatim's usage policy allows at most one request
  per second and requires an identifying User-Agent. Both are enforced here
  rather than left to the caller's good intentions.
* **Ask once per road, not once per pothole.** Results are cached in the
  database, and coordinates are snapped to a coarse grid before lookup so
  several defects along the same stretch share one request.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.parse
import urllib.request

log = logging.getLogger("potholesense.geocode")

ENDPOINT = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "PotholeSense/1.0 (road defect survey; https://github.com/ArhanL/potholesense)"

MIN_INTERVAL_S = 1.1        # Nominatim allows 1 req/s; leave headroom.
TIMEOUT_S = 8.0
GRID_DEG = 0.0005           # ~55 m: defects this close share a lookup.

_lock = threading.Lock()
_last_call = 0.0
_memo: dict[tuple[float, float], str | None] = {}


def _snap(lat: float, lon: float) -> tuple[float, float]:
    return (round(lat / GRID_DEG) * GRID_DEG, round(lon / GRID_DEG) * GRID_DEG)


def _describe(address: dict) -> str | None:
    """Best available human description, most specific part first."""
    road = (address.get("road") or address.get("pedestrian")
            or address.get("footway") or address.get("residential"))
    area = (address.get("suburb") or address.get("village")
            or address.get("town") or address.get("city")
            or address.get("county"))
    if road and area:
        return f"{road}, {area}"
    return road or area


def road_name(lat: float, lon: float) -> str | None:
    """Return a road description, or None if unavailable.

    Never raises: no network, a rate limit or a malformed reply all mean the
    report simply carries a coordinate, exactly as it did before.
    """
    global _last_call
    key = _snap(lat, lon)
    if key in _memo:
        return _memo[key]

    query = urllib.parse.urlencode({
        "lat": f"{lat:.6f}", "lon": f"{lon:.6f}",
        "format": "jsonv2", "zoom": 17, "addressdetails": 1,
    })
    req = urllib.request.Request(f"{ENDPOINT}?{query}",
                                 headers={"User-Agent": USER_AGENT})
    try:
        with _lock:
            wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.monotonic()
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                payload = json.loads(r.read().decode("utf-8"))
        name = _describe(payload.get("address") or {})
    except Exception as exc:                                   # noqa: BLE001
        log.info("Reverse geocode failed for %.5f,%.5f (%s)", lat, lon, exc)
        return None

    _memo[key] = name
    return name
