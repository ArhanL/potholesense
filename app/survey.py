"""Repeat-survey differencing: what changed since last time.

A one-off survey tells a council what is wrong today. Driving the same road
again a month later tells them something more useful: which defects are new,
which are getting worse, and which have been repaired. That is the difference
between a report and a monitoring record, and it is what makes the second
drive worth more than the first.

Three of the four verdicts fall out of the stored detections. The fourth,
`fixed`, is the interesting one, because it is an argument from absence: not
seeing a pothole is only evidence it is gone if you actually drove past where
it was. Every survey therefore logs the vehicle's track, and a defect is only
called fixed when this survey's track passed close to it and nothing was
detected. A defect on a road we did not drive this time is reported as
`not_surveyed`, which is an honest answer rather than a flattering one.
"""
from __future__ import annotations

from statistics import mean, stdev

from app import storage
from app.geo import haversine_m

# How close the vehicle must have passed for "we did not see it" to mean
# anything. Comfortably wider than the localisation error (~1 m) and the
# lane offsets involved, so a genuine miss is a real miss.
PASS_RADIUS_M = 25.0

# What counts as deterioration.
#
# A fixed threshold is the wrong instrument here: 3 cm of growth on a 25 cm
# defect and on a 1 m defect are not equally meaningful, and how meaningful
# either is depends on how precisely that particular defect was measured.
# Each pass produces several independent width measurements, so their spread
# is a direct estimate of the noise. Growth is called real when it exceeds
# both a floor - below which we would be reporting sub-centimetre changes to
# a council - and twice the combined standard error of the two means, which
# is roughly a 95% confidence that the road actually changed.
GROWTH_FLOOR_M = 0.03
GROWTH_SIGMAS = 2.0
ASSUMED_NOISE_M = 0.02      # used when a pass yielded only one measurement


def _mean_and_sem(samples: list[float]) -> tuple[float, float] | None:
    """Mean and standard error of the mean for one survey's measurements."""
    if not samples:
        return None
    m = mean(samples)
    if len(samples) < 2:
        return m, ASSUMED_NOISE_M
    return m, max(stdev(samples), ASSUMED_NOISE_M) / (len(samples) ** 0.5)


def _passed_near(track: list[tuple[float, float]], lat: float, lon: float) -> bool:
    return any(haversine_m(lat, lon, t_lat, t_lon) <= PASS_RADIUS_M
               for t_lat, t_lon in track)


def diff_session(session_id: int) -> dict:
    """Compare one survey against everything recorded before it."""
    track = storage.track_points(session_id)
    potholes = storage.all_potholes()

    buckets: dict[str, list[dict]] = {
        "new": [], "worse": [], "unchanged": [], "fixed": [], "not_surveyed": [],
    }

    for p in potholes:
        seen_in = storage.sessions_that_saw(p["id"])
        earlier = [s for s in seen_in if s < session_id]
        entry = {
            "id": p["id"], "lat": p["lat"], "lon": p["lon"],
            "road_name": p.get("road_name"), "severity": p["severity"],
            "width_m": p.get("width_m"), "sightings": p["sightings"],
        }

        if session_id not in seen_in:
            if not earlier:
                continue                      # belongs to a later survey
            verdict = "fixed" if _passed_near(track, p["lat"], p["lon"]) \
                else "not_surveyed"
            buckets[verdict].append(entry)
            continue

        if not earlier:
            buckets["new"].append(entry)
            continue

        now = _mean_and_sem(storage.width_samples_in_session(p["id"], session_id))
        before = _mean_and_sem(storage.width_samples_in_session(p["id"], earlier[-1]))

        if now is None or before is None:
            buckets["unchanged"].append(entry)
            continue

        (now_w, now_sem), (before_w, before_sem) = now, before
        growth = now_w - before_w
        threshold = max(GROWTH_FLOOR_M,
                        GROWTH_SIGMAS * (now_sem ** 2 + before_sem ** 2) ** 0.5)
        entry.update({
            "previous_width_m": round(before_w, 3),
            "current_width_m": round(now_w, 3),
            "growth_m": round(growth, 3),
            "growth_threshold_m": round(threshold, 3),
        })
        buckets["worse" if growth >= threshold else "unchanged"].append(entry)

    return {
        "session_id": session_id,
        "track_points": len(track),
        "counts": {k: len(v) for k, v in buckets.items()},
        **buckets,
    }
