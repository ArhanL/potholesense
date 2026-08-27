"""SQLite persistence with GPS-based deduplication.

Design note: a car passing a pothole at 30mph produces ~10-20 frames containing
the same defect. Writing each as a separate report would flood the council with
duplicates, so every raw detection is matched against existing potholes within
DEDUPE_RADIUS_M and merged if found.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from config import (DB_PATH, DEDUPE_RADIUS_M, TRACK_TIMEOUT_S,
                    TRACK_MAX_RADIUS_M, DEFAULT_SPEED_MPS, EVIDENCE_DIR)
from app.geo import haversine_m, bbox_around
from app import severity as sev

SCHEMA = """
CREATE TABLE IF NOT EXISTS potholes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    lat           REAL NOT NULL,
    lon           REAL NOT NULL,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    sightings     INTEGER NOT NULL DEFAULT 1,
    max_conf      REAL NOT NULL,
    area_fraction REAL NOT NULL,
    width_m       REAL,
    length_m      REAL,
    severity      TEXT NOT NULL,
    road_name     TEXT,
    evidence      TEXT,
    status        TEXT NOT NULL DEFAULT 'new',
    session_id    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_potholes_latlon ON potholes(lat, lon);

CREATE TABLE IF NOT EXISTS detections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    pothole_id    INTEGER NOT NULL REFERENCES potholes(id),
    ts            TEXT NOT NULL,
    lat           REAL NOT NULL,
    lon           REAL NOT NULL,
    confidence    REAL NOT NULL,
    area_fraction REAL NOT NULL,
    width_m       REAL,
    length_m      REAL,
    distance_m    REAL,
    accuracy_m    REAL,
    speed_mps     REAL,
    session_id    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_detections_session ON detections(session_id);
CREATE INDEX IF NOT EXISTS idx_detections_pothole ON detections(pothole_id);

-- Where the vehicle actually went, so a later survey can tell "this defect
-- was not detected" (it may be gone) from "we never drove past it".
CREATE TABLE IF NOT EXISTS track (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    ts         TEXT NOT NULL,
    lat        REAL NOT NULL,
    lon        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_track_session ON track(session_id);

CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    device     TEXT,
    frames     INTEGER NOT NULL DEFAULT 0,
    notes      TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@contextmanager
def connect():
    """Open a connection, commit on success, and always close it.

    sqlite3's own context manager commits or rolls back but does NOT close,
    so the previous `with connect() as conn` leaked a file handle on every
    call - thousands of them over a survey. This wraps both.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


# Columns added after the first release. Applied to existing databases so an
# earlier survey keeps working instead of failing on a missing column.
_MIGRATIONS = {
    "potholes": {"width_m": "REAL", "length_m": "REAL", "road_name": "TEXT"},
    "detections": {"width_m": "REAL", "length_m": "REAL", "distance_m": "REAL",
                   "session_id": "INTEGER"},
}


def _migrate(conn) -> None:
    for table, columns in _MIGRATIONS.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns.items():
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def start_session(device: str = "unknown") -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (started_at, device) VALUES (?, ?)", (_now(), device)
        )
        return cur.lastrowid


def end_session(session_id: int) -> None:
    with connect() as conn:
        conn.execute("UPDATE sessions SET ended_at=? WHERE id=?", (_now(), session_id))


def bump_frames(session_id: int, n: int = 1) -> None:
    with connect() as conn:
        conn.execute("UPDATE sessions SET frames = frames + ? WHERE id=?", (n, session_id))


def _delete_evidence(name: str) -> None:
    """Remove a superseded evidence image; never raise if it is already gone."""
    try:
        (EVIDENCE_DIR / Path(name).name).unlink(missing_ok=True)
    except OSError:
        pass


def _age_seconds(iso_ts: str) -> float:
    try:
        then = datetime.fromisoformat(iso_ts)
    except ValueError:
        return float("inf")
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds()


def _association_radius(speed_mps: float | None) -> float:
    """Widen the merge radius while a sighting run is still in progress."""
    speed = speed_mps if speed_mps and speed_mps > 0 else DEFAULT_SPEED_MPS
    return min(max(DEDUPE_RADIUS_M, speed * TRACK_TIMEOUT_S), TRACK_MAX_RADIUS_M)


def _find_nearby(conn, lat: float, lon: float, speed_mps: float | None = None):
    """Find the pothole this detection belongs to, or None.

    Two-tier matching:
      * within DEDUPE_RADIUS_M            -> same pothole, whenever it was seen
      * within the wider track radius     -> same pothole only if it was seen
                                             within TRACK_TIMEOUT_S (i.e. we are
                                             mid-pass, still approaching it)
    """
    track_radius = _association_radius(speed_mps)
    min_lat, max_lat, min_lon, max_lon = bbox_around(lat, lon, track_radius)
    rows = conn.execute(
        "SELECT * FROM potholes WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
        (min_lat, max_lat, min_lon, max_lon),
    ).fetchall()

    best, best_d = None, float("inf")
    for r in rows:
        d = haversine_m(lat, lon, r["lat"], r["lon"])
        if d > track_radius:
            continue
        if d > DEDUPE_RADIUS_M and _age_seconds(r["last_seen"]) > TRACK_TIMEOUT_S:
            continue          # too far away and not part of a live sighting run
        if d < best_d:
            best, best_d = r, d
    return best


def record_detection(
    lat: float,
    lon: float,
    confidence: float,
    area_fraction: float,
    width_m: float | None = None,
    length_m: float | None = None,
    distance_m: float | None = None,
    size_reliable: bool = True,
    evidence: str | None = None,
    accuracy_m: float | None = None,
    speed_mps: float | None = None,
    session_id: int | None = None,
    evidence_writer=None,
) -> dict:
    """Insert a detection, merging into an existing pothole when nearby.

    `evidence_writer`, if given, is a zero-argument callable returning a
    filename. It is invoked only when this sighting will actually become the
    stored evidence for the pothole (a new defect, or a better view than the
    one already held), so we do not write one JPEG per frame. Superseded
    evidence files are deleted.

    Returns a dict describing the pothole and whether it was newly created.
    """
    with connect() as conn:
        existing = _find_nearby(conn, lat, lon, speed_mps)

        if existing is None:
            if evidence_writer is not None and evidence is None:
                evidence = evidence_writer()
            severity = sev.classify(width_m, length_m, distance_m, size_reliable)
            cur = conn.execute(
                """INSERT INTO potholes
                   (lat, lon, first_seen, last_seen, sightings, max_conf,
                    area_fraction, width_m, length_m, severity, evidence, session_id)
                   VALUES (?,?,?,?,1,?,?,?,?,?,?,?)""",
                (lat, lon, _now(), _now(), confidence, area_fraction,
                 width_m, length_m, severity, evidence, session_id),
            )
            pothole_id = cur.lastrowid
            created = True
        else:
            pothole_id = existing["id"]
            sightings = existing["sightings"] + 1
            max_conf = max(existing["max_conf"], confidence)
            max_area = max(existing["area_fraction"], area_fraction)
            # Keep the best physical measurement, not the biggest-looking one.
            # A closer sighting is a better measurement, so a reliable estimate
            # supersedes an absent one and later reliable estimates are
            # averaged in, which damps per-frame bounding-box jitter.
            best_w, best_l = existing["width_m"], existing["length_m"]
            if size_reliable and width_m:
                if best_w is None:
                    best_w, best_l = width_m, length_m
                else:
                    k = existing["sightings"]
                    best_w = (best_w * k + width_m) / (k + 1)
                    if best_l is not None and length_m:
                        best_l = (best_l * k + length_m) / (k + 1)
            # Running mean position - repeated GPS fixes average out jitter.
            n = existing["sightings"]
            new_lat = (existing["lat"] * n + lat) / (n + 1)
            new_lon = (existing["lon"] * n + lon) / (n + 1)
            severity = sev.classify(best_w, best_l, distance_m,
                                    reliable=best_w is not None)
            # Keep only the best view of each defect; discard the rest.
            evid = existing["evidence"]
            if confidence >= existing["max_conf"]:
                new_name = evidence
                if new_name is None and evidence_writer is not None:
                    new_name = evidence_writer()
                if new_name:
                    if evid and evid != new_name:
                        _delete_evidence(evid)
                    evid = new_name
            elif evidence:
                _delete_evidence(evidence)
            conn.execute(
                """UPDATE potholes SET lat=?, lon=?, last_seen=?, sightings=?,
                   max_conf=?, area_fraction=?, width_m=?, length_m=?,
                   severity=?, evidence=? WHERE id=?""",
                (new_lat, new_lon, _now(), sightings, max_conf, max_area,
                 best_w, best_l, severity, evid, pothole_id),
            )
            created = False

        conn.execute(
            """INSERT INTO detections
               (pothole_id, ts, lat, lon, confidence, area_fraction,
                width_m, length_m, distance_m, accuracy_m, speed_mps, session_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (pothole_id, _now(), lat, lon, confidence, area_fraction,
             width_m, length_m, distance_m, accuracy_m, speed_mps, session_id),
        )
        row = conn.execute("SELECT * FROM potholes WHERE id=?", (pothole_id,)).fetchone()
        return {"created": created, **dict(row)}


def set_road_name(pothole_id: int, name: str | None) -> None:
    with connect() as conn:
        conn.execute("UPDATE potholes SET road_name=? WHERE id=?", (name, pothole_id))


def potholes_missing_road_name() -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM potholes WHERE road_name IS NULL OR road_name='' ORDER BY id")]


# --------------------------------------------------------------- track ----
# One point every TRACK_MIN_SPACING_M, not one per frame: at 3 fps and 30 mph
# a per-frame track would be 10,000 rows for a 15-minute drive and would tell
# us nothing extra. What matters is only which stretches of road were covered.
TRACK_MIN_SPACING_M = 8.0
_track_tail: dict[int, tuple[float, float]] = {}


def record_track_point(session_id: int | None, lat: float, lon: float) -> bool:
    """Log where the vehicle was. Returns True if the point was kept."""
    if session_id is None:
        return False
    prev = _track_tail.get(session_id)
    if prev is not None and haversine_m(prev[0], prev[1], lat, lon) < TRACK_MIN_SPACING_M:
        return False
    _track_tail[session_id] = (lat, lon)
    with connect() as conn:
        conn.execute(
            "INSERT INTO track (session_id, ts, lat, lon) VALUES (?,?,?,?)",
            (session_id, _now(), lat, lon))
    return True


def track_points(session_id: int) -> list[tuple[float, float]]:
    with connect() as conn:
        return [(r["lat"], r["lon"]) for r in conn.execute(
            "SELECT lat, lon FROM track WHERE session_id=? ORDER BY id", (session_id,))]


def sessions() -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM sessions ORDER BY id")]


def detections_for_session(session_id: int) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM detections WHERE session_id=? ORDER BY id", (session_id,))]


def sessions_that_saw(pothole_id: int) -> list[int]:
    """Session ids that recorded this defect, oldest first."""
    with connect() as conn:
        return [r["session_id"] for r in conn.execute(
            """SELECT session_id, MIN(id) AS first_id FROM detections
               WHERE pothole_id=? AND session_id IS NOT NULL
               GROUP BY session_id ORDER BY first_id""", (pothole_id,))]


def width_samples_in_session(pothole_id: int, session_id: int) -> list[float]:
    """Every individual width measurement of this defect in this survey.

    Several sightings per pass is what makes a change detectable: the spread
    across them is a direct estimate of the measurement noise, so growth can
    be tested against the instrument's own precision instead of a threshold
    picked by hand.
    """
    with connect() as conn:
        return [r["width_m"] for r in conn.execute(
            """SELECT width_m FROM detections
               WHERE pothole_id=? AND session_id=? AND width_m IS NOT NULL""",
            (pothole_id, session_id))]


def mean_width_in_session(pothole_id: int, session_id: int) -> float | None:
    samples = width_samples_in_session(pothole_id, session_id)
    return sum(samples) / len(samples) if samples else None


def all_potholes(status: str | None = None) -> list[dict]:
    q = "SELECT * FROM potholes"
    args: tuple = ()
    if status:
        q += " WHERE status=?"
        args = (status,)
    q += " ORDER BY id"
    with connect() as conn:
        return [dict(r) for r in conn.execute(q, args).fetchall()]


def mark_reported(ids: list[int]) -> int:
    if not ids:
        return 0
    with connect() as conn:
        cur = conn.execute(
            f"UPDATE potholes SET status='reported' WHERE id IN ({','.join('?' * len(ids))})",
            ids,
        )
        return cur.rowcount


def stats() -> dict:
    with connect() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS potholes,
                      COALESCE(SUM(sightings),0) AS sightings,
                      SUM(severity='high') AS high,
                      SUM(severity='medium') AS medium,
                      SUM(severity='low') AS low,
                      SUM(severity='unknown') AS unknown
               FROM potholes"""
        ).fetchone()
        dets = conn.execute("SELECT COUNT(*) AS c FROM detections").fetchone()["c"]
        return {**{k: (row[k] or 0) for k in row.keys()}, "detections": dets}


def reset(delete_evidence: bool = True) -> None:
    """Wipe all data - handy when re-running demos."""
    with connect() as conn:
        conn.executescript(
            "DELETE FROM detections; DELETE FROM potholes; "
            "DELETE FROM sessions; DELETE FROM track;"
        )
    _track_tail.clear()
    if delete_evidence:
        for f in EVIDENCE_DIR.glob("*.jpg"):
            try:
                f.unlink()
            except OSError:
                pass
