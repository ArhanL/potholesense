"""PotholeSense API server.

The phone acts as a camera + GPS sensor; this server runs the detection model,
deduplicates defects geospatially, stores evidence and produces council reports.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import (FileResponse, HTMLResponse, PlainTextResponse,
                               JSONResponse)
from fastapi.staticfiles import StaticFiles

from config import EVIDENCE_DIR, DATA_DIR, MIN_GPS_ACCURACY_M, CONF_THRESHOLD
from app import storage, reports, geocode, survey
from app.detector import get_detector
from app.severity import bbox_area_fraction
from app.localise import locate_detection, measure_defect, bearing_between

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("potholesense")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="PotholeSense", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_stats = {"frames": 0, "infer_ms_total": 0.0, "detections": 0}


@app.on_event("startup")
def _startup():
    storage.init_db()
    log.info("Database ready at %s", DATA_DIR)

    # Load the weights and run one dummy inference now, in the background.
    # Otherwise the first frame of a real drive pays several seconds of lazy
    # model initialisation - exactly when the car is already moving.
    def _warm():
        try:
            get_detector().warmup()
            log.info("Detector warm and ready")
        except Exception as exc:                       # noqa: BLE001
            log.warning("Detector warmup failed (%s); will load on first frame", exc)

    threading.Thread(target=_warm, name="detector-warmup", daemon=True).start()


# ---------------------------------------------------------------- pages ----
@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "capture.html").read_text(encoding="utf-8")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return (STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    d = get_detector()
    avg = (_stats["infer_ms_total"] / _stats["frames"]) if _stats["frames"] else 0
    return {
        "status": "ok",
        "model": d.model_name or "not loaded",
        "using_fallback_weights": d.using_fallback,
        "frames_processed": _stats["frames"],
        "avg_inference_ms": round(avg, 1),
        "conf_threshold": CONF_THRESHOLD,
    }


# -------------------------------------------------------------- sessions ---
@app.post("/api/session/start")
def session_start(device: str = Form("unknown")):
    sid = storage.start_session(device)
    log.info("Session %s started (%s)", sid, device)
    return {"session_id": sid}


@app.post("/api/session/{session_id}/end")
def session_end(session_id: int):
    storage.end_session(session_id)
    return {"ok": True, "session_id": session_id}


# ----------------------------------------------------------------- frame ---
@app.post("/api/frame")
def ingest_frame(
    image: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    accuracy_m: float | None = Form(None),
    speed_mps: float | None = Form(None),
    heading_deg: float | None = Form(None),
    session_id: int | None = Form(None),
):
    """Accept one frame + GPS fix, run detection, persist any new potholes."""
    if accuracy_m is not None and accuracy_m > MIN_GPS_ACCURACY_M:
        return JSONResponse(
            {"skipped": "gps_accuracy", "accuracy_m": accuracy_m}, status_code=202
        )

    raw = image.file.read()
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")

    h, w = frame.shape[:2]
    t0 = time.perf_counter()
    detections = get_detector().detect(frame)
    infer_ms = (time.perf_counter() - t0) * 1000

    _stats["frames"] += 1
    _stats["infer_ms_total"] += infer_ms
    if session_id:
        storage.bump_frames(session_id)
        # Record where we drove, so a later survey can distinguish "this
        # defect is gone" from "we never came back down this road".
        storage.record_track_point(session_id, lat, lon)

    heading = _resolve_heading(session_id, lat, lon, heading_deg)

    results = []
    for det in detections:
        area = bbox_area_fraction(det.bbox, w, h)
        # Estimate where the pothole actually is, not where the car is.
        p_lat, p_lon, gp = locate_detection(det.bbox, w, h, lat, lon, heading)
        # ...and how big it actually is, in metres, not in pixels.
        size = measure_defect(det.bbox, w, h)
        rec = storage.record_detection(
            lat=p_lat, lon=p_lon, confidence=det.confidence, area_fraction=area,
            width_m=size.width_m or None, length_m=size.length_m or None,
            distance_m=size.distance_m if size.reliable else None,
            size_reliable=size.reliable,
            accuracy_m=accuracy_m, speed_mps=speed_mps, session_id=session_id,
            evidence_writer=lambda f=frame, d=det: _save_evidence(f, d),
        )
        _stats["detections"] += 1
        results.append({
            "pothole_id": rec["id"],
            "new": rec["created"],
            "severity": rec["severity"],
            "sightings": rec["sightings"],
            "confidence": round(det.confidence, 3),
            "bbox": [round(v) for v in det.bbox],
            "distance_m": (round(gp.distance_m, 1)
                           if gp.distance_m != float("inf") else None),
            "width_m": round(size.width_m, 2) if size.reliable else None,
            "range_reliable": gp.reliable,
        })
        if rec["created"]:
            log.info("NEW pothole #%s (%s) at %.5f,%.5f conf=%.2f",
                     rec["id"], rec["severity"], lat, lon, det.confidence)

    return {
        "detections": results,
        "inference_ms": round(infer_ms, 1),
        "frame": {"w": w, "h": h},
    }


# Last known position per session, used to derive heading when the phone's
# own compass/course is unavailable (common at low speed or indoors).
# Bounded: a long-running server would otherwise accumulate one entry per
# session for the life of the process.
_MAX_TRACKED_SESSIONS = 64
_last_fix: dict[int, tuple[float, float]] = {}
_last_heading: dict[int, float] = {}


def _forget_oldest_sessions() -> None:
    while len(_last_fix) > _MAX_TRACKED_SESSIONS:
        oldest = next(iter(_last_fix))
        _last_fix.pop(oldest, None)
        _last_heading.pop(oldest, None)


def _resolve_heading(session_id, lat, lon, reported):
    """Prefer the device's reported course; otherwise derive it from successive
    GPS fixes; otherwise reuse the last known heading."""
    key = session_id or 0
    _forget_oldest_sessions()
    if reported is not None and not (isinstance(reported, float) and reported != reported):
        _last_heading[key] = reported
        _last_fix[key] = (lat, lon)
        return reported

    prev = _last_fix.get(key)
    _last_fix[key] = (lat, lon)
    if prev is not None:
        from app.geo import haversine_m
        # Only trust a derived bearing if we actually moved a few metres,
        # otherwise GPS jitter produces a random heading.
        if haversine_m(prev[0], prev[1], lat, lon) >= 3.0:
            h = bearing_between(prev[0], prev[1], lat, lon)
            _last_heading[key] = h
            return h
    return _last_heading.get(key)


def _save_evidence(frame: np.ndarray, det) -> str:
    """Save an annotated crop-in-context image as evidence."""
    annotated = frame.copy()
    x1, y1, x2, y2 = (int(v) for v in det.bbox)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
    label = f"pothole {det.confidence:.2f}"
    cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    name = f"{uuid.uuid4().hex[:12]}.jpg"
    cv2.imwrite(str(EVIDENCE_DIR / name), annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return name


@app.post("/api/detection")
def ingest_detection(
    lat: float = Form(...),
    lon: float = Form(...),
    confidence: float = Form(...),
    x1: float = Form(...), y1: float = Form(...),
    x2: float = Form(...), y2: float = Form(...),
    frame_w: int = Form(...), frame_h: int = Form(...),
    heading_deg: float | None = Form(None),
    accuracy_m: float | None = Form(None),
    speed_mps: float | None = Form(None),
    session_id: int | None = Form(None),
):
    """Accept a detection computed elsewhere, without the image.

    Used by clients that run the model on-device (no frame upload needed, which
    saves bandwidth and keeps imagery on the phone), and by the evaluation
    harness to measure localisation and deduplication independently of
    detector accuracy.
    """
    bbox = (x1, y1, x2, y2)
    storage.record_track_point(session_id, lat, lon)
    heading = _resolve_heading(session_id, lat, lon, heading_deg)
    p_lat, p_lon, gp = locate_detection(bbox, frame_w, frame_h, lat, lon, heading)
    area = bbox_area_fraction(bbox, frame_w, frame_h)
    size = measure_defect(bbox, frame_w, frame_h)
    rec = storage.record_detection(
        lat=p_lat, lon=p_lon, confidence=confidence, area_fraction=area,
        width_m=size.width_m or None, length_m=size.length_m or None,
        distance_m=size.distance_m if size.reliable else None,
        size_reliable=size.reliable,
        evidence=None, accuracy_m=accuracy_m, speed_mps=speed_mps,
        session_id=session_id,
    )
    # Deliberately not counted as a frame: this endpoint receives one
    # detection, not an image, so counting it would corrupt avg_inference_ms.
    _stats["detections"] += 1
    return {
        "pothole_id": rec["id"], "new": rec["created"],
        "severity": rec["severity"], "sightings": rec["sightings"],
        "distance_m": round(gp.distance_m, 1) if gp.distance_m != float("inf") else None,
        "width_m": round(size.width_m, 2) if size.reliable else None,
        "range_reliable": gp.reliable,
    }


@app.get("/evidence/{name}")
def evidence(name: str):
    p = EVIDENCE_DIR / Path(name).name
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="image/jpeg")


# ------------------------------------------------------------------ data ---
@app.get("/api/potholes")
def list_potholes(status: str | None = None):
    return {"potholes": storage.all_potholes(status)}


@app.post("/api/geocode")
def geocode_potholes(limit: int = 200):
    """Fill in road names for defects that do not have one yet.

    Run after a drive, not during: reverse geocoding is a network round trip
    per road and is rate-limited to one request a second, so it has no place
    on the frame path. Defects within about 55 m of each other share a single
    lookup. Anything that fails simply keeps its coordinate.
    """
    pending = storage.potholes_missing_road_name()[:limit]
    named = 0
    for p in pending:
        name = geocode.road_name(p["lat"], p["lon"])
        if name:
            storage.set_road_name(p["id"], name)
            named += 1
    return {"considered": len(pending), "named": named,
            "unresolved": len(pending) - named}


@app.post("/api/track")
def ingest_track(
    lat: float = Form(...),
    lon: float = Form(...),
    accuracy_m: float | None = Form(None),
    session_id: int | None = Form(None),
):
    """Report the vehicle's position for a frame that produced no detection.

    Needed by any client that runs the model itself and therefore only posts
    detections: without this, the track would have a hole everywhere the road
    was clear, and a defect repaired since the last survey would be reported
    as 'not surveyed' rather than 'fixed'. Coverage is the evidence that
    absence means anything, so it has to be reported whether or not there was
    something to see.
    """
    if accuracy_m is not None and accuracy_m > MIN_GPS_ACCURACY_M:
        return {"recorded": False, "reason": "gps_accuracy"}
    return {"recorded": storage.record_track_point(session_id, lat, lon)}


@app.get("/api/sessions")
def list_sessions():
    return {"sessions": storage.sessions()}


@app.get("/api/survey/{session_id}/diff")
def survey_diff(session_id: int):
    """What changed in this survey compared with everything before it.

    Verdicts are new, worse, unchanged, fixed and not_surveyed. 'fixed' is
    only claimed where this survey's track actually passed the defect - a
    pothole on a road we did not drive is reported as not_surveyed rather
    than quietly counted as repaired.
    """
    return survey.diff_session(session_id)


@app.get("/api/stats")
def get_stats():
    s = storage.stats()
    avg = (_stats["infer_ms_total"] / _stats["frames"]) if _stats["frames"] else 0
    s["frames_processed"] = _stats["frames"]
    s["avg_inference_ms"] = round(avg, 1)
    return s


@app.get("/api/export/csv", response_class=PlainTextResponse)
def export_csv():
    return PlainTextResponse(
        reports.to_csv(),
        headers={"Content-Disposition": "attachment; filename=pothole_report.csv"},
    )


@app.get("/api/export/pdf")
def export_pdf(council: str = "Local Highways Authority"):
    out = DATA_DIR / "pothole_report.pdf"
    reports.to_pdf(out, council=council)
    return FileResponse(out, media_type="application/pdf",
                        filename="pothole_report.pdf")


@app.post("/api/reset")
def reset():
    storage.reset()
    _stats.update({"frames": 0, "infer_ms_total": 0.0, "detections": 0})
    return {"ok": True}
