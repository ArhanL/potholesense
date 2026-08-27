"""Central configuration for PotholeSense."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Where surveys are stored. Overridable so that tests - which wipe the
# database between cases - can never touch real survey data. Set
# POTHOLESENSE_DATA_DIR to a scratch directory before importing this module.
DATA_DIR = Path(os.environ.get("POTHOLESENSE_DATA_DIR") or (BASE_DIR / "data"))
EVIDENCE_DIR = DATA_DIR / "evidence"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "potholes.db"

# --- Model -------------------------------------------------------------
# Point this at your own fine-tuned weights once training finishes.
# Falls back to a generic YOLO checkpoint so the app runs before training.
WEIGHTS_PATH = MODELS_DIR / "pothole_yolov8n.pt"
FALLBACK_WEIGHTS = "yolov8n.pt"

# Same network exported to ONNX, served to the phone so it can run inference
# itself. Optional: without it the phone streams frames to this server instead.
ONNX_WEIGHTS_PATH = MODELS_DIR / "pothole_yolov8n.onnx"

# Inference
CONF_THRESHOLD = 0.45      # min confidence to accept a detection
IOU_THRESHOLD = 0.50
IMG_SIZE = 640

# --- Deduplication -----------------------------------------------------
# Two detections closer than this are always treated as the same pothole.
DEDUPE_RADIUS_M = 12.0

# Motion-aware track association.
# A car at 30 mph covers ~4.5 m between frames, so a single pothole visible for
# 8 consecutive frames is tagged with GPS fixes spread over ~35 m - far wider
# than DEDUPE_RADIUS_M. Within TRACK_TIMEOUT_S of the last sighting we therefore
# widen the association radius to (speed x timeout), capped by TRACK_MAX_RADIUS_M,
# which merges a continuous sighting run without merging genuinely distinct
# defects encountered minutes apart.
TRACK_TIMEOUT_S = 2.5
TRACK_MAX_RADIUS_M = 40.0
DEFAULT_SPEED_MPS = 11.0   # assumed when the phone reports no speed

MIN_GPS_ACCURACY_M = 50.0  # reject fixes worse than this

# --- Severity ----------------------------------------------------------
# Severity is banded on the defect's measured width across the carriageway,
# in metres, recovered by projecting the bounding box onto the road plane
# (see app/localise.measure_defect).
#
# The thresholds are anchored to how UK highway authorities actually decide.
# The most commonly specified intervention level for a carriageway pothole is
# 40 mm deep by 300 mm wide; a 2018 RAC Foundation survey of local highway
# authorities found 40 mm depth used by 56% of them, with "40 mm deep, 300 mm
# wide" the most common depth-and-width pairing. Depth cannot be recovered
# from a single monocular frame, so width is the criterion we can measure:
#
#   low     < 0.30 m   below the usual intervention width
#   medium  0.30-0.60 m  meets the common 300 mm intervention width
#   high    >= 0.60 m    twice it - wide enough to span a wheel track
#
# Reporting a width in metres against a published criterion is defensible to
# a council in a way that "7% of the image" is not.
SEVERITY_WIDTH_BANDS_M = {
    "medium": 0.30,
    "high": 0.60,
}

# A measurement is only as good as the geometry behind it. Beyond this range
# the bounding box is a few pixels tall and the size estimate is not worth
# banding on, so such sightings inform position but not severity.
SEVERITY_MAX_RANGE_M = 25.0

for _d in (DATA_DIR, EVIDENCE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Camera geometry (inverse perspective mapping) ----------------------
# Measure these for your own setup; defaults suit a phone in a windscreen
# cradle in a typical hatchback.
CAMERA_HEIGHT_M = 1.25      # lens height above the road
CAMERA_PITCH_DEG = 8.0      # downward tilt from horizontal
CAMERA_VFOV_DEG = 48.0      # vertical field of view
CAMERA_HFOV_DEG = 65.0      # horizontal field of view
MIN_RANGE_M = 2.0           # closer than this the road plane is out of frame
MAX_RANGE_M = 35.0          # beyond this, rays near the horizon are unreliable
