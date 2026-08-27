"""YOLO inference wrapper.

Loads a fine-tuned pothole model if present, otherwise falls back to a stock
checkpoint (useful only for verifying the pipeline - it will NOT find potholes).
A stub backend is provided so the full pipeline can be tested without torch.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, asdict

import numpy as np

from config import (WEIGHTS_PATH, FALLBACK_WEIGHTS, CONF_THRESHOLD,
                    IOU_THRESHOLD, IMG_SIZE)

log = logging.getLogger("potholesense.detector")

# Class names that count as a road defect. A fine-tuned pothole model emits
# one of these; the stock COCO checkpoint emits "car", "person", "truck" and
# so on, none of which are defects. Without this filter, running before
# training turns every passing vehicle into a reported pothole.
DEFECT_LABELS = {"pothole", "potholes", "pot-hole", "pot hole",
                 "defect", "crack", "damage", "road_damage", "pothole_cv"}


def is_defect(label: str) -> bool:
    l = label.strip().lower().replace("-", " ")
    return l in DEFECT_LABELS or "pothole" in l


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    confidence: float
    label: str = "pothole"

    def as_dict(self):
        return asdict(self)


class Detector:
    """Thread-safe lazy-loading YOLO detector."""

    def __init__(self, weights=None, stub: bool = False):
        self._lock = threading.Lock()
        self._model = None
        self._stub = stub
        self._weights = weights
        self.using_fallback = False
        self.model_name = "stub" if stub else None

    # -- loading ---------------------------------------------------------
    def _load(self):
        if self._model is not None or self._stub:
            return
        from ultralytics import YOLO  # imported lazily: heavy

        weights = self._weights
        if weights is None:
            if WEIGHTS_PATH.exists():
                weights = str(WEIGHTS_PATH)
            else:
                weights = FALLBACK_WEIGHTS
                self.using_fallback = True
                log.warning(
                    "No fine-tuned weights at %s - falling back to %s. "
                    "This model does NOT detect potholes; train first.",
                    WEIGHTS_PATH, FALLBACK_WEIGHTS,
                )
        self._model = YOLO(weights)
        self.model_name = str(weights)
        log.info("Loaded detector: %s", self.model_name)

    def warmup(self):
        """Load weights and run one dummy inference so the first real frame
        is not penalised by lazy initialisation."""
        self._load()
        if self._model is not None:
            blank = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            self._model.predict(blank, imgsz=IMG_SIZE, verbose=False)

    # -- inference -------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._stub:
            return self._stub_detect(frame)

        with self._lock:
            self._load()
            results = self._model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
            )

        out: list[Detection] = []
        dropped = 0
        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id))
                # A single-class fine-tuned model may name its one class
                # anything, so accept everything when we know the weights are
                # ours. On the stock checkpoint, keep only defect classes -
                # otherwise cars and pedestrians get filed as potholes.
                if self.using_fallback and not is_defect(label):
                    dropped += 1
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                out.append(Detection((x1, y1, x2, y2), float(box.conf[0]), label))
        if dropped:
            log.debug("Dropped %d non-defect detections from fallback weights", dropped)
        return out

    @staticmethod
    def _stub_detect(frame: np.ndarray) -> list[Detection]:
        """Classical-CV fallback: exercises the full pipeline with no weights,
        and doubles as the benchmark baseline for the trained model."""
        from app import baseline
        return [Detection(bbox, conf, "pothole_cv")
                for bbox, conf in baseline.detect(frame)
                if conf >= CONF_THRESHOLD]


_default: Detector | None = None


def get_detector(stub: bool | None = None) -> Detector:
    """Return the process-wide detector.

    Set POTHOLESENSE_STUB=1 to run the deterministic stub backend, which lets
    the whole pipeline be demoed or tested without model weights or a GPU.
    """
    global _default
    if _default is None:
        if stub is None:
            stub = os.getenv("POTHOLESENSE_STUB", "").lower() in ("1", "true", "yes")
        _default = Detector(stub=stub)
        if stub:
            log.warning("Detector running in STUB mode - no real model loaded.")
    return _default
