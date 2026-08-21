"""Classical computer-vision pothole detector (no learning).

This exists for two reasons:
  1. It lets the whole pipeline be demoed/tested with zero model weights.
  2. It is an honest baseline to benchmark the trained YOLO model against -
     "my CNN beats a hand-tuned CV pipeline by X mAP" is a far stronger claim
     than an unanchored accuracy number.

Method: restrict to the road region (lower centre of frame), equalise
illumination, threshold for locally dark blobs, then filter candidate contours
by area, aspect ratio, solidity and vertical position.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_AREA_FRAC = 0.0035     # ignore specks
MAX_AREA_FRAC = 0.30       # ignore shadows covering the whole road
MIN_SOLIDITY = 0.55        # potholes are blobby, not stringy like cracks
MAX_ASPECT = 4.0


def detect(frame: np.ndarray) -> list[tuple[tuple[float, float, float, float], float]]:
    """Return [((x1,y1,x2,y2), confidence), ...] in full-frame pixel coords."""
    h, w = frame.shape[:2]
    y0 = int(h * 0.45)                     # road ROI: below the horizon
    x0, x1r = int(w * 0.10), int(w * 0.90)
    roi = frame[y0:, x0:x1r]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # CLAHE flattens uneven road lighting so a global threshold behaves.
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    mu, sd = float(gray.mean()), float(gray.std())
    thresh_val = max(0.0, mu - 1.2 * sd)
    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(w * h)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        frac = area / frame_area
        if not (MIN_AREA_FRAC <= frac <= MAX_AREA_FRAC):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bh == 0 or bw == 0:
            continue
        aspect = max(bw / bh, bh / bw)
        if aspect > MAX_ASPECT:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < MIN_SOLIDITY:
            continue

        # Confidence: how dark the blob is versus the surrounding road,
        # scaled by how blob-like it is. Bounded to [0.30, 0.95].
        blob_mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(blob_mask, [c], -1, 255, -1)
        blob_mean = cv2.mean(gray, mask=blob_mask)[0]
        contrast = np.clip((mu - blob_mean) / max(sd, 1e-6) / 2.5, 0, 1)
        conf = float(np.clip(0.30 + 0.65 * (0.6 * contrast + 0.4 * solidity), 0.30, 0.95))

        out.append(((bx + x0, by + y0, bx + x0 + bw, by + y0 + bh), conf))

    out.sort(key=lambda t: -t[1])
    return out[:5]
