#!/usr/bin/env python3
"""Closed-loop simulator and evaluation harness for PotholeSense.

Places potholes at *known world coordinates* along a route, drives a virtual
car past them, and renders each frame using the same camera geometry the
server uses to invert it. The server therefore has to recover the true
positions from imagery alone - a genuine end-to-end test of detection,
localisation and deduplication, with no car, phone or trained model required.

    POTHOLESENSE_STUB=1 python run.py &
    python scripts/simulate_drive.py --frames 120 --potholes 6
"""
from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.geo import haversine_m
from app.localise import ground_to_image, offset_latlon
from config import MIN_RANGE_M, MAX_RANGE_M

ROUTE_START = (51.45450, -2.58790)     # Bristol
HEADING_DEG = 35.0
POTHOLE_RADIUS_M = 0.45                # typical defect ~0.9 m across
FRAME_W, FRAME_H = 640, 480


def bbox_for(forward: float, lateral: float):
    """Ground-truth bounding box of a pothole at (forward, lateral) metres."""
    cx, cy = ground_to_image(forward, lateral, FRAME_W, FRAME_H)
    xr, _ = ground_to_image(forward, lateral + POTHOLE_RADIUS_M, FRAME_W, FRAME_H)
    _, y_near = ground_to_image(max(forward - POTHOLE_RADIUS_M, 0.5), lateral,
                                FRAME_W, FRAME_H)
    _, y_far = ground_to_image(forward + POTHOLE_RADIUS_M, lateral, FRAME_W, FRAME_H)
    ax = max(3.0, abs(xr - cx))
    return (cx - ax, min(y_near, y_far), cx + ax, max(y_near, y_far)), None


# ------------------------------------------------------------------ render --
def render_frame(visible, rng) -> np.ndarray:
    """Draw a road scene containing `visible` = [(forward_m, lateral_m), ...]."""
    img = np.full((FRAME_H, FRAME_W, 3), 118, dtype=np.uint8)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 8, (FRAME_H, FRAME_W, 1))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # Sky above the horizon, which sits where the road plane projects to infinity.
    horizon_y = int(ground_to_image(1e6, 0.0, FRAME_W, FRAME_H)[1])
    cv2.rectangle(img, (0, 0), (FRAME_W, max(0, horizon_y)), (185, 175, 160), -1)

    # Lane edges drawn in world space so perspective is consistent.
    for lat_off in (-1.8, 1.8):
        pts = []
        for d in np.linspace(MIN_RANGE_M, 80, 25):
            x, y = ground_to_image(float(d), lat_off, FRAME_W, FRAME_H)
            pts.append([int(x), int(y)])
        cv2.polylines(img, [np.array(pts, np.int32)], False, (225, 225, 220), 3)

    for forward, lateral in visible:
        cx, cy = ground_to_image(forward, lateral, FRAME_W, FRAME_H)
        xr, _ = ground_to_image(forward, lateral + POTHOLE_RADIUS_M, FRAME_W, FRAME_H)
        _, y_near = ground_to_image(max(forward - POTHOLE_RADIUS_M, 0.5), lateral, FRAME_W, FRAME_H)
        _, y_far = ground_to_image(forward + POTHOLE_RADIUS_M, lateral, FRAME_W, FRAME_H)
        ax = max(3, int(abs(xr - cx)))
        ay = max(2, int(abs(y_near - y_far) / 2))
        if cy < horizon_y or cy > FRAME_H + 40:
            continue
        cv2.ellipse(img, (int(cx), int(cy)), (ax, ay), 0, 0, 360, (36, 34, 31), -1)
        cv2.ellipse(img, (int(cx), int(cy)), (ax, ay), 0, 0, 360, (72, 68, 64), 2)
    return img


# -------------------------------------------------------------------- main --
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--speed-mps", type=float, default=13.4)     # ~30 mph
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--potholes", type=int, default=6)
    ap.add_argument("--spacing-m", type=float, default=55.0)
    ap.add_argument("--gps-noise-m", type=float, default=4.0)
    ap.add_argument("--match-radius", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--oracle", action="store_true",
                    help="bypass the detector and push ground-truth boxes, "
                         "isolating localisation + deduplication error")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    step_m = args.speed_mps / args.fps
    route_len = args.frames * step_m

    # Ground truth: potholes at known distances along the route.
    truth = []
    for i in range(args.potholes):
        along = 25.0 + i * args.spacing_m + rng.uniform(-6, 6)
        if along > route_len - 5:
            break
        lateral = rng.uniform(-1.4, 1.4)
        lat, lon = offset_latlon(*ROUTE_START, HEADING_DEG, along, lateral)
        truth.append({"along": along, "lateral": lateral, "lat": lat, "lon": lon})

    sess = requests.Session()
    sid = sess.post(f"{args.server}/api/session/start",
                    data={"device": "simulate_drive.py"}).json()["session_id"]
    print(f"session {sid}: {args.frames} frames @ {args.fps}fps, "
          f"{args.speed_mps:.1f} m/s, {len(truth)} ground-truth potholes "
          f"over {route_len:.0f} m")

    errors, sent = 0, 0
    for i in range(args.frames):
        travelled = i * step_m
        visible = [(t["along"] - travelled, t["lateral"]) for t in truth
                   if MIN_RANGE_M < (t["along"] - travelled) < MAX_RANGE_M]
        if not args.oracle:
            img = render_frame(visible, rng)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 72])

        car_lat, car_lon = offset_latlon(*ROUTE_START, HEADING_DEG, travelled, 0.0)
        # Realistic GPS noise on the car's own fix.
        n = args.gps_noise_m
        car_lat, car_lon = offset_latlon(car_lat, car_lon, rng.uniform(0, 360),
                                         rng.uniform(0, n), 0.0)
        heading = HEADING_DEG + rng.uniform(-2, 2)

        if args.oracle:
            for forward, lateral in visible:
                bx, by = bbox_for(forward, lateral)
                try:
                    r = sess.post(f"{args.server}/api/detection", data={
                        "lat": car_lat, "lon": car_lon, "confidence": 0.9,
                        "x1": bx[0], "y1": bx[1], "x2": bx[2], "y2": bx[3],
                        "frame_w": FRAME_W, "frame_h": FRAME_H,
                        "heading_deg": heading, "accuracy_m": rng.uniform(4, 8),
                        "speed_mps": args.speed_mps, "session_id": sid}, timeout=30)
                    r.raise_for_status()
                    sent += 1
                    d = r.json()
                    if d["new"] and not args.quiet:
                        print(f"  frame {i:3d}  NEW #{d['pothole_id']} {d['severity']:<6} "
                              f"{d['distance_m']} m ahead")
                except Exception as e:
                    errors += 1
                    print(f"  frame {i:3d}  ERROR {e}", file=sys.stderr)
            continue

        try:
            r = sess.post(
                f"{args.server}/api/frame",
                files={"image": ("f.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
                data={"lat": car_lat, "lon": car_lon,
                      "accuracy_m": rng.uniform(4, 8),
                      "speed_mps": args.speed_mps,
                      "heading_deg": HEADING_DEG + rng.uniform(-2, 2),
                      "session_id": sid},
                timeout=30)
            r.raise_for_status()
            sent += 1
            for d in r.json()["detections"]:
                if d["new"] and not args.quiet:
                    print(f"  frame {i:3d}  NEW #{d['pothole_id']} {d['severity']:<6} "
                          f"conf {d['confidence']:.2f}  {d['distance_m']} m ahead")
        except Exception as e:
            errors += 1
            print(f"  frame {i:3d}  ERROR {e}", file=sys.stderr)

    sess.post(f"{args.server}/api/session/{sid}/end")
    st = sess.get(f"{args.server}/api/stats").json()
    stored = sess.get(f"{args.server}/api/potholes").json()["potholes"]

    # Greedy one-to-one matching, nearest first.
    pairs = sorted(
        ((haversine_m(t["lat"], t["lon"], p["lat"], p["lon"]), ti, pi)
         for ti, t in enumerate(truth) for pi, p in enumerate(stored)),
        key=lambda x: x[0])
    used_t, used_p, dists = set(), set(), []
    for d, ti, pi in pairs:
        if d > args.match_radius or ti in used_t or pi in used_p:
            continue
        used_t.add(ti); used_p.add(pi); dists.append(d)

    tp = len(dists)
    fn = len(truth) - tp
    fp = len(stored) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print("\n" + "=" * 52)
    print("  END-TO-END LOCALISATION EVALUATION")
    print("=" * 52)
    print(f"  Ground-truth potholes   : {len(truth)}")
    print(f"  Unique potholes stored  : {st['potholes']}")
    print(f"  Raw detections          : {st['detections']} "
          f"({st['detections']/max(st['potholes'],1):.1f} per pothole)")
    print(f"  True positives          : {tp}")
    print(f"  False negatives         : {fn}")
    print(f"  False positives         : {fp}")
    print(f"  Precision               : {prec:.2f}")
    print(f"  Recall                  : {rec:.2f}")
    print(f"  F1                      : {f1:.2f}")
    if dists:
        print(f"  Localisation error      : mean {sum(dists)/len(dists):.1f} m, "
              f"max {max(dists):.1f} m")
    print(f"  Frames sent             : {sent} ({errors} errors)")
    print(f"  Avg inference           : {st['avg_inference_ms']} ms")
    print("=" * 52)
    print(f"\nOpen {args.server}/dashboard")
    return 0 if (fn == 0 and fp == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
