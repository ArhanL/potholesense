# PotholeSense

**Turn a phone in a windscreen cradle into an automated road-defect survey vehicle.**

Mount your phone, start driving. It streams the road ahead to a detection
model, works out where each pothole actually *is* on the map, merges repeat
sightings of the same defect, ranks them by severity, and produces a report you
can send to the council.

![Dashboard](docs_dashboard.png)

---

## Why this is harder than "run YOLO on a video"

Object detection is the easy half. Three problems sit between a bounding box
and something a highways department could use:

**1. The pothole is not where the car is.**
A detection tagged with the phone's GPS fix is wrong by 5–30 metres, and wrong
by a *different* amount in every frame as you approach. PotholeSense inverts
the camera projection under a flat-ground assumption (`app/localise.py`), using
the bounding box's bottom edge — where the defect meets the road plane — to
estimate distance ahead and lateral offset, then projects that along the
vehicle's heading. Every sighting of one pothole now lands on roughly the same
world point.

**2. One pothole produces twenty detections.**
At 30 mph a car covers 4.5 m between frames, so a defect visible for eight
frames spans ~35 m of GPS trace. Naively you report the same pothole twenty
times. PotholeSense associates each detection with existing defects by
geospatial proximity, averages positions across sightings (which cancels GPS
jitter rather than compounding it), and keeps only the highest-confidence
evidence photo.

**3. Councils need prioritisation, and "big in the image" is not a size.**
The same 0.9 m pothole covers 0.7% of the frame at 5 m and 0.003% at 30 m — a
factor of 200. Rank on apparent size and you have ranked how close the car
happened to get, not how bad the road is. PotholeSense projects the bounding
box onto the road plane and reports the defect's **width in metres**
(`app/severity.py`), which is stable to about 2 cm across the usable range.
That number can then be banded against a council's own published criterion
rather than an arbitrary one: the most common UK intervention level for a
carriageway pothole is 40 mm deep by 300 mm wide, so 300 mm is where `medium`
begins. Detector confidence and repeat sightings order defects *within* a band
but can never move one across a band — the physical measurement always
dominates, which is what makes the ordering defensible.

## Results

Measured with `scripts/simulate_drive.py`, which places potholes at **known
world coordinates**, drives a virtual car past them, renders each frame with
the same camera geometry the server has to invert, and adds 4 m of GPS noise to
the car's own fix. The server sees only imagery and noisy GPS.

Separating geometry error from detector error is the point of the harness:

| Configuration | Precision | Recall | F1 | Localisation error | Width error | Severity band |
|---|---|---|---|---|---|---|
| **Oracle detector** (isolates localisation + dedup) | **1.00** | **1.00** | **1.00** | **0.9 m** | **2 cm** | **9/10** |
| Classical CV baseline (`app/baseline.py`) | 0.30 | 0.50 | 0.37 | 8.2 m | — | — |
| Fine-tuned YOLOv8n | *run the notebook* | | | | | |

The oracle row is the meaningful engineering result. Given correct boxes, the
pipeline recovers 10/10 pothole positions to **0.9 m mean error despite 4 m of
GPS noise**, because averaging repeat sightings cancels the jitter; and it
measures their width to **2 cm mean error** (5 cm worst case) against defects
seeded anywhere from 0.18 m to 1.00 m across, putting 9 of 10 into the correct
severity band. The one miss sits within a few centimetres of a band boundary.

The baseline row is the honest control. A hand-tuned OpenCV pipeline (CLAHE →
adaptive threshold → contour shape filtering) reaches only F1 0.37 — it cannot
tell a pothole from a shadow or a drain cover. That gap is what the learned
model has to earn, and quoting it makes any mAP number you report mean
something.

```
====================================================
  END-TO-END LOCALISATION EVALUATION
====================================================
  Ground-truth potholes   : 10
  Unique potholes stored  : 10
  Raw detections          : 72 (7.2 per pothole)
  Precision               : 1.00
  Recall                  : 1.00
  F1                      : 1.00
  Localisation error      : mean 0.9 m, max 1.4 m
  Width measurement error : mean 2 cm, max 5 cm
  Severity band correct   : 9/10 (90%)
====================================================
```

## Architecture

```
   PHONE (windscreen cradle)          LAPTOP / SERVER
  ┌──────────────────────────┐      ┌────────────────────────────────┐
  │  capture.html            │      │  FastAPI            app/main.py│
  │   • rear camera @ 3 fps  │─────▶│   POST /api/frame              │
  │   • GPS + heading        │ HTTPS│      ↓                         │
  │   • live box overlay     │◀─────│   YOLOv8      app/detector.py  │
  │   • wake-lock, haptics   │ JSON │      ↓                         │
  └──────────────────────────┘      │   IPM geometry app/localise.py │
                                    │      ↓                         │
                                    │   dedup + store app/storage.py │
                                    │      ↓                         │
                                    │   SQLite + evidence JPEGs      │
                                    │      ↓                         │
                                    │   Leaflet map  /dashboard      │
                                    │   PDF / CSV    app/reports.py  │
                                    └────────────────────────────────┘
```

Everything on the server side is Python. The phone runs a small web page — no
app install, no App Store, works on Android and iOS.

**Why the model runs on the laptop, not the phone:** real-time camera inference
*on* a phone needs a native Kotlin/Swift shell, which puts the project outside
Python entirely. Offloading over a local hotspot keeps the whole system in one
language while still giving live on-screen feedback at 3 fps. `POST /api/detection`
already accepts detections computed elsewhere, so moving inference on-device
later is a client change, not a rewrite — and the notebook exports ONNX/TFLite
weights ready for it.

## Quick start

**macOS / Linux**

```bash
git clone <your-repo> && cd potholesense
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Try it with no model and no car:
POTHOLESENSE_STUB=1 python run.py &
python scripts/simulate_drive.py --frames 220 --potholes 10 --oracle
open http://localhost:8000/dashboard
```

**Windows (PowerShell)**

```powershell
git clone <your-repo>; cd potholesense
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Try it with no model and no car - run these in two terminals:
$env:POTHOLESENSE_STUB=1; python run.py
python scripts/simulate_drive.py --frames 220 --potholes 10 --oracle
start http://localhost:8000/dashboard
```

If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.
No `openssl` needed on any platform - the TLS certificate is generated in
pure Python.

### Real driving

```bash
python run.py --https      # TLS is mandatory: browsers block camera/GPS on plain HTTP
```

1. Put the laptop and phone on the same network — **phone hotspot, laptop tethered
   to it** is the arrangement that works in a moving car.
2. Open the printed `https://<ip>:8000/` on the phone, accept the self-signed
   certificate warning, allow camera and location.
3. Mount the phone, press **Start survey**, drive.
4. Watch `/dashboard` live, then download the PDF report.

### Calibrate for your car

Localisation accuracy depends on four numbers in `config.py`. Measure them once:

```python
CAMERA_HEIGHT_M  = 1.25   # lens height above the road (tape measure)
CAMERA_PITCH_DEG = 8.0    # downward tilt (phone's own inclinometer)
CAMERA_VFOV_DEG  = 48.0   # vertical field of view
CAMERA_HFOV_DEG  = 65.0   # horizontal field of view
```

Sanity check: park with a marker 10 m ahead, run one frame, confirm the
reported `distance_m` is close to 10.

## Training your own model

`notebooks/train_pothole_yolo.ipynb` — open in Colab with a T4 GPU, fine-tunes
YOLOv8n on a Roboflow pothole dataset (~25 min), benchmarks it against the
classical baseline, and exports `pothole_yolov8n.pt`. Drop that into `models/`
and restart; `/health` confirms which weights loaded.

## Tests

```bash
python -m pytest tests/ -v      # 41 tests: geometry, sizing, dedup, severity
```

## Project layout

| Path | What it does |
|---|---|
| `app/localise.py` | Inverse perspective mapping — image point → world coordinate |
| `app/storage.py` | SQLite + geospatial deduplication of repeat sightings |
| `app/detector.py` | YOLO wrapper, lazy loading, thread-safe |
| `app/baseline.py` | Classical CV detector — benchmark control |
| `app/severity.py` | Metric severity banding against council criteria |
| `app/reports.py` | Council PDF dossier + CSV export |
| `app/static/capture.html` | Phone client: camera, GPS, live overlay |
| `app/static/dashboard.html` | Live map, stats, exports |
| `scripts/simulate_drive.py` | Closed-loop evaluation harness |

## Limitations

Stated plainly, because an interviewer will ask:

- **Width is measured; depth is not.** A single monocular frame contains no
  depth information, so a wide shallow patch and a narrow deep hole are not
  distinguished. Width is half of the usual 40 mm × 300 mm criterion, and the
  report says so rather than implying a full assessment.
- **Size accuracy depends on the camera calibration.** The measurement inherits
  any error in the four numbers in `config.py`; a pitch that is 2° out biases
  every distance, and therefore every width, in the same direction.
- **Flat-ground assumption.** On a crest, dip or steep camber the projection
  degrades; error grows with distance, which is why estimates beyond 35 m are
  rejected outright.
- **No public-road validation yet.** All numbers above are from the synthetic
  harness. Real dashcam evaluation is the obvious next step.
- **Not a legal reporting channel.** The report is a well-formatted submission
  aid; no council currently ingests it automatically.

## Safety and legal

Mount the phone in a proper cradle and start the survey **before** you move.
Never interact with it while driving — UK law (Road Traffic Act 1988 s.41D and
the 2022 handheld-device regulations) prohibits handling a phone at the wheel,
and the whole design goal is that you press one button and then drive normally.
Footage of public roads from a vehicle is generally lawful in the UK, but if
you publish clips, blur faces and number plates.

## Roadmap

- On-device TFLite inference (weights already export; `POST /api/detection` already accepts them)
- Depth estimation for true severity, via monocular depth models or stereo from consecutive frames
- Road-name reverse geocoding so reports name the street
- Repeat-survey differencing: which potholes are new, which got worse, which were fixed
