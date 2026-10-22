# dynamic_nav: navigating among moving people with stereo depth, optical flow and learned motion prediction

A robot (here: two USB webcams as a stereo camera) sees people and objects, measures where they
are (stereo depth) and how they move (tracking + optical flow), **predicts where each one will
be over the next 3 seconds with a neural network**, turns that into nested risk zones, and plans
a path that stays out of them. It runs on an ordinary laptop CPU and is designed to fit a
Raspberry Pi 4.

```
2 webcams ─> rectify ─> SGBM stereo depth ──────────┐
                     ├> YOLO11n detection ───────────┼─> 3-D position on the floor ─┐
                     └> DIS optical flow ────────────┘   + sideways speed          │
                                                                                     v
            Kalman tracker (one track per object: position, velocity, history) <───┘
                                   │
                                   v
   social GRU predictor (trained on ETH/UCY): future path + uncertainty, per object
                                   │
                                   v
   risk zones: inner (no-go) / middle / outer ellipses for every future time step
                                   │
                                   v
   sampling planner: ~90 candidate paths, checked against the zones at the time the robot
   would be there -> fastest path that stays out of the no-go zones
```

## The idea: four ways to turn a moving object into "where not to drive"

| method | zones are | thinking |
|---|---|---|
| `static` | where the object is now | what a normal costmap does; ignores motion |
| `reach` | circles growing at ½ × typical, typical and maximum speed of the object's **class** (person 1.4 / 2.5 m/s, bicycle, dog…) | "a person can't be further than 2.5 m/s × t away": safe but very cautious |
| `cv` | ellipses around a constant-velocity prediction, at 1/2/3 σ of its measured error | "keeps walking the same way" |
| `learned` | ellipses around the **social GRU's** prediction, at 1/2/3 σ of the uncertainty **it predicts itself** | "where it will most likely be, and how sure we are", learned from real people, including how they move around each other |

The smallest (inner) zone is the most likely hit zone and is never entered; the outer ones
only cost something; outside all zones is free. Zones are checked **at the time** the robot would
be there, so the robot can pass behind a walking person instead of freezing in front of their
whole possible path.

## Results so far (22 Oct)

**Motion prediction**, ETH/UCY, leave-one-scene-out, 3.2 s observed → 4.8 s predicted
(full table: [results/predictor_metrics.md](results/predictor_metrics.md)):

| | ADE (m) ↓ | FDE (m) ↓ | NLL ↓ |
|---|---|---|---|
| constant velocity | 0.53 | 1.15 | 1.39 |
| GRU (own track only) | 0.55 | 1.16 | 1.31 |
| **social GRU (ours)** | **0.52** | **1.11** | **1.12** |

**Navigation**, simulated robot crossing the real recorded crowds, 40 crossings × 5 scenes per
method (tables: [360° sensor](results/nav_metrics_fov360.md), [70° webcam view](results/nav_metrics_webcam.md)):

| zones | success (360°) | collisions (360°) | time to goal | success (webcam 70°) |
|---|---|---|---|---|
| static | 42% | 58% | 14.9 s | 36% |
| reach (class max speed) | 52% | 38% | 21.6 s | 42% |
| cv | 74% | 24% | 16.4 s | 52% |
| **learned (social GRU)** | **75%** | **24%** | **16.8 s** | **56%** |

Pedestrians in the recordings don't react to the robot, so some collisions can't be avoided by
any method (being caught from behind at 0.8 m/s). See [docs/DEVLOG.md](docs/DEVLOG.md) for what's next.

## Setup (Ubuntu 24.04)

```bash
sudo apt install -y python3-pip python3.12-venv git v4l-utils
cd ~/dynamic_nav
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
Every time you open a new terminal: `cd ~/dynamic_nav && source .venv/bin/activate`.

## Steps

| script | what it does |
|---|---|
| `01_check_cameras.py` | finds the two webcams, checks which is left (cover a lens) |
| `02_capture_calibration.py` | photos of the ChArUco board for calibration (SPACE to save, 20–30 pairs) |
| `03_calibrate.py` | stereo calibration → `calib/stereo_calib.npz` (aim for RMS < 0.5 px) |
| `04_live_zones.py` | **the live demo**: detection, depth, flow, tracks, zones and the planned path. `M` switches method, `R` records, `P` screenshot |
| `05_get_data.py` | downloads the ETH/UCY pedestrian datasets (14 MB) |
| `06_train_predictor.py` | trains the predictors (leave-one-out + final model), writes `results/predictor_metrics.md` |
| `07_evaluate.py` | crowd simulator benchmark of the four zone methods, writes `results/nav_metrics*.md` |
| `08_export_model.py` | exports YOLO to OpenVINO (Intel) or NCNN (Pi 4) |
| `09_speed_test.py` | milliseconds per stage on this computer → `results/speed_<name>.md` |

Settings (camera numbers, board size, class speeds, robot limits, zone method) are all in
`config.py`. Tests: `python -m unittest`.

## Hardware

| computer | detection model | expected |
|---|---|---|
| laptop, Intel i5 (this one) | `yolo11n_openvino_model/` | whole pipeline > 15 fps at 320×240 |
| Intel NUC | `yolo11n_openvino_model/` (`--half` for the iGPU) | same or faster |
| Raspberry Pi 4 | `yolo11n_ncnn_model/`, `DETECT_EVERY = 2` | ~5 fps detection, tracking every frame |

The predictor runs in plain numpy (about 1 ms for 5 people), so the robot needs no PyTorch;
only training does.

## Layout

```
config.py          all settings
dynnav/            the library: cams, board, stereo, flow, detector, tracker, data,
                   predictor, zones, planner, sim, viz
models/            trained predictors (predictor.npz = deployed; *_<scene>.npz = leave-one-out)
results/           metric tables, speed tests, videos
docs/              PLAN.md (schedule), DEVLOG.md (daily log), report material
```
