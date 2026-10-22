# Plan: 22 → 27 October

Deadline: **27 October**. One commit per day minimum, with a DEVLOG entry.

| day | goal | done when |
|---|---|---|
| **Wed 22** | Project from scratch: pipeline, predictor trained, simulator benchmark, laptop set up | all scripts run; first results tables; first commit |
| **Thu 23** | Real cameras: calibrate on Ubuntu, live demo with people walking past, record 3–5 test clips. Speed test on the laptop, OpenVINO export | `04_live_zones.py` live at > 10 fps; clips in `recordings/`; `results/speed_*.md` |
| **Fri 24** | Improve the ML part: collision attribution in the benchmark, faster robot, predictor tuning (bigger history, more epochs, ablations: no social / no flow). Measure stereo distance error with a tape measure | final results tables; ablation table |
| **Sat 25** | Report: introduction, method (with figures from `viz`), experiments, results, limitations | draft in `docs/report/` |
| **Sun 26** | Demo video (live + simulator), slides, README polish, Pi 4 numbers if a Pi is available | video + slides |
| **Mon 27** | Buffer, final read-through, submit | submitted |

## Scope guard
- In: stereo depth, optical flow, YOLO detection, tracking, learned prediction, zones, planner,
  simulator benchmark, live webcam demo.
- Out (future work): driving a real robot, ROS 2 integration, Depth Anything, training YOLO on
  own data.
