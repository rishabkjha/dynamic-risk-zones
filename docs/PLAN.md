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

## Status: completed early, Wed 22 Oct

Everything planned for 22–26 Oct was done on 22 Oct, including the real-camera work (camera
check, calibration, depth vs tape measure, live demo recordings) and the report. The days left
are for review: read the report through, make slides from the figures, rehearse the demo.

## Status at midday, Wed 22

Done:
- Pipeline, predictor (leave-one-out + final), crowd benchmark, laptop set up, repo on GitHub
  (`rishabkjha/dynamic-risk-zones`).
- Pulled forward from Fri 24: collision attribution, ablations (social input, optical flow,
  robot speed, zone size), all tables final for the simulator part.
- Pulled forward from Sat 25: report draft with all simulator results and figures
  (`docs/REPORT.md`, `docs/figures/`), demo video of the simulator (`results/demo_learned.mp4`,
  not in git).
- Speed test and OpenVINO export on the laptop (OpenVINO turned out slower than PyTorch here).

Remaining:
- **Thu 23 (needs the user at the laptop):** confirm left/right with `01_check_cameras.py`
  (Day 1 frames suggest camera 4 is the left one), calibrate, measure depth error with a tape
  measure, run `04_live_zones.py` with people walking past, record clips and screenshots.
- Fill the report's Day 2 TODOs (calibration RMS, depth error, live fps, observations).
- Optional: OpenVINO on the integrated GPU (`--half`), NCNN on a Pi 4 if one is available, more
  crossings to separate cv from learned.
- Sun 26: demo video (live + simulator) and slides. Mon 27: final read-through and submit.

## Scope guard
- In: stereo depth, optical flow, YOLO detection, tracking, learned prediction, zones, planner,
  simulator benchmark, live webcam demo.
- Out (future work): driving a real robot, ROS 2 integration, Depth Anything, training YOLO on
  own data.
