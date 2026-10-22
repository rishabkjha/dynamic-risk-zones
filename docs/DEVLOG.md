# Development log

## Day 1: Wed 22 Oct 2026

**Built**
- New repo `dynamic_nav`, started from scratch (only the camera calibration tools were carried
  over from `stereo_nav`, ported to Linux).
- `dynnav/` library: V4L2 webcams + recording replay, rectification and raw SGBM depth, DIS
  optical flow with camera-motion removal, YOLO11n / HOG detector, ground-plane Kalman tracker
  that fuses stereo position with the optical-flow sideways speed, ETH/UCY loader with neighbour
  features, constant-velocity and social GRU predictors (numpy inference), the four zone
  methods, a sampling planner, a crowd simulator and a top-down viewer.
- Scripts 01–09 (calibration, live demo, data, training, benchmark, export, speed test).

**Results**
- Social GRU vs constant velocity (leave-one-scene-out, 5 scenes): ADE 0.52 vs 0.53 m,
  FDE 1.11 vs 1.15 m, NLL 1.12 vs 1.39. The GRU without neighbours is no better than constant
  velocity (ADE 0.55); the neighbour input is what helps. Constant velocity is known to be a
  strong baseline on ETH/UCY, so beating it at all is meaningful.
- Crowd benchmark (200 crossings per method, 360° sensor): success static 42%, reach 52%,
  cv 74%, learned 75%. Prediction-based zones clearly beat both "where it is now" and "where it
  could be"; reach zones keep the largest gap (0.39 m) but take 45% longer.
- With the webcams' 70° view, success drops to 36–56%: people approaching from the side are
  seen too late. Discussion point for the report.
- Speed test on this laptop (i5-10210U, 320×240, `09_speed_test.py`): 22.9 fps with YOLO11n in
  PyTorch (SGBM 12.1 ms, DIS 1.9 ms, YOLO 26.5 ms, prediction 2.3 ms, planning 1.0 ms) and
  17.1 fps with the OpenVINO export (SGBM 14.3 ms, DIS 2.2 ms, YOLO 38.5 ms, prediction 2.5 ms,
  planning 1.0 ms). OpenVINO was slower than PyTorch here (repeat run: 37.2 vs 26.1 ms).

**Setup**
- Laptop: Dell Vostro 3590, i5-10210U (8 threads), 23 GB RAM, Ubuntu 24.04. Project lives in
  `~/dynamic_nav` (the Windows drive under `/media/rishab/sda1` is mounted read-only).
- Webcams: Lenovo FHD at `/dev/video2` (left) and `/dev/video4` (right); to be confirmed with
  `01_check_cameras.py`.

**Next (Day 2)**
- Calibrate on Ubuntu, run the live demo with real people, record clips.
- OpenVINO export + speed test on the laptop.
- Benchmark: split collisions into "robot drove into person" vs "person walked into robot".
