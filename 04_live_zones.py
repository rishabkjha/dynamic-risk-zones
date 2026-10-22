"""Step 4: the live demo. Webcams -> detection + stereo depth + optical flow -> tracks -> risk zones -> plan.

    python 04_live_zones.py                          # live from the two webcams
    python 04_live_zones.py --method reach           # start with another zone method
    python 04_live_zones.py --video recordings/20261022_1530   # replay a recording (left.mp4 + right.mp4)
    python 04_live_zones.py --video recordings/20261022_1530 --out demo.mp4   # save the view as a video

Left panel: the left camera with detections (class, distance, speed). Right panel: the floor seen
from above, the camera rig at the bottom facing up, 1 m grid. Each moving object gets its
zones (red = no-go, orange, yellow) drawn for now, +1 s, +2 s and +3 s; the green line is the
path the robot would drive to the goal (star) 4 m ahead.
Without calib/stereo_calib.npz it still runs: distances then come from the box height (people only).

Keys: M = next zone method, R = start/stop recording, P = screenshot, Q = quit.
"""
import argparse
import os
import time

import cv2
import numpy as np

import config
from dynnav import zones
from dynnav.cams import StereoCams, VideoPair
from dynnav.detector import make_detector
from dynnav.flow import Flow, background_flow, box_flow, lateral_speed
from dynnav.planner import Planner
from dynnav.predictor import ConstantVelocity, GRUPredictor
from dynnav.stereo import Rectifier, StereoDepth, box_depth, box_to_ground, size_depth
from dynnav.tracker import Tracker
from dynnav.viz import TopDown


def load_predictors():
    if not os.path.exists(config.PREDICTOR_FILE):
        print(f"{config.PREDICTOR_FILE} not found: only the static and reach methods will work.")
        return {}
    predictors = {"learned": GRUPredictor.load(config.PREDICTOR_FILE),
                  "cv": ConstantVelocity.load(config.PREDICTOR_FILE)}
    if os.path.exists(config.PREDICTOR_NOSOCIAL_FILE):
        predictors["learned_nosocial"] = GRUPredictor.load(config.PREDICTOR_NOSOCIAL_FILE)
    return predictors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=config.ZONE_METHOD, choices=zones.METHODS)
    ap.add_argument("--video", help="folder with left.mp4 and right.mp4 from R in this script")
    ap.add_argument("--calib", default=config.CALIB_FILE)
    ap.add_argument("--out", help="also save what is shown to this .mp4 (for the report / slides)")
    ap.add_argument("--no-window", action="store_true", help="don't open a window (with --out)")
    args = ap.parse_args()

    S = config.PROCESS_SCALE
    W, H = int(config.WIDTH * S), int(config.HEIGHT * S)
    if os.path.exists(args.calib):
        rect = Rectifier(args.calib, S)
        stereo = StereoDepth(rect.fx, rect.baseline_m, disparity_offset=config.DISPARITY_OFFSET_PX)
        fx, cx = rect.fx, rect.cx
        print(f"calibration loaded: baseline {rect.baseline_m * 1000:.0f} mm, focal {fx:.0f} px")
    else:
        rect = stereo = None
        fx, cx = config.GUESS_FOCAL_PX * S, W / 2
        print("No calibration: distances come from box height (people only). Run steps 2-3 for stereo.")

    source = VideoPair(os.path.join(args.video, "left.mp4"), os.path.join(args.video, "right.mp4")) \
        if args.video else StereoCams()
    detector = make_detector(config.DETECTOR_MODEL, config.DETECT_SIZE, config.DETECT_CONF)
    flow = Flow()
    tracker = Tracker(gate=config.TRACK_GATE_M, max_age=config.TRACK_MAX_AGE_S,
                      min_hits=config.TRACK_MIN_HITS)
    planner = Planner()
    predictors = load_predictors()
    method = args.method if args.method in ("static", "reach") or args.method in predictors else "reach"
    view = TopDown(size=H * 2, metres=8.0, centre=(3.6, 0.0))
    robot = np.zeros(3)

    os.makedirs("screenshots", exist_ok=True)
    rec = None
    out = None
    frame_no = 0
    t_video = 0.0
    t = 0.0
    t_prev = time.time()
    fps = 0.0
    dets = []
    while True:
        fl, fr = source.read()
        t_grab = time.time()
        if fl is None:
            if args.video:
                break
            continue
        if rec is not None:
            rec[0].write(fl)
            rec[1].write(fr)
            rec[2].write(f"{t_grab:.4f}\n")
        if rect is not None:
            left, right = rect.rectify(fl, fr)
        else:
            left = cv2.resize(fl, (W, H), interpolation=cv2.INTER_AREA)
            right = None
        gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        depth = stereo.compute(gray, cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)) if stereo else None
        fl_now = flow.update(gray)

        if args.video:
            t_video = source.time if source.time is not None else t_video + 1.0 / source.fps
            dt_frame = max(t_video - t, 1e-3) if frame_no else 1.0 / source.fps
            t = t_video
        else:
            dt_frame = max(t_grab - t, 1e-3) if frame_no else 1.0 / 15
            t = t_grab
        if frame_no % config.DETECT_EVERY == 0:
            dets = [d for d in detector(left) if d.label not in config.IGNORE_CLASSES]
        frame_no += 1

        bg = background_flow(fl_now, [d.box for d in dets]) if fl_now is not None else None
        measurements, labels = [], []
        for d in dets:
            if depth is not None:
                z = box_depth(depth, d.box)
                std = 0.02 + 0.01 * z * z
            else:
                z = size_depth(d.box, fx) if d.label == "person" else 0.0
                std = 0.15 * z
            if z <= 0:
                labels.append(f"{d.label} ?")
                continue
            pos = box_to_ground(d.box, z, fx, cx)
            m = dict(pos=pos, label=d.label, pos_std=max(std, 0.05))
            if fl_now is not None:
                du = float(box_flow(fl_now, d.box)[0] - bg[0])
                ray = pos / np.linalg.norm(pos)
                m["lateral"] = (np.array([-ray[1], ray[0]]), lateral_speed(du, z, fx, dt_frame), 0.15)
            measurements.append(m)
            labels.append(f"{d.label} {z:.1f}m")

        tracks = tracker.step(t, measurements)
        z_set = zones.build(method, tracks, planner.times, predictors.get(method))
        cmd, paths, best, _ = planner.plan(robot, 0.0, config.LIVE_GOAL, z_set)

        cam_view = left.copy()
        for d, lab in zip(dets, labels):
            x0, y0, x1, y1 = (int(v) for v in d.box)
            cv2.rectangle(cam_view, (x0, y0), (x1, y1), (0, 200, 0), 2)
            cv2.putText(cam_view, lab, (x0, max(y0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        status = f"{fps:.1f} fps  zones: {method}  plan v={cmd[0]:.2f} w={cmd[1]:+.2f}"
        if rec is not None:
            status += "  REC"
        cv2.putText(cam_view, status, (5, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cam_view = cv2.resize(cam_view, (W * 2, H * 2))
        top = view.draw(robot, config.LIVE_GOAL, tracks, z_set, paths, best, title=f"zones: {method}")
        screen = np.hstack([cam_view, top])
        if args.out:
            if out is None:
                out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                      (screen.shape[1], screen.shape[0]))
            out.write(screen)
        if args.no_window:
            continue
        cv2.imshow("dynamic zones", screen)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            options = [m for m in zones.METHODS if m in ("static", "reach") or m in predictors]
            method = options[(options.index(method) + 1) % len(options)]
        elif key == ord("p"):
            path = os.path.join("screenshots", time.strftime("%Y%m%d_%H%M%S.png"))
            cv2.imwrite(path, screen)
            print("saved", path)
        elif key == ord("r") and not args.video:
            if rec is None:
                folder = os.path.join("recordings", time.strftime("%Y%m%d_%H%M%S"))
                os.makedirs(folder, exist_ok=True)
                four = cv2.VideoWriter_fourcc(*"mp4v")
                size = (fl.shape[1], fl.shape[0])
                # The real frame times go to times.txt, so a replay sees the true speeds.
                rec = (cv2.VideoWriter(os.path.join(folder, "left.mp4"), four, 15, size),
                       cv2.VideoWriter(os.path.join(folder, "right.mp4"), four, 15, size),
                       open(os.path.join(folder, "times.txt"), "w"))
                print("recording to", folder)
            else:
                rec[0].release()
                rec[1].release()
                rec[2].close()
                rec = None
                print("recording stopped")

    if rec is not None:
        rec[0].release()
        rec[1].release()
        rec[2].close()
    if out is not None:
        out.release()
    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
