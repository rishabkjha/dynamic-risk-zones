"""Step 9: how fast does each part of the pipeline run on THIS computer?

    python 09_speed_test.py                     # uses DETECTOR_MODEL from config.py
    python 09_speed_test.py --model yolo11n_openvino_model/

Times every stage on a 320x240 image (milliseconds per frame, median of 50 runs) and writes
results/speed_<computer name>.md, so you can put laptop / NUC / Pi 4 side by side in the report.
No webcams needed.
"""
import argparse
import os
import platform
import socket
import time

import cv2
import numpy as np

import config
from dynnav import zones
from dynnav.detector import make_detector
from dynnav.flow import Flow
from dynnav.planner import Planner
from dynnav.predictor import GRUPredictor
from dynnav.stereo import StereoDepth
from dynnav.tracker import Tracker


def timed(fn, n=50):
    fn()                                   # warm-up (first YOLO call loads the model)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DETECTOR_MODEL)
    ap.add_argument("--objects", type=int, default=5, help="tracked people for the prediction stages")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    W, H = int(config.WIDTH * config.PROCESS_SCALE), int(config.HEIGHT * config.PROCESS_SCALE)
    sample = os.path.join(os.path.dirname(__import__("ultralytics").__file__), "assets", "bus.jpg")
    img = cv2.resize(cv2.imread(sample), (W, H)) if os.path.exists(sample) else \
        rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = np.roll(gray, -8, axis=1)

    stereo = StereoDepth(300.0, 0.095)
    flow = Flow()
    flow.update(gray)
    detector = make_detector(args.model, config.DETECT_SIZE, config.DETECT_CONF)
    planner = Planner()
    predictor = GRUPredictor.load(config.PREDICTOR_FILE)
    tracker = Tracker(min_hits=1)
    for k in range(10):
        tracker.step(0.1 * k, [dict(pos=np.array([2.0 + 0.5 * i, -1.0 + 0.1 * k + 0.4 * i]),
                                    label="person", pos_std=0.05) for i in range(args.objects)])
    tracks = tracker.confirmed()

    rows = [
        ("stereo depth (SGBM)", timed(lambda: stereo.compute(gray, gray2))),
        ("optical flow (DIS)", timed(lambda: flow.dis.calc(gray, gray2, None))),
        (f"detection ({os.path.basename(str(args.model).rstrip('/'))})", timed(lambda: detector(img))),
        (f"prediction, {len(tracks)} people (social GRU)",
         timed(lambda: zones.build("learned", tracks, planner.times, predictor))),
    ]
    z = zones.build("learned", tracks, planner.times, predictor)
    rows.append((f"planning ({len(planner.cmds)} candidate paths)",
                 timed(lambda: planner.plan(np.zeros(3), 0.0, (4.0, 0.0), z))))
    total = sum(r[1] for r in rows)

    name = socket.gethostname()
    cpu = platform.processor() or platform.machine()
    os.makedirs("results", exist_ok=True)
    lines = [f"Speed on **{name}** ({cpu}, {os.cpu_count()} threads), {W}x{H}\n",
             "| stage | ms per frame |", "|---|---|"]
    lines += [f"| {n} | {t:.1f} |" for n, t in rows]
    lines += [f"| **total** | **{total:.1f}** ({1000 / total:.1f} fps) |"]
    text = "\n".join(lines)
    print(text)
    with open(f"results/speed_{name}.md", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
