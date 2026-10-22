"""Step 11: check the stereo distance against a tape measure.

    python 11_depth_check.py

Put something flat and textured (a box with print on it, a poster, the calibration board) at
1, 2, 3, 4 and 5 m from the cameras, measured with a tape from the lenses. The window shows the
left camera (rectified) and the depth image; the green cross is where the distance is measured
(median depth in a small square). Click to move the cross onto the object.
Keys 1-5: save the current reading as "true distance 1..5 m" to results/depth_check.csv.
Q: quit, and print the table (measured vs true, error) and write results/depth_check_latest.md.
It also prints the DISPARITY_OFFSET_PX that would make the readings match the tape; put that
in config.py (readings are saved with the offset in use at the time, which is stored with them).
"""
import csv
import os

import cv2
import numpy as np

import config
from dynnav.cams import StereoCams
from dynnav.stereo import Rectifier, StereoDepth

WINDOW = "depth check (click = move cross, 1-5 = save at that distance, Q = quit)"
HALF = 6                   # the measured square is (2*HALF+1) pixels wide at 320x240
OUT = "results/depth_check.csv"


def main():
    rect = Rectifier(config.CALIB_FILE, config.PROCESS_SCALE)
    stereo = StereoDepth(rect.fx, rect.baseline_m, disparity_offset=config.DISPARITY_OFFSET_PX)
    cams = StereoCams()
    point = [rect.size[0] // 2 + 32, rect.size[1] // 2]       # a little right of centre: the left strip has no depth

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            point[0] = int(np.clip(x % rect.size[0], HALF, rect.size[0] - HALF - 1))
            point[1] = int(np.clip(y, HALF, rect.size[1] - HALF - 1))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_click)
    rows = []
    readings = []
    while True:
        fl, fr = cams.read()
        if fl is None:
            continue
        left, right = rect.rectify(fl, fr)
        depth = stereo.compute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
        x, y = point
        patch = depth[y - HALF:y + HALF + 1, x - HALF:x + HALF + 1]
        patch = patch[patch > 0]
        z = float(np.median(patch)) if patch.size >= 10 else 0.0
        readings = (readings + [z])[-15:]                            # steady the number over ~1 s
        valid = [r for r in readings if r > 0]
        z_avg = float(np.median(valid)) if valid else 0.0

        view = left.copy()
        vis = cv2.applyColorMap(np.clip(depth * 40, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[depth <= 0] = 0
        for img in (view, vis):
            cv2.rectangle(img, (x - HALF, y - HALF), (x + HALF, y + HALF), (0, 255, 0), 1)
            cv2.drawMarker(img, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
        screen = cv2.resize(np.hstack([view, vis]), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        text = f"distance {z_avg:.2f} m" if z_avg > 0 else "no depth here (needs texture)"
        cv2.putText(screen, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(screen, f"saved: {len(rows)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow(WINDOW, screen)

        key = cv2.waitKey(1) & 0xFF
        if ord("1") <= key <= ord("9") and z_avg > 0:
            true = float(chr(key))
            rows.append((true, z_avg))
            print(f"true {true:.1f} m  measured {z_avg:.3f} m  error {100 * (z_avg - true):+.1f} cm")
        elif key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cams.release()
    cv2.destroyAllWindows()
    if not rows:
        return
    os.makedirs("results", exist_ok=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["true_m", "measured_m", "offset_px"])
        w.writerows([(t, m, config.DISPARITY_OFFSET_PX) for t, m in rows])
    with open(OUT) as f:
        table = list(csv.reader(f))[1:]
    allrows = [(float(r[0]), float(r[1])) for r in table if float(r[2] if len(r) > 2 else 0) == config.DISPARITY_OFFSET_PX]
    # Disparity each reading had before any offset, against the disparity the tape distance needs.
    fb = rect.fx * rect.baseline_m
    raw = [(float(r[0]), fb / float(r[1]) + float(r[2] if len(r) > 2 else 0)) for r in table]
    best = float(np.mean([d - fb / t for t, d in raw]))
    print(f"\nDISPARITY_OFFSET_PX that fits all {len(raw)} readings: {best:.2f} (config.py has "
          f"{config.DISPARITY_OFFSET_PX:.2f})")
    lines = ["| true distance (m) | stereo (m) | error (cm) | error (%) |", "|---|---|---|---|"]
    for t, m in sorted(allrows):
        lines.append(f"| {t:.1f} | {m:.3f} | {100 * (m - t):+.1f} | {100 * (m - t) / t:+.1f}% |")
    text = "\n".join(lines)
    print(text)
    with open("results/depth_check_latest.md", "w") as f:
        f.write(f"Stereo distance vs tape measure (baseline {rect.baseline_m * 1000:.1f} mm, "
                f"focal {rect.fx:.1f} px at {rect.size[0]}x{rect.size[1]}, disparity offset "
                f"{config.DISPARITY_OFFSET_PX:.2f} px)\n\n" + text + "\n")


if __name__ == "__main__":
    main()
