"""Step 3: compute the stereo calibration from the saved board pairs.

Prints the error (RMS, in pixels) and the distance between the cameras.
Good result: RMS under about 0.5 px, and a baseline close to what you measure with a ruler.
Saves everything depth needs to stereo_calib.npz.
"""
import glob
import os

import cv2
import numpy as np

from dynnav import board
import config

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
obj_pts, pts_l, pts_r, names = [], [], [], []
size = None

for path_l in sorted(glob.glob(os.path.join(config.CALIB_DIR, "left_*.png"))):
    path_r = path_l.replace("left_", "right_")
    gl = cv2.imread(path_l, cv2.IMREAD_GRAYSCALE)
    gr = cv2.imread(path_r, cv2.IMREAD_GRAYSCALE)
    if gl is None or gr is None:
        continue
    size = gl.shape[::-1]
    pair = board.match_pair(board.detect(gl), board.detect(gr))
    if pair is None:
        print("skipped", os.path.basename(path_l))
        continue
    obj_pts.append(pair[0])
    pts_l.append(pair[1])
    pts_r.append(pair[2])
    names.append(os.path.basename(path_l))

print(f"Using {len(obj_pts)} pairs")
if len(obj_pts) < 10:
    raise SystemExit("Need at least 10 good pairs. Run 02_capture_calibration.py again.")

# Calibrate each camera on its own, then the pair. Webcam lenses have little distortion,
# so the 3rd distortion term is fixed at 0: fitting it makes rectification zoom in wildly.
rms_l, K1, D1, _, _ = cv2.calibrateCamera(obj_pts, pts_l, size, None, None, flags=cv2.CALIB_FIX_K3)
rms_r, K2, D2, _, _ = cv2.calibrateCamera(obj_pts, pts_r, size, None, None, flags=cv2.CALIB_FIX_K3)

# Stereo calibration. Both cameras' own settings are refined together with the pair, which
# fixes focal lengths that one camera's photos alone could not pin down. A pair taken while the board (or the rig) was moving has the two
# cameras seeing it at slightly different moments, which ruins the result. Those pairs show
# up with a large error, so they are dropped and the calibration is repeated.
MAX_PAIR_ERROR_PX = 1.0
while True:
    rms, K1, D1, K2, D2, R, T, _, _, _, _, errors = cv2.stereoCalibrateExtended(
        obj_pts, pts_l, pts_r, K1, D1, K2, D2, size, None, None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3, criteria=criteria)
    pair_err = errors.mean(axis=1)
    worst = int(np.argmax(pair_err))
    if pair_err[worst] <= MAX_PAIR_ERROR_PX or len(obj_pts) <= 10:
        break
    print(f"dropped {names[worst]} (error {pair_err[worst]:.1f} px, probably moving)")
    for lst in (obj_pts, pts_l, pts_r, names):
        del lst[worst]
print(f"Stereo pairs kept: {len(obj_pts)}")

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, alpha=0)

baseline_mm = float(np.linalg.norm(T))
print(f"Left camera RMS  {rms_l:.3f} px")
print(f"Right camera RMS {rms_r:.3f} px")
print(f"Stereo RMS       {rms:.3f} px   (under ~0.5 is good, over 1.0 = recapture)")
print(f"Baseline         {baseline_mm:.1f} mm  (check with a ruler)")
print(f"Focal length     {P1[0, 0]:.1f} px  (640x480 webcams: usually 500-900)")

# A low RMS can still be a wrong answer when the photos all show the board in nearly the same
# position: then the focal length and baseline cannot be worked out.
if not (200 < P1[0, 0] < 3000) or not (20 < baseline_mm < 500):
    raise SystemExit(
        "\nCalibration FAILED: focal length or baseline is impossible, so the photos are too similar.\n"
        "Delete the calib_images folder and run 02_capture_calibration.py again, putting the board\n"
        "in a different place for every pair (near, far, every corner of the image, tilted).")

if rms > 1.0:
    print("\nWARNING: stereo RMS is over 1 px, so depth will be inaccurate. Usual causes: the two\n"
          "cameras shifted relative to each other (fix them rigidly), or too few tilted views.")

np.savez(config.CALIB_FILE, K1=K1, D1=D1, K2=K2, D2=D2, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
         size=np.array(size), baseline_mm=baseline_mm)
print("Saved", config.CALIB_FILE, "- now run 04_live_zones.py")
