"""Step 2: take 20-30 photo pairs of the calibration board.

Hold the board (flat, taped to cardboard) where both cameras see it.
When both cameras see the board AND it has been still for a moment, the text turns green;
press SPACE to save the pair. The two webcams are not synchronised, so a pair taken while
anything moves gives a bad calibration. Rest the board (or the camera rig) on something.
Every pair must show the board in a NEW position: near, far, left, right, corners of the
image, and tilted different ways. The same position saved many times breaks calibration.
Keys: SPACE = save pair, Q = quit.
"""
import os

import cv2
import numpy as np

from dynnav import board
import config
from dynnav.cams import StereoCams

os.makedirs(config.CALIB_DIR, exist_ok=True)
count = len([f for f in os.listdir(config.CALIB_DIR) if f.startswith("left_")])
WINDOW = "calibration capture (SPACE save, Q quit)"
STILL_PX = 1.0       # corners moved less than this between frames = still
STILL_FRAMES = 10    # frames in a row the board must be still before saving
NEW_POSE_PX = 40     # corners must have moved this far since the last saved pair


def movement(now, before):
    common = set(now) & set(before)
    if not common:
        return 1e9
    return max(float(np.abs(now[i] - before[i]).max()) for i in common)


prev_l, prev_r, still = {}, {}, 0
last_saved = {}

cams = StereoCams()
while True:
    fl, fr = cams.read()
    if fl is None:
        continue
    found_l = board.detect(cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY))
    found_r = board.detect(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    seen = board.match_pair(found_l, found_r) is not None
    if seen and max(movement(found_l, prev_l), movement(found_r, prev_r)) < STILL_PX:
        still += 1
    else:
        still = 0
    prev_l, prev_r = found_l, found_r
    new_pose = movement(found_l, last_saved) > NEW_POSE_PX
    good = seen and still >= STILL_FRAMES and new_pose

    view_l, view_r = fl.copy(), fr.copy()
    board.draw(view_l, found_l)
    board.draw(view_r, found_r)
    view = np.hstack([view_l, view_r])
    if good:
        status = "STILL - press SPACE"
    elif seen and not new_pose:
        status = "move/tilt board to a NEW position"
    elif seen:
        status = "hold still..."
    else:
        status = "move board so both cameras see it"
    cv2.putText(view, f"{status}   corners {len(found_l)}/{len(found_r)}   saved: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if good else (0, 0, 255), 2)
    cv2.imshow(WINDOW, view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") and good:
        cv2.imwrite(os.path.join(config.CALIB_DIR, f"left_{count:02d}.png"), fl)
        cv2.imwrite(os.path.join(config.CALIB_DIR, f"right_{count:02d}.png"), fr)
        count += 1
        still = 0
        last_saved = found_l
        print("saved pair", count)
    elif key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break

cams.release()
cv2.destroyAllWindows()
print(f"{count} pairs in {config.CALIB_DIR}/. Aim for 20-30, then run 03_calibrate.py")
