"""Step 1: find which camera numbers your two webcams have and check left/right.

On Linux, `v4l2-ctl --list-devices` also lists them (use the first /dev/videoN of each webcam).

Shows the two cameras side by side. Cover the LEFT lens with your hand:
the left half of the window should go dark. If the right half goes dark, press S to swap.
Keys: S = swap, Q = quit (prints the numbers to put in config.py).
"""
import cv2
import numpy as np

import config
from dynnav.cams import open_camera

# Hide OpenCV's warnings for camera numbers that don't exist.
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

found = []
for i in range(10):
    cap = open_camera(i)
    if cap is not None:
        ok, frame = cap.read()
        if ok:
            found.append(i)
            print(f"Camera {i}: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()

print("Cameras found:", found)
if len(found) < 2:
    raise SystemExit("Need two cameras. Plug both webcams into different USB ports and try again.")

left, right = config.LEFT_CAM, config.RIGHT_CAM
if left not in found or right not in found:
    left, right = found[-2], found[-1]  # a built-in laptop camera is usually the lowest number

cap_l, cap_r = open_camera(left), open_camera(right)
print("Click on the video window, then press S to swap or Q to finish.")
while True:
    cap_l.grab(); cap_r.grab()
    _, fl = cap_l.retrieve(); _, fr = cap_r.retrieve()
    if fl is None or fr is None:
        print("A camera stopped sending frames (USB bandwidth?). Try other USB ports.")
        break
    view = np.hstack([fl, fr])
    cv2.putText(view, f"LEFT = cam {left}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(view, f"RIGHT = cam {right}", (fl.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("check cameras (S swap, Q quit)", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        left, right = right, left
        cap_l, cap_r = cap_r, cap_l
    elif key == ord("q") or cv2.getWindowProperty("check cameras (S swap, Q quit)", cv2.WND_PROP_VISIBLE) < 1:
        break

cap_l.release(); cap_r.release()
cv2.destroyAllWindows()
print(f"\nPut these in config.py:\nLEFT_CAM = {left}\nRIGHT_CAM = {right}")
