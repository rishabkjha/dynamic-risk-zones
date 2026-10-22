"""Plays a video from results/ with OpenCV, for when no video player is installed.

    python play_video.py results/live_comparison.mp4

Keys: SPACE = pause / continue, Q = quit.
"""
import sys

import cv2

path = sys.argv[1] if len(sys.argv) > 1 else "results/live_comparison.mp4"
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    raise SystemExit(f"Could not open {path}")
delay = int(1000 / (cap.get(cv2.CAP_PROP_FPS) or 10))
paused = False
frame = None
while True:
    if not paused or frame is None:
        ok, frame = cap.read()
        if not ok:
            break
    cv2.imshow(path, frame)
    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q") or cv2.getWindowProperty(path, cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == ord(" "):
        paused = not paused
cap.release()
cv2.destroyAllWindows()
