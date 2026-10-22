"""Opens the two webcams and reads frame pairs as close in time as possible (Linux and Windows)."""
import sys

import cv2

import config


def open_camera(index):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [cv2.CAP_V4L2, cv2.CAP_ANY]
    cap = None
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            break
    if cap is None or not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Fixed focus stops the lens changing after calibration (ignored if unsupported).
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return cap


class StereoCams:
    def __init__(self, left=config.LEFT_CAM, right=config.RIGHT_CAM):
        self.left = open_camera(left)
        self.right = open_camera(right)
        if self.left is None or self.right is None:
            raise RuntimeError(
                f"Could not open cameras {left} and {right}. "
                "Run 01_check_cameras.py and set LEFT_CAM / RIGHT_CAM in config.py.")

    def read(self):
        # grab() both first, then decode, so the two images are taken close together.
        ok_l = self.left.grab()
        ok_r = self.right.grab()
        if not (ok_l and ok_r):
            return None, None
        _, frame_l = self.left.retrieve()
        _, frame_r = self.right.retrieve()
        return frame_l, frame_r

    def release(self):
        self.left.release()
        self.right.release()


class VideoPair:
    """Replays a stereo recording (left.mp4 + right.mp4) as if it were the cameras."""

    def __init__(self, left_path, right_path):
        self.left = cv2.VideoCapture(left_path)
        self.right = cv2.VideoCapture(right_path)
        if not (self.left.isOpened() and self.right.isOpened()):
            raise RuntimeError(f"Could not open {left_path} / {right_path}")
        self.fps = self.left.get(cv2.CAP_PROP_FPS) or 15.0

    def read(self):
        ok_l, fl = self.left.read()
        ok_r, fr = self.right.read()
        if not (ok_l and ok_r):
            return None, None
        return fl, fr

    def release(self):
        self.left.release()
        self.right.release()
