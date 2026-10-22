"""Dense optical flow (OpenCV DIS) and the sideways speed of a detected object.

Stereo depth gives an object's distance, but a few centimetres of depth noise per frame make
its speed jumpy. Optical flow measures how far the object moved across the image between two
frames very precisely; with the distance from stereo that becomes a sideways speed in m/s.
The tracker fuses both.
"""
import cv2
import numpy as np


class Flow:
    def __init__(self):
        # The "fast" preset runs at well over 100 fps at 320x240 on a laptop CPU.
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.prev = None

    def update(self, gray):
        """Flow from the previous frame to this one (H x W x 2, pixels), or None on the first frame."""
        prev, self.prev = self.prev, gray
        if prev is None or prev.shape != gray.shape:
            return None
        return self.dis.calc(prev, gray, None)


def background_flow(flow, boxes):
    """Median flow outside all boxes: the image motion caused by the camera itself turning."""
    mask = np.ones(flow.shape[:2], bool)
    for x0, y0, x1, y1 in boxes:
        mask[max(int(y0), 0):int(y1), max(int(x0), 0):int(x1)] = False
    if mask.sum() < 100:
        return np.zeros(2, np.float32)
    return np.median(flow[mask], axis=0)


def box_flow(flow, box):
    """Median flow inside the centre of a box, in pixels per frame."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    patch = flow[int(y0 + 0.25 * h):int(y1 - 0.25 * h), int(x0 + 0.25 * w):int(x1 - 0.25 * w)]
    if patch.size == 0:
        return np.zeros(2, np.float32)
    return np.median(patch.reshape(-1, 2), axis=0)


def lateral_speed(du_px, z, fx, dt):
    """Sideways image motion (pixels per frame) at distance z -> speed to the LEFT in m/s."""
    return -du_px * z / fx / dt


def flow_colour(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 30, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
