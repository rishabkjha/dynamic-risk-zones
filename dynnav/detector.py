"""Object detection: YOLO11n through Ultralytics, with OpenCV's HOG person detector as a fallback.

Ultralytics loads every exported format from the same call, so the same code runs the PyTorch
model on any PC, the OpenVINO export on an Intel laptop or NUC, and the NCNN export on a Pi 4.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    box: tuple          # x0, y0, x1, y1 in pixels
    label: str
    score: float


class YoloDetector:
    def __init__(self, model, size=320, conf=0.4):
        from ultralytics import YOLO
        self.model = YOLO(model, task="detect")
        self.size = size
        self.conf = conf

    def __call__(self, bgr):
        r = self.model.predict(bgr, imgsz=self.size, conf=self.conf, verbose=False)[0]
        names = r.names
        out = []
        for (x0, y0, x1, y1), c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                          r.boxes.conf.cpu().numpy()):
            out.append(Detection((float(x0), float(y0), float(x1), float(y1)), names[int(c)], float(s)))
        return out


class HogDetector:
    """People only, no ML training needed; much weaker than YOLO but has no dependencies."""

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def __call__(self, bgr):
        rects, weights = self.hog.detectMultiScale(bgr, winStride=(8, 8), padding=(8, 8), scale=1.05)
        return [Detection((float(x), float(y), float(x + w), float(y + h)), "person", float(s))
                for (x, y, w, h), s in zip(rects, np.ravel(weights)) if s > 0.5]


def make_detector(model, size, conf):
    try:
        return YoloDetector(model, size, conf)
    except ImportError:
        print("ultralytics is not installed: using the HOG person detector instead "
              "(pip install ultralytics for YOLO).")
        return HogDetector()
