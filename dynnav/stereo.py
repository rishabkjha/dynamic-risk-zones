"""Stereo depth: rectification, SGBM depth, and the ground position of a detection box.

Camera frame: x right, y down, z forward (OpenCV). Ground frame used everywhere else in the
project: x forward, y left, in metres, with the robot (camera) at the origin.
"""
import cv2
import numpy as np


class Rectifier:
    """Undistorts and rectifies webcam pairs with calib/stereo_calib.npz from 03_calibrate.py."""

    def __init__(self, calib_file, scale):
        c = np.load(calib_file)
        size = tuple(int(v) for v in c["size"])
        self.map_l = cv2.initUndistortRectifyMap(c["K1"], c["D1"], c["R1"], c["P1"], size, cv2.CV_16SC2)
        self.map_r = cv2.initUndistortRectifyMap(c["K2"], c["D2"], c["R2"], c["P2"], size, cv2.CV_16SC2)
        P1 = c["P1"]
        self.fx = float(P1[0, 0]) * scale
        self.cx = float(P1[0, 2]) * scale
        self.cy = float(P1[1, 2]) * scale
        self.baseline_m = float(c["baseline_mm"]) / 1000.0
        self.size = (int(size[0] * scale), int(size[1] * scale))

    def rectify(self, frame_l, frame_r):
        rl = cv2.remap(frame_l, *self.map_l, cv2.INTER_LINEAR)
        rr = cv2.remap(frame_r, *self.map_r, cv2.INTER_LINEAR)
        rl = cv2.resize(rl, self.size, interpolation=cv2.INTER_AREA)
        rr = cv2.resize(rr, self.size, interpolation=cv2.INTER_AREA)
        return rl, rr


class StereoDepth:
    """SGBM on a rectified grey pair -> depth in metres (0 = no reading).

    Raw SGBM only: hole-filling filters (WLS) invent depths, which turn into phantom obstacles.
    """

    def __init__(self, fx, baseline_m, num_disp=64, block=7, disparity_offset=0.0):
        self.fx = fx
        self.baseline_m = baseline_m
        self.disparity_offset = disparity_offset
        self.num_disp = num_disp
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disp, blockSize=block,
            P1=8 * block * block, P2=32 * block * block,
            disp12MaxDiff=2, preFilterCap=63, uniquenessRatio=10,
            speckleWindowSize=50, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def compute(self, gray_l, gray_r):
        disp = self.sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
        disp[:, :self.num_disp] = 0.0  # the left strip is only seen by one camera
        disp[disp > 0] -= self.disparity_offset
        depth = np.zeros_like(disp)
        valid = disp > 0.5
        depth[valid] = self.fx * self.baseline_m / disp[valid]
        return depth


def box_depth(depth, box, min_range=0.3, max_range=8.0):
    """Median depth of the central part of a box (x0, y0, x1, y1), or 0 with too few readings.

    The centre half of the box is used so the background around a person is left out.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    patch = depth[int(y0 + 0.25 * h):int(y1 - 0.25 * h), int(x0 + 0.25 * w):int(x1 - 0.25 * w)]
    patch = patch[(patch > min_range) & (patch < max_range)]
    return float(np.median(patch)) if patch.size >= 15 else 0.0


def size_depth(box, fx, real_height_m=1.7):
    """Distance from the box height alone (pinhole camera), for when there is no stereo."""
    h = box[3] - box[1]
    return fx * real_height_m / h if h > 5 else 0.0


def box_to_ground(box, z, fx, cx):
    """Box centre at distance z -> ground position (x forward, y left) in metres."""
    u = 0.5 * (box[0] + box[2])
    x_cam = (u - cx) * z / fx
    return np.array([z, -x_cam])
