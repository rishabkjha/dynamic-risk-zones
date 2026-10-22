"""Finds the calibration board in an image (ChArUco or plain chessboard, set in config.py)."""
import cv2
import numpy as np

import config

aruco = cv2.aruco
if config.BOARD_TYPE == "charuco":
    _dict = aruco.getPredefinedDictionary(getattr(aruco, config.CHARUCO_DICT))
    _board = aruco.CharucoBoard(config.CHARUCO_SQUARES, config.SQUARE_SIZE_MM,
                                config.MARKER_SIZE_MM, _dict)
    _detector = aruco.CharucoDetector(_board)
    MIN_CORNERS = 8
else:
    _cols, _rows = config.CHESSBOARD
    _grid = np.zeros((_cols * _rows, 3), np.float32)
    _grid[:, :2] = np.mgrid[0:_cols, 0:_rows].T.reshape(-1, 2) * config.SQUARE_SIZE_MM
    MIN_CORNERS = _cols * _rows

_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def detect(gray):
    """Returns {corner_id: (x, y)} for every board corner found, or {} if none."""
    if config.BOARD_TYPE == "charuco":
        corners, ids, _, _ = _detector.detectBoard(gray)
        if ids is None:
            return {}
        return {int(i): c.reshape(2) for i, c in zip(ids.flatten(), corners)}
    ok, corners = cv2.findChessboardCorners(
        gray, config.CHESSBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return {}
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _criteria)
    return {i: c.reshape(2) for i, c in enumerate(corners)}


def board_point(corner_id):
    """Position of a corner on the board, in mm."""
    if config.BOARD_TYPE == "charuco":
        return _board.getChessboardCorners()[corner_id]
    return _grid[corner_id]


def match_pair(found_l, found_r):
    """Corners seen in both images, as (board points, left points, right points), or None."""
    common = sorted(set(found_l) & set(found_r))
    if len(common) < MIN_CORNERS:
        return None
    obj = np.array([board_point(i) for i in common], np.float32).reshape(-1, 1, 3)
    pl = np.array([found_l[i] for i in common], np.float32).reshape(-1, 1, 2)
    pr = np.array([found_r[i] for i in common], np.float32).reshape(-1, 1, 2)
    return obj, pl, pr


def draw(img, found):
    for (x, y) in found.values():
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
