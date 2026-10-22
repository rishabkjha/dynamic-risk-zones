"""Settings shared by every script. Change these to match your setup."""

# ---------------------------------------------------------------- cameras
# Camera numbers. On Linux each USB webcam usually shows up as two /dev/video nodes and only
# the even one gives images, so two webcams are often 0 and 2 (or 2 and 4 on a laptop with
# a built-in camera). Run 01_check_cameras.py to find them.
LEFT_CAM = 4
RIGHT_CAM = 2

# 640x480 MJPG keeps two webcams within USB bandwidth.
WIDTH = 640
HEIGHT = 480

# Depth, flow and detection run at this scale of the frame (0.5 = 320x240) for speed.
PROCESS_SCALE = 0.5

# ---------------------------------------------------------------- calibration board
# "charuco" (checkerboard with markers in the white squares) or "chessboard".
BOARD_TYPE = "charuco"
SQUARE_SIZE_MM = 30.0           # one black square on your print, measured with a ruler
CHARUCO_SQUARES = (5, 7)        # your board: 5 x 7 squares
CHARUCO_DICT = "DICT_5X5_50"
MARKER_SIZE_MM = 22.0           # one marker (the black pattern inside a white square)
CHESSBOARD = (9, 6)             # plain chessboard: INNER corners

CALIB_DIR = "calib/images"
CALIB_FILE = "calib/stereo_calib.npz"

# Without a calibration the live demo still runs: distance then comes from the height of the
# detection box (a person is about 1.7 m tall) and this guessed focal length, at 640x480.
GUESS_FOCAL_PX = 600.0

# ---------------------------------------------------------------- detection
# Any Ultralytics model path works: yolo11n.pt (PyTorch), yolo11n_openvino_model/ (Intel
# laptop or NUC, fastest there), yolo11n_ncnn_model/ (Raspberry Pi 4). See 08_export_model.py.
DETECTOR_MODEL = "yolo11n.pt"
DETECT_SIZE = 320               # YOLO input size; 320 is enough at 320x240
DETECT_CONF = 0.4
DETECT_EVERY = 1                # run YOLO every N frames (2 on a Pi 4); tracking fills the gaps

# Per class: (typical speed, maximum speed) in m/s and body radius in m.
# A speed of 0 means the object does not move by itself (chair, table...): a plain obstacle.
CLASS_MOTION = {
    "person":     (1.4, 2.5, 0.30),
    "bicycle":    (4.0, 8.0, 0.60),
    "motorcycle": (6.0, 15.0, 0.80),
    "car":        (5.0, 15.0, 1.20),
    "bus":        (5.0, 15.0, 1.50),
    "truck":      (5.0, 15.0, 1.50),
    "dog":        (1.5, 6.0, 0.35),
    "cat":        (1.0, 4.0, 0.25),
    "sports ball": (3.0, 10.0, 0.15),
}
DEFAULT_MOTION = (0.0, 0.0, 0.30)   # every other class: treated as static
# Classes that are drawn but never treated as obstacles (things on a desk, the floor...).
IGNORE_CLASSES = {"tv", "laptop", "cell phone", "book", "mouse", "keyboard", "remote"}

# ---------------------------------------------------------------- tracking and prediction
TRACK_GATE_M = 1.0              # a detection further than this from a track starts a new one
TRACK_MAX_AGE_S = 1.0           # a track not seen for this long is dropped
TRACK_MIN_HITS = 3              # detections before a track counts

PRED_DT = 0.4                   # predictor time step (the ETH/UCY datasets are at 2.5 Hz)
PRED_OBS = 8                    # past steps it looks at (3.2 s)
PRED_STEPS = 12                 # future steps it predicts (4.8 s)
PREDICTOR_FILE = "models/predictor.npz"
PREDICTOR_NOSOCIAL_FILE = "models/predictor_nosocial.npz"   # ablation: no neighbour input

# ---------------------------------------------------------------- zones and planning
# How moving objects become no-go areas:
#   static  = only where the object is now
#   reach   = rings growing with the object's class speed (where it COULD be)
#   cv      = constant-velocity prediction with growing uncertainty
#   learned = the GRU predictor (where it WILL LIKELY be, with its uncertainty)
#   learned_nosocial = the same GRU without the neighbour input (ablation)
ZONE_METHOD = "learned"
ZONE_COSTS = (1.0, 0.5, 0.15)   # inner (no-go), middle, outer zone
ZONE_SIGMAS = (1.0, 2.0, 3.0)   # cv / learned: zone edges at 1, 2, 3 standard deviations

ROBOT_RADIUS = 0.25
ROBOT_MAX_V = 0.8               # m/s
ROBOT_MAX_W = 1.5               # rad/s
ROBOT_MAX_ACC = 1.0             # m/s^2
PLAN_HORIZON = 3.0              # s
PLAN_DT = 0.2                   # s, also how often the plan is refreshed

# Live demo: the robot (the camera rig) stands at (0, 0) facing +x; the goal is this far ahead.
LIVE_GOAL = (4.0, 0.0)
