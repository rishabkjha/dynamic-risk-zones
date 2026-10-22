"""Step 8: export YOLO11n to the fastest format for the computer it runs on.

    python 08_export_model.py --format openvino    # Intel laptop / NUC (CPU or integrated GPU)
    python 08_export_model.py --format ncnn        # Raspberry Pi 4 / other ARM boards
    python 08_export_model.py --format onnx        # anything with onnxruntime

Then set DETECTOR_MODEL in config.py to the folder or file it prints. Run 09_speed_test.py
before and after to see the difference.
"""
import argparse

from ultralytics import YOLO

import config

ap = argparse.ArgumentParser()
ap.add_argument("--format", default="openvino", choices=["openvino", "ncnn", "onnx"])
ap.add_argument("--model", default="yolo11n.pt")
ap.add_argument("--half", action="store_true", help="FP16 weights (OpenVINO on the integrated GPU)")
args = ap.parse_args()

path = YOLO(args.model).export(format=args.format, imgsz=config.DETECT_SIZE, half=args.half)
print(f'\nDone. In config.py set:\nDETECTOR_MODEL = "{path}"')
