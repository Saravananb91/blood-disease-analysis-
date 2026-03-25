"""
module1_cell_detection/inference.py
-------------------------------------
Inference helpers for blood cell detection:
  - Run YOLOv8 predictions on test images
  - Convert raw detection counts to clinical metrics (/µL)
  - Interpret results against normal clinical ranges

Usage
-----
    python -m module1_cell_detection.inference --model /path/to/best.pt --img /path/to/image.jpg
"""

import argparse
import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

from configs.config import (
    M1_DATA_DIR, M1_MODEL_SAVE, M1_CONF_THRESH,
    M1_CLASS_NAMES, M1_NORMAL_RANGES
)


# ── Medical conversion ────────────────────────────────────────────────────────

def convert_to_medical_metrics(raw_counts: dict) -> dict:
    """
    Scale raw detection counts to estimated clinical counts per µL,
    assuming approximately 1 nL of blood per microscopy image.

    Parameters
    ----------
    raw_counts : {class_name: int}  detected counts per class

    Returns
    -------
    dict {class_name: int}  estimated cells/µL
    """
    return {k: round(v * 1e6) for k, v in raw_counts.items()}


def interpret_metric(name: str, value: float) -> str:
    """
    Return a human-readable clinical interpretation for a cell count.

    Parameters
    ----------
    name  : one of "RBC", "WBC", "Platelet"
    value : estimated count per µL

    Returns
    -------
    str  e.g. "RBC: NORMAL (4.80e+06/µL)"
    """
    if name not in M1_NORMAL_RANGES:
        return f"{name}: {value:.2e}/µL (no reference range)"
    low, high = M1_NORMAL_RANGES[name]
    if value < low:
        status = "LOW ⬇️"
    elif value > high:
        status = "HIGH ⬆️"
    else:
        status = "NORMAL ✅"
    return f"{name}: {status}  ({value:.2e}/µL)"


# ── Inference ─────────────────────────────────────────────────────────────────

def analyze_image(img_path: str, model: YOLO) -> dict:
    """
    Run detection on a single image and print clinical metrics.

    Parameters
    ----------
    img_path : path to the microscopy image
    model    : loaded YOLO model

    Returns
    -------
    dict with keys: raw_counts, medical_metrics, interpretations
    """
    res = model.predict(img_path, conf=M1_CONF_THRESH)
    cls = res[0].boxes.cls.cpu().numpy()

    unique, counts_arr = np.unique(cls, return_counts=True)
    raw_counts = {
        M1_CLASS_NAMES[int(u)]: int(c)
        for u, c in zip(unique, counts_arr)
    }

    medical_metrics   = convert_to_medical_metrics(raw_counts)
    interpretations   = [interpret_metric(k, v) for k, v in medical_metrics.items()]

    print(f"\n🔬 Image : {os.path.basename(img_path)}")
    print(f"🔍 Raw counts    : {raw_counts}")
    print(f"🩸 Clinical (/µL): {medical_metrics}")
    for line in interpretations:
        print(f"📊 {line}")

    return {
        "raw_counts":      raw_counts,
        "medical_metrics": medical_metrics,
        "interpretations": interpretations,
    }


def run_test_inference(
    model: YOLO,
    test_dir: str | None = None,
    n_images: int = 5,
) -> None:
    """
    Run inference on up to `n_images` test images and display results.

    Parameters
    ----------
    model    : loaded YOLO model
    test_dir : directory containing test images (defaults to config path)
    n_images : max number of images to process
    """
    test_dir   = test_dir or os.path.join(M1_DATA_DIR, "test", "images")
    test_imgs  = glob.glob(os.path.join(test_dir, "*"))[:n_images]

    if not test_imgs:
        print(f"⚠️  No test images found in {test_dir}")
        return

    for img_path in test_imgs:
        model.predict(img_path, conf=M1_CONF_THRESH, save=True)
        img = cv2.imread(img_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Detected Blood Cells")
        plt.axis("off")
        plt.show()
        analyze_image(img_path, model)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blood cell detection inference")
    parser.add_argument("--model", default=M1_MODEL_SAVE, help="Path to trained .pt model")
    parser.add_argument("--img",   default=None,          help="Single image path (optional)")
    parser.add_argument("--n",     type=int, default=5,   help="Number of test images to run")
    args = parser.parse_args()

    yolo_model = YOLO(args.model)
    if args.img:
        analyze_image(args.img, yolo_model)
    else:
        run_test_inference(yolo_model, n_images=args.n)
