"""
utils/plot_utils.py
-------------------
Shared plotting helpers used across all three modules.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_yolo_label(img_path: str, label_path: str) -> None:
    """
    Visualise a YOLO bounding-box annotation overlaid on its image.

    Parameters
    ----------
    img_path   : path to the .jpg image
    label_path : path to the corresponding YOLO .txt label file
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    h, w, _ = img.shape
    with open(label_path, "r") as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.strip().split())
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("YOLO Label Visualisation")
    plt.tight_layout()
    plt.show()


def plot_training_history(history, title_prefix: str = "") -> None:
    """
    Plot accuracy and loss curves from a Keras training History object.

    Parameters
    ----------
    history      : keras History object returned by model.fit()
    title_prefix : optional label prepended to each chart title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title(f"{title_prefix} Accuracy".strip())
    axes[0].legend()
    axes[0].set_xlabel("Epoch")

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title(f"{title_prefix} Loss".strip())
    axes[1].legend()
    axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.show()
