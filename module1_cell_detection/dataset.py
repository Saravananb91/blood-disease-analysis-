"""
module1_cell_detection/dataset.py
----------------------------------
Dataset validation and YAML generation for YOLOv8 training.
"""

import os
import glob

from configs.config import (
    M1_DATA_DIR, M1_YAML_PATH, M1_CLASS_NAMES
)


def check_dataset_structure(img_dir: str, lbl_dir: str) -> dict:
    """
    Verify image-label pairing for a YOLO split directory.

    Parameters
    ----------
    img_dir : directory containing .jpg images
    lbl_dir : directory containing .txt YOLO labels

    Returns
    -------
    dict with keys: total_images, total_labels, missing_labels (list)
    """
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    lbl_files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))

    missing = []
    for img in img_files:
        lbl_name = os.path.splitext(os.path.basename(img))[0] + ".txt"
        if not os.path.exists(os.path.join(lbl_dir, lbl_name)):
            missing.append(lbl_name)

    result = {
        "total_images":  len(img_files),
        "total_labels":  len(lbl_files),
        "missing_labels": missing,
    }
    print(f"✅ Total Images : {result['total_images']}")
    print(f"✅ Total Labels : {result['total_labels']}")
    if missing:
        print(f"⚠️  Missing Labels: {len(missing)} → {missing[:5]}")
    else:
        print("✅ All labels found!")
    return result


def create_yaml(base_path: str = M1_DATA_DIR, yaml_path: str = M1_YAML_PATH) -> str:
    """
    Write a YOLO dataset YAML file for the blood-cell detection task.

    Parameters
    ----------
    base_path : root directory of the dataset
    yaml_path : output path for the .yaml file

    Returns
    -------
    str : path to the written YAML file
    """
    content = (
        f"path: {base_path}\n"
        f"train: train/images\n"
        f"val:   valid/images\n\n"
        f"nc: {len(M1_CLASS_NAMES)}\n"
        f"names: {M1_CLASS_NAMES}\n"
    )
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"✅ YAML written to: {yaml_path}")
    return yaml_path


def validate_all_splits() -> None:
    """Run check_dataset_structure for train and validation splits."""
    for split in ("train", "valid"):
        img_dir = os.path.join(M1_DATA_DIR, split, "images")
        lbl_dir = os.path.join(M1_DATA_DIR, split, "labels")
        print(f"\n── {split.upper()} split ──")
        check_dataset_structure(img_dir, lbl_dir)
