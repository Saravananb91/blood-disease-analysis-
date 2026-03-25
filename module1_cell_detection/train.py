"""
module1_cell_detection/train.py
--------------------------------
YOLOv8 training script for blood cell detection (RBC / WBC / Platelets).

Usage
-----
    python -m module1_cell_detection.train
"""

from ultralytics import YOLO

from configs.config import (
    M1_YAML_PATH, M1_YOLO_BASE, M1_EPOCHS,
    M1_IMG_SIZE, M1_BATCH, M1_MODEL_SAVE, WORKING_DIR
)
from module1_cell_detection.dataset import create_yaml, validate_all_splits


def train(
    yaml_path: str  = M1_YAML_PATH,
    base_model: str = M1_YOLO_BASE,
    epochs: int     = M1_EPOCHS,
    imgsz: int      = M1_IMG_SIZE,
    batch: int      = M1_BATCH,
) -> YOLO:
    """
    Train YOLOv8s on the blood-cell detection dataset.

    Parameters
    ----------
    yaml_path  : path to the dataset YAML
    base_model : pretrained YOLOv8 checkpoint name
    epochs     : number of training epochs
    imgsz      : input image size
    batch      : batch size

    Returns
    -------
    Trained YOLO model instance
    """
    # Validate dataset before training
    validate_all_splits()

    # Write YAML if it does not exist yet
    create_yaml(yaml_path=yaml_path)

    print(f"\n🚀 Starting YOLOv8 training for {epochs} epochs…")
    model = YOLO(base_model)
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=WORKING_DIR,
        name="bloodcell_detection",
        device=0,
    )
    return model


def save_model(model: YOLO, path: str = M1_MODEL_SAVE) -> None:
    """Persist a trained YOLO model to disk."""
    model.save(path)
    print(f"✅ Model saved to: {path}")


if __name__ == "__main__":
    trained_model = train()
    save_model(trained_model)
