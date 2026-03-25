"""
module3_disease_detection/inference.py
----------------------------------------
Predict blood disease from a microscopy image using the trained EfficientNetB3.

Usage
-----
    python -m module3_disease_detection.inference --model /path/model.h5 --img /path/img.png
"""

import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from configs.config import M3_MODEL_FINAL, M3_IMG_SIZE


def predict_disease(img_path: str, class_indices: dict, model_path: str = M3_MODEL_FINAL) -> dict:
    """
    Predict the blood disease class for a single image.

    Parameters
    ----------
    img_path      : path to the microscopy image
    class_indices : dict mapping class name → integer index
                    (obtain from train_gen.class_indices after training)
    model_path    : path to the saved .h5 model

    Returns
    -------
    dict with keys: predicted_class, confidence, all_scores
    """
    model = load_model(model_path)

    img_arr = image.load_img(img_path, target_size=(M3_IMG_SIZE, M3_IMG_SIZE))
    img_arr = image.img_to_array(img_arr) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    preds      = model.predict(img_arr)[0]
    idx_to_cls = {v: k for k, v in class_indices.items()}
    pred_label = idx_to_cls[int(np.argmax(preds))]
    confidence = float(np.max(preds)) * 100

    print(f"🩸 Predicted Disease : {pred_label}")
    print(f"📊 Confidence        : {confidence:.2f}%")

    all_scores = {
        idx_to_cls[i]: round(float(p) * 100, 2)
        for i, p in enumerate(preds)
    }
    return {
        "predicted_class": pred_label,
        "confidence":      confidence,
        "all_scores":      all_scores,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disease detection inference")
    parser.add_argument("--model", default=M3_MODEL_FINAL, help="Path to .h5 model")
    parser.add_argument("--img",   required=True,          help="Path to input image")
    args = parser.parse_args()

    # When running from CLI, load class_indices from a saved JSON if available
    import json, os
    indices_path = os.path.join(os.path.dirname(args.model), "class_indices.json")
    if os.path.exists(indices_path):
        with open(indices_path) as f:
            class_indices = json.load(f)
    else:
        print("⚠️  class_indices.json not found — pass class_indices manually.")
        class_indices = {}

    predict_disease(args.img, class_indices, args.model)
