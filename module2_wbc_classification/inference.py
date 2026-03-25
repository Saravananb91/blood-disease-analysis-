"""
module2_wbc_classification/inference.py
-----------------------------------------
Predict WBC subtype from a single microscopy image using the trained ResNet-50.

Usage
-----
    python -m module2_wbc_classification.inference --model /path/to/model.h5 --img /path/to/cell.jpg
"""

import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from configs.config import M2_MODEL_FINAL, M2_IMG_SIZE, M2_CLASS_NAMES


def predict_wbc(img_path: str, model_path: str = M2_MODEL_FINAL) -> dict:
    """
    Predict the WBC subtype for a single image.

    Parameters
    ----------
    img_path   : path to the WBC microscopy image
    model_path : path to the saved .h5 model

    Returns
    -------
    dict with keys: predicted_class (str), confidence (float), all_scores (list)
    """
    model = load_model(model_path)

    img_arr = image.load_img(img_path, target_size=M2_IMG_SIZE)
    img_arr = image.img_to_array(img_arr) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    preds      = model.predict(img_arr)[0]
    pred_idx   = int(np.argmax(preds))
    pred_label = M2_CLASS_NAMES[pred_idx]
    confidence = float(np.max(preds)) * 100

    print(f"🔬 WBC Subtype : {pred_label}")
    print(f"📊 Confidence  : {confidence:.2f}%")

    return {
        "predicted_class": pred_label,
        "confidence":      confidence,
        "all_scores":      {M2_CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(preds)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WBC subtype classification inference")
    parser.add_argument("--model", default=M2_MODEL_FINAL, help="Path to .h5 model")
    parser.add_argument("--img",   required=True,          help="Path to input image")
    args = parser.parse_args()

    predict_wbc(args.img, args.model)
