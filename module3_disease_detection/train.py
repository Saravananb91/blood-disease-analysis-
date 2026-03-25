"""
module3_disease_detection/train.py
------------------------------------
Blood disease detection using EfficientNetB3 with fine-tuning.
Dataset split is done automatically via validation_split on the base directory.

Usage
-----
    python -m module3_disease_detection.train
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, CSVLogger
)

from configs.config import (
    M3_DATA_DIR, M3_IMG_SIZE, M3_BATCH_SIZE, M3_EPOCHS,
    M3_LR, M3_VAL_SPLIT, M3_FINE_TUNE_LAYERS,
    M3_MODEL_BEST, M3_MODEL_FINAL, M3_CSV_LOG
)
from utils.plot_utils import plot_training_history


# ── Data generators ───────────────────────────────────────────────────────────

def build_generators():
    """
    Build train / validation generators using an 85/15 split of base_path.
    Heavy augmentation is applied to the training set only.

    Returns
    -------
    (train_gen, valid_gen)
    """
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=M3_VAL_SPLIT,
    )

    common = dict(
        directory=M3_DATA_DIR,
        target_size=(M3_IMG_SIZE, M3_IMG_SIZE),
        batch_size=M3_BATCH_SIZE,
        class_mode="categorical",
    )

    train_gen = datagen.flow_from_directory(subset="training",   **common)
    valid_gen = datagen.flow_from_directory(subset="validation", **common)
    return train_gen, valid_gen


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, lr: float = M3_LR) -> Sequential:
    """
    Build an EfficientNetB3 model with the last `M3_FINE_TUNE_LAYERS`
    layers unfrozen for fine-tuning.

    Parameters
    ----------
    num_classes : number of disease classes inferred from the dataset
    lr          : Adam learning rate

    Returns
    -------
    Compiled Keras Sequential model
    """
    base = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(M3_IMG_SIZE, M3_IMG_SIZE, 3),
    )

    # Freeze early layers, fine-tune the last N
    for layer in base.layers[:-M3_FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in base.layers[-M3_FINE_TUNE_LAYERS:]:
        layer.trainable = True

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dropout(0.4),
        Dense(512, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> None:
    """Full training pipeline: data → model → fit → plot → save."""
    train_gen, valid_gen = build_generators()
    num_classes = train_gen.num_classes
    print(f"📁 Detected {num_classes} disease classes: {list(train_gen.class_indices.keys())}")

    model = build_model(num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ModelCheckpoint(M3_MODEL_BEST, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
        CSVLogger(M3_CSV_LOG, append=True),
    ]

    print(f"\n🚀 Training EfficientNetB3 for up to {M3_EPOCHS} epochs…")
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=M3_EPOCHS,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True,
    )

    plot_training_history(history, title_prefix="Module 3 – Disease Detection")

    model.save(M3_MODEL_FINAL)
    print(f"✅ Final model saved to: {M3_MODEL_FINAL}")


if __name__ == "__main__":
    train()
