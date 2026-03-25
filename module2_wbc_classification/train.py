"""
module2_wbc_classification/train.py
-------------------------------------
WBC subtype classification using ResNet-50 transfer learning.
Classes: Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil

Usage
-----
    python -m module2_wbc_classification.train
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from configs.config import (
    M2_DATA_DIR, M2_IMG_SIZE, M2_BATCH_SIZE,
    M2_NUM_CLASSES, M2_EPOCHS, M2_LR,
    M2_MODEL_BEST, M2_MODEL_FINAL
)
from utils.plot_utils import plot_training_history


# ── Data generators ───────────────────────────────────────────────────────────

def build_generators():
    """
    Create train / validation / test data generators with augmentation.

    Returns
    -------
    (train_generator, valid_generator, test_generator)
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    kwargs = dict(target_size=M2_IMG_SIZE, batch_size=M2_BATCH_SIZE, class_mode="categorical")

    train_gen = train_datagen.flow_from_directory(os.path.join(M2_DATA_DIR, "train"), **kwargs)
    valid_gen = eval_datagen.flow_from_directory(os.path.join(M2_DATA_DIR, "valid"), **kwargs)
    test_gen  = eval_datagen.flow_from_directory(
        os.path.join(M2_DATA_DIR, "test"), shuffle=False, **kwargs
    )
    return train_gen, valid_gen, test_gen


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = M2_NUM_CLASSES, lr: float = M2_LR) -> Model:
    """
    Build a ResNet-50 classification head (frozen base + custom top).

    Parameters
    ----------
    num_classes : number of WBC subtypes
    lr          : Adam learning rate

    Returns
    -------
    Compiled Keras Model
    """
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers:
        layer.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> None:
    """Full training pipeline: build → fit → evaluate → save."""
    train_gen, valid_gen, test_gen = build_generators()
    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(M2_MODEL_BEST, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    ]

    print(f"\n🚀 Training ResNet-50 for {M2_EPOCHS} epochs…")
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=M2_EPOCHS,
        callbacks=callbacks,
    )

    plot_training_history(history, title_prefix="Module 2 – WBC")

    # ── Evaluation
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\n📊 Test Accuracy : {test_acc * 100:.2f}%")
    print(f"📊 Test Loss     : {test_loss:.4f}")

    # ── Save
    model.save(M2_MODEL_FINAL)
    print(f"✅ Final model saved to: {M2_MODEL_FINAL}")


if __name__ == "__main__":
    train()
