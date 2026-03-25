"""
configs/config.py
-----------------
Centralised configuration for all 3 modules.
Override paths by setting environment variables or editing this file.
"""

import os

# ── Kaggle Dataset Root ───────────────────────────────────────────────────────
DATASET_ROOT = os.getenv(
    "DATASET_ROOT",
    "/kaggle/input/new-disease-analysis/new dataset"
)

WORKING_DIR = os.getenv("WORKING_DIR", "/kaggle/working")

# ── Module 1 : Blood Cell Detection (YOLOv8) ─────────────────────────────────
M1_DATA_DIR      = os.path.join(DATASET_ROOT, "module-1 cell dection")
M1_YAML_PATH     = os.path.join(WORKING_DIR,  "bloodcell.yaml")
M1_MODEL_SAVE    = os.path.join(WORKING_DIR,  "yolov8_module1_best.pt")
M1_YOLO_BASE     = "yolov8s.pt"           # pretrained checkpoint
M1_EPOCHS        = 50
M1_IMG_SIZE      = 640
M1_BATCH         = 8
M1_CONF_THRESH   = 0.25
M1_CLASS_NAMES   = ["RBC", "WBC", "Platelets"]

# Normal clinical ranges per µL (used for medical interpretation)
M1_NORMAL_RANGES = {
    "RBC":      (4.2e6, 5.4e6),
    "WBC":      (4e3,   11e3),
    "Platelet": (150e3, 450e3),
}

# ── Module 2 : WBC Subtype Classification (ResNet-50) ────────────────────────
M2_DATA_DIR      = os.path.join(DATASET_ROOT, "module-2 wbc cell dection")
M2_IMG_SIZE      = (224, 224)
M2_BATCH_SIZE    = 32
M2_NUM_CLASSES   = 5
M2_EPOCHS        = 30
M2_LR            = 1e-4
M2_MODEL_BEST    = os.path.join(WORKING_DIR, "wbc_resnet50_best.h5")
M2_MODEL_FINAL   = os.path.join(WORKING_DIR, "wbc_resnet50_final.h5")
M2_CLASS_NAMES   = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]

# ── Module 3 : Disease Detection (EfficientNetB3) ────────────────────────────
M3_DATA_DIR      = os.path.join(DATASET_ROOT, "module-3 diseases dection")
M3_IMG_SIZE      = 224
M3_BATCH_SIZE    = 32
M3_EPOCHS        = 30
M3_LR            = 1e-4
M3_VAL_SPLIT     = 0.15
M3_FINE_TUNE_LAYERS = 30
M3_MODEL_BEST    = "best_module3_model.h5"
M3_MODEL_FINAL   = os.path.join(WORKING_DIR, "module3_disease_detector_large.h5")
M3_CSV_LOG       = "training_log.csv"
