# 🩸 Blood Cell Disease Analysis System

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00CFFF?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> A three-module deep learning pipeline for automated blood cell analysis — from raw detection to disease classification. Built as a **PG Mini Project** on Kaggle.

---

## 🔬 Project Overview

This system analyses blood microscopy images in three sequential stages:

| Module | Task | Model | Input → Output |
|--------|------|-------|----------------|
| **1** | Blood Cell Detection | YOLOv8s | Image → RBC / WBC / Platelet bounding boxes + clinical counts |
| **2** | WBC Subtype Classification | ResNet-50 | WBC crop → Basophil / Eosinophil / Lymphocyte / Monocyte / Neutrophil |
| **3** | Disease Detection | EfficientNetB3 | Blood smear → Disease class + confidence |

---

## 🏗️ Project Structure

```
blood-cell-analysis/
│
├── run_all.py                            # Master training runner (all 3 modules)
├── requirements.txt
├── .gitignore
├── README.md
│
├── configs/
│   └── config.py                         # All paths, hyperparameters, class names
│
├── utils/
│   └── plot_utils.py                     # Shared: YOLO label viz + training curves
│
├── module1_cell_detection/
│   ├── dataset.py                        # Dataset validation + YAML generation
│   ├── train.py                          # YOLOv8 training script
│   └── inference.py                      # Prediction + clinical metric interpretation
│
├── module2_wbc_classification/
│   ├── train.py                          # ResNet-50 transfer learning + evaluation
│   └── inference.py                      # Single-image WBC subtype prediction
│
└── module3_disease_detection/
    ├── train.py                          # EfficientNetB3 fine-tuning + callbacks
    └── inference.py                      # Disease prediction with class_indices
```

---

## 🔄 System Architecture

```
Blood Microscopy Image
        │
        ▼
┌───────────────────────────────────────┐
│   MODULE 1 — Cell Detection (YOLO)    │
│                                       │
│   YOLOv8s (pretrained ImageNet)       │
│   → Detects RBC, WBC, Platelets       │
│   → Counts converted to cells/µL      │
│   → Clinical interpretation (H/N/L)   │
└──────────────┬────────────────────────┘
               │  WBC crops
               ▼
┌───────────────────────────────────────┐
│   MODULE 2 — WBC Classification       │
│                                       │
│   ResNet-50 (frozen base + custom top)│
│   → 5 WBC subtypes                    │
│   → Basophil / Eosinophil / etc.      │
└──────────────┬────────────────────────┘
               │  full smear
               ▼
┌───────────────────────────────────────┐
│   MODULE 3 — Disease Detection        │
│                                       │
│   EfficientNetB3 (last 30 layers      │
│   unfrozen for fine-tuning)           │
│   → Multi-class disease output        │
│   → Confidence score per class        │
└───────────────────────────────────────┘
```

---

## ⚙️ Setup

### 1. Clone

```bash
git clone https://github.com/Saravananb91/blood-disease-analysis.git
cd blood-cell-analysis
```

### 2. Virtual environment

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset

Download from Kaggle:
```bash
kaggle datasets download -d <dataset-slug> -p ./data --unzip
```
Then update `DATASET_ROOT` in `configs/config.py` to point to your local data path.

---

## ▶️ Running

### Train all modules sequentially
```bash
python run_all.py
```

### Train a specific module
```bash
python run_all.py --module 1   # YOLOv8 detection
python run_all.py --module 2   # ResNet-50 WBC classification
python run_all.py --module 3   # EfficientNetB3 disease detection
```

### Run inference

**Module 1 — detect cells in an image:**
```bash
python -m module1_cell_detection.inference --model runs/best.pt --img sample.jpg
```

**Module 2 — classify WBC subtype:**
```bash
python -m module2_wbc_classification.inference --model wbc_resnet50_final.h5 --img wbc_cell.jpg
```

**Module 3 — predict disease:**
```bash
python -m module3_disease_detection.inference --model module3_disease_detector_large.h5 --img smear.png
```

---

## 🧬 Module Details

### Module 1 — Blood Cell Detection (YOLOv8)

- **Architecture:** YOLOv8s pretrained on COCO, fine-tuned on blood cell data
- **Classes:** `RBC`, `WBC`, `Platelets` (3 classes)
- **Training:** 50 epochs, `imgsz=640`, `batch=8`
- **Output:** Bounding boxes + estimated clinical count per µL

| Cell | Normal Range |
|------|-------------|
| RBC | 4.2M – 5.4M / µL |
| WBC | 4,000 – 11,000 / µL |
| Platelet | 150,000 – 450,000 / µL |

---

### Module 2 — WBC Subtype Classification (ResNet-50)

- **Architecture:** ResNet-50 (frozen base) + GlobalAvgPool + Dropout + Dense
- **Classes:** Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil (5 classes)
- **Training:** 30 epochs, `lr=1e-4`, EarlyStopping (patience=10)
- **Augmentation:** Rotation, shifts, zoom, horizontal flip

---

### Module 3 — Disease Detection (EfficientNetB3)

- **Architecture:** EfficientNetB3 with last 30 layers unfrozen for fine-tuning
- **Training:** 30 epochs, `lr=1e-4`, ReduceLROnPlateau, EarlyStopping (patience=6)
- **Augmentation:** Heavy augmentation including vertical flip, brightness variation
- **Callbacks:** ModelCheckpoint, CSVLogger (training_log.csv)

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| ultralytics | ≥8.1 | YOLOv8 detection |
| tensorflow | 2.15.0 | ResNet-50 / EfficientNetB3 |
| keras | 2.15.0 | Model building |
| opencv-python | ≥4.9 | Image processing |
| numpy | ≥1.24 | Numerical ops |
| matplotlib | ≥3.8 | Visualisation |

---

## 🔮 Roadmap

- [ ] Streamlit web app for interactive inference
- [ ] Docker container for reproducible environment
- [ ] ONNX export for deployment
- [ ] Grad-CAM visualisations for model explainability
- [ ] Module 1 + 2 pipeline: detect WBC → auto-classify subtype

---

## 📄 Kaggle Notebook

```bash
kaggle kernels pull saransaravanan/allinone
```

---

## 👤 Author

Saravanan B - [mrsaravananb@gmail.com ](mrsaravananb@gmail.com)

Project Link: [https://github.com/Saravananb91/road-pothole-](https://github.com/Saravananb91/road-pothole-)

Portfolio website : [portfolio-saravananb.vercel.app](https://v0-portfolio-saravanan-b.vercel.app/) 

Linkedin: [www.linkedin.com/in/saravanan-b-46244b290](www.linkedin.com/in/saravanan-b-46244b290)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
