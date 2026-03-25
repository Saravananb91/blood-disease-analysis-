"""
run_all.py
----------
Master script that trains all three modules sequentially.
Run each module independently or all at once.

Usage
-----
    # Train all modules
    python run_all.py

    # Train specific module only
    python run_all.py --module 1
    python run_all.py --module 2
    python run_all.py --module 3
"""

import argparse


def run_module1():
    print("\n" + "="*60)
    print("  MODULE 1 — Blood Cell Detection (YOLOv8)")
    print("="*60)
    from module1_cell_detection.train import train, save_model
    model = train()
    save_model(model)


def run_module2():
    print("\n" + "="*60)
    print("  MODULE 2 — WBC Subtype Classification (ResNet-50)")
    print("="*60)
    from module2_wbc_classification.train import train
    train()


def run_module3():
    print("\n" + "="*60)
    print("  MODULE 3 — Disease Detection (EfficientNetB3)")
    print("="*60)
    from module3_disease_detection.train import train
    train()


MODULE_MAP = {1: run_module1, 2: run_module2, 3: run_module3}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blood Cell Analysis — run training pipeline")
    parser.add_argument(
        "--module",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a specific module (1, 2, or 3). Omit to run all.",
    )
    args = parser.parse_args()

    if args.module:
        MODULE_MAP[args.module]()
    else:
        for fn in MODULE_MAP.values():
            fn()

    print("\n✅ All requested modules completed.")
