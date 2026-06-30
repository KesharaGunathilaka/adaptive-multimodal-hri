"""
Fine-tune YOLO11-Nano on a custom HRI object dataset.

Deployment currently uses the pretrained COCO weights (its classes already cover
the HRI objects we need). Use this when extending detection to non-COCO objects:
drop a Roboflow/YOLO dataset at object_detection/data/data.yaml and run it.

Run from the object_detection/ folder:
    python scripts/train.py
"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]          # object_detection/
DATASET_YAML = ROOT / "data" / "data.yaml"
BASE_WEIGHTS = ROOT / "checkpoints" / "yolo11n.pt"


def train_model():
    if not DATASET_YAML.exists():
        print(f"Error: dataset config not found at {DATASET_YAML}")
        print("Place a YOLO dataset (data.yaml + images/labels) there first.")
        return

    print("Initializing YOLO11-Nano...")
    model = YOLO(str(BASE_WEIGHTS) if BASE_WEIGHTS.exists() else "yolo11n.pt")

    print("Starting training...")
    model.train(
        data=str(DATASET_YAML),
        epochs=100,
        imgsz=640,
        batch=32,
        device=0,
        project=str(ROOT / "runs" / "detect"),
        name="hri_yolo11n",
        patience=15,
        workers=8,
    )


if __name__ == "__main__":
    train_model()
