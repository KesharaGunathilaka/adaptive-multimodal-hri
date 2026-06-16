from pathlib import Path
from ultralytics import YOLO


def train_model():
    # 1. Define paths
    workspace_dir = Path(__file__).resolve().parent

    # Point this to your dataset's YAML file (e.g., downloaded from Roboflow)
    dataset_yaml = workspace_dir / "data" / "data.yaml"

    if not dataset_yaml.exists():
        print(f"Error: Could not find dataset configuration at {dataset_yaml}")
        return

    # 2. Initialize the YOLO11-Nano model
    print("Initializing YOLO11-Nano...")
    model = YOLO("yolo11n.pt")

    # 3. Start Training
    print("Starting training process...")
    model.train(
        data=str(dataset_yaml),
        epochs=100,  # Max epochs (Early stopping will halt it sooner if needed)
        imgsz=640,  # Standard image size for YOLO
        batch=32,  # Efficient batch size for most GPUs
        device=0,  # Target the primary GPU
        project="runs/detect",  # Directory to save outputs
        name="hri_yolo11n",  # Name of this specific training run
        patience=15,  # Halt if no improvement for 15 epochs
        workers=8,  # Dataloader workers for faster image loading
    )


if __name__ == "__main__":
    train_model()
