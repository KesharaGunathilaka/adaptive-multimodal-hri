# 🤖 High-Accuracy Gesture Modality

This folder contains the real-time hand gesture recognition pipeline for the Adaptive Multimodal HRI framework, implemented natively in **PyTorch** for consistency with the rest of the project's modalities.

---

## 🛠️ How It Works & Techniques
*   **MediaPipe Hands**: Extracts 21 skeletal joints (x, y, z landmarks) per hand in real-time.
*   **PyTorch Classifier (MLP)**: Performs classification using a PyTorch multi-layer perceptron (MLP) loaded from a `.pth` file to identify 6 poses: Open Palm, Close (Fist), Pointer, Thumbs Up, Thumbs Down, and Beckoning.
*   **EMA Smoothing**: An Exponential Moving Average filter ($\alpha = 0.45$) stabilizes coordinate streams and eliminates landmark jitter ("dancing points").
*   **Dynamic Trajectory Resolving**: Tracks motion histories over a 25-frame rolling window to detect actions like waving, beckoning (scale-invariantly), and hand raising (flex-pose tracking, supporting fists/unknown shapes).

---

## 📂 Directory Layout
```
gesture/
├── config.py                 # Central configurations, paths, labels, and colors
├── checkpoints/              # Folder housing model weights and label mappings
│   ├── keypoint_classifier.pth
│   └── keypoint_classifier_label.csv
├── src/                      # Modality source components
│   ├── __init__.py           # Package interfaces
│   ├── models.py             # PyTorch model class and wrapper
│   ├── tracker.py            # Temporal coordinate history tracking
│   ├── engine.py             # GestureEngine state-machine resolver
│   └── transforms.py         # Landmark scaling and coordinate preprocessing
├── scripts/                  # Training and conversion utilities
│   ├── convert_keras_to_pytorch.py # Converts original Keras .hdf5 to PyTorch .pth
│   └── train_pytorch.py      # Trains the MLP directly in PyTorch from keypoint.csv
└── inference/                # Execution scripts
    └── realtime_realsense.py # Real-time camera stream detector (RealSense/Webcam)
```

---

## 🚀 Setup & Execution

### 1. Setup Environment
Ensure your active environment has the required packages installed:
```bash
pip install torch opencv-python mediapipe numpy
```

### 2. Run Real-time Inference
To start real-time detection from your connected Intel RealSense camera or default webcam fallback:
```bash
python inference/realtime_realsense.py
```

To run with a custom PyTorch checkpoint file:
```bash
python inference/realtime_realsense.py --checkpoint checkpoints/keypoint_classifier.pth
```
