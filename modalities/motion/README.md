# 🏃‍♂️ High-Accuracy Motion Modality

This folder contains the real-time human body posture and motion tracking pipeline for the Adaptive Multimodal HRI framework, implemented in **PyTorch**.

---

## 🛠️ How It Works & Techniques
*   **MediaPipe Pose**: Tracks 33 body skeletal keypoints in real-time.
*   **Postural Checking**: Rule-based classifier checks coordinates relative to each other (Standing, Sitting, Crouching, Lying).
*   **PyTorch LSTM**: Evaluates sequences across a 30-frame rolling window to track velocity and motion directions.
*   **EMA Probability Smoothing**: An Exponential Moving Average filter ($\alpha = 0.25$) on the output probability vector stabilizes transitions between active motion states.

---

## 📂 Directory Layout
```
motion/
├── config.py                 # Central configurations, paths, labels, and colors
├── checkpoints/              # Model weights and configuration mappings
│   ├── motion_lstm_v2_best.pth
│   └── model_config_v2.json
├── src/                      # Modality source components
│   ├── __init__.py           # Package interfaces
│   ├── models.py             # PyTorch LSTM model class
│   └── engine.py             # MotionEngine coordinate tracking & resolver
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
