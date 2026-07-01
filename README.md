# 🤖 Adaptive Multimodal HRI Framework

An adaptive human–robot interaction (HRI) framework integrating multimodal emotion, gesture, motion, and contextual cues for real-time policy learning and dynamic robot behavior.

This repository organizes different perception streams into self-contained **modalities** under the `modalities/` directory.

---

## 📂 Project Structure

*   **`modalities/emotion/`**: Real-time facial emotion recognition using MobileNetV2 in PyTorch.
*   **`modalities/gesture/`**: Real-time hand gesture recognition (MLP classifier + GestureEngine resolver) in PyTorch.
*   **`modalities/motion/`**: Tracks body skeletal trajectories and classifies dynamic action states.
*   **`modalities/context/`**: Processes environment and contextual features.

---

## 🚀 Setup & Execution (Gesture Modality)

### 1. Setup Environment
Ensure your active environment has the required packages installed:
```bash
pip install torch opencv-python mediapipe numpy
```

### 2. Run Real-time Inference
To start real-time detection from your connected Intel RealSense camera or default webcam fallback:
```bash
python modalities/gesture/inference/realtime_realsense.py
```

To run with a custom PyTorch checkpoint file:
```bash
python modalities/gesture/inference/realtime_realsense.py --checkpoint modalities/gesture/checkpoints/keypoint_classifier.pth
```
