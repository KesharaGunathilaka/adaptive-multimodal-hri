# 🏃‍♂️ High-Accuracy Motion Modality

This folder contains the real-time human body posture and motion tracking pipeline for the Adaptive Multimodal HRI framework, implemented in **PyTorch**.

---

## 🛠️ How It Works & Techniques
*   **MediaPipe Pose**: Tracks 33 body skeletal keypoints in real-time.
*   **Postural Checking**: Rule-based classifier checks coordinates relative to each other (Standing, Sitting, Crouching, Lying).
*   **PyTorch LSTM**: Evaluates sequences across a 30-frame rolling window to track velocity and motion directions.
*   **EMA Probability Smoothing**: An Exponential Moving Average filter ($\alpha = 0.25$) on the output probability vector stabilizes transitions between active motion states.

---

## 📂 File-by-File Description

*   **`config.py`**: central configurations file. Stores directory paths, active motion category labels, and display color definitions.
*   **`checkpoints/motion_lstm_v2_best.pth`**: PyTorch LSTM model weights checkpoint.
*   **`checkpoints/model_config_v2.json`**: JSON file detailing configuration variables like input size, layer counts, and dropout settings.
*   **`src/__init__.py`**: package initialization file exposing classes/wrappers for clean external importing.
*   **`src/models.py`**: houses the PyTorch `MotionLSTM` class definition.
*   **`src/engine.py`**: implements the `PoseClassifier` class (posture checks) and the `MotionEngine` tracker class which performs rolling velocity checking and EMA smoothing.
*   **`inference/realtime_realsense.py`**: real-time video/camera streaming script displaying an overlay dashboard of human skeletal pose and active motion states.

---

## 🚀 Main Script Integration (How to Call)

To call this modality in your main HRI multimodal loop, import `MotionEngine` and call it frame-by-frame:

```python
import cv2 as cv
import mediapipe as mp
from modalities.motion.src.engine import MotionEngine

# 1. Initialize detector components
mp_pose = mp.solutions.pose.Pose()
motion_detector = MotionEngine()

# 2. In your video/camera processing loop:
cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Extract body keypoints using MediaPipe Pose
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = mp_pose.process(rgb_frame)

    # 3. Call the MotionEngine to resolve the active action state
    # Returns: (motion_state_label, confidence_score, posture_class_label)
    motion_state, confidence, posture = motion_detector.process(
        results.pose_landmarks
    )
    
    print(f"Motion: {motion_state} | Posture: {posture}")
```

---

## 🚀 Setup & Execution

### 1. Setup Environment
Ensure your active environment has the required packages installed:
```bash
pip install torch opencv-python mediapipe numpy
```

### 2. Run Real-time Inference
To start real-time detection:
```bash
python inference/realtime_realsense.py
```
