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

## 📂 File-by-File Description

*   **`config.py`**: central configurations file. Stores directory paths, the 6 gesture category labels, and BGR color maps for the UI.
*   **`checkpoints/keypoint_classifier.pth`**: trained PyTorch neural weights state dict.
*   **`checkpoints/keypoint_classifier_label.csv`**: mapping reference sheet between class indices and gesture strings.
*   **`src/__init__.py`**: package initialization file exposing classes/wrappers for clean external importing.
*   **`src/models.py`**: houses the PyTorch `KeyPointClassifierNet` model class definition and the `KeyPointClassifier` loading wrapper.
*   **`src/transforms.py`**: holds coordinate preprocessing helper functions (`calc_landmark_list`, `pre_process_landmark`, `hand_y_norm`) to normalize landmark values relative to the wrist coordinate and current hand scale.
*   **`src/tracker.py`**: manages coordinate history buffers (`deque` of size 25) for each tracked hand to calculate spatial travel metrics (horizontal waving and vertical raising).
*   **`src/engine.py`**: defines the `GestureEngine` class, which handles EMA landmark smoothing, invokes PyTorch inference, and resolves dynamic HRI scenarios using the history tracker.
*   **`scripts/convert_keras_to_pytorch.py`**: script mapping trained Keras weights over to PyTorch layer weights.
*   **`scripts/train_pytorch.py`**: script to train a fresh classifier in PyTorch from the raw `keypoint.csv` dataset.
*   **`inference/realtime_realsense.py`**: real-time camera inference script supporting both Intel RealSense depth streams and standard webcam fallback indices.

---

## 🚀 Main Script Integration (How to Call)

To call this modality in your main HRI multimodal loop, import `GestureEngine` and call it frame-by-frame:

```python
import cv2 as cv
import mediapipe as mp
from modalities.gesture.src.engine import GestureEngine

# 1. Initialize detector components
mp_hands = mp.solutions.hands.Hands(max_num_hands=2)
gesture_detector = GestureEngine()

# 2. In your video/camera processing loop:
cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Extract raw hand landmarks using MediaPipe
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = mp_hands.process(rgb_frame)

    # 3. Call the GestureEngine to resolve high-level HRI intents
    # Returns: (intent_label_str, color_tuple)
    gesture_intent, color = gesture_detector.process(
        results.multi_hand_landmarks, 
        frame.shape
    )
    
    print(f"Active Gesture: {gesture_intent}")
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
