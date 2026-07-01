"""
Standalone real-time hand gesture recognition from a camera.
"""
import argparse
import os
import sys
import time
import cv2 as cv
import mediapipe as mp
import numpy as np

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from config import GESTURE_COLORS, DEFAULT_CHECKPOINT
from src import GestureEngine

def draw_overlay(frame, label, color):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv.rectangle(overlay, (0, 0), (w, 75), (15, 15, 15), -1)
    cv.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv.putText(frame, "HRI REAL-TIME GESTURE DETECTOR  |  PYTORCH SMOOTHED",
                (12, 22), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv.LINE_AA)
    cv.putText(frame, label,
                (12, 62), cv.FONT_HERSHEY_DUPLEX, 1.4, color, 2, cv.LINE_AA)
    cv.rectangle(frame, (0, 0), (6, h), color, -1)

def _try_start_realsense():
    if not _RS_AVAILABLE:
        return None, None
    try:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(cfg)
        return pipeline, rs.align(rs.stream.color)
    except Exception:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to PyTorch checkpoint.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam fallback index.")
    args = parser.parse_args()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        model_complexity=1,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    engine = GestureEngine(model_path=args.checkpoint)
    pipeline, align = _try_start_realsense()
    
    if pipeline is not None:
        print("RealSense camera connected.")
        window_title, cap = "Gesture Recognition (RealSense)", None
    else:
        print(f"RealSense fallback webcam (index {args.camera}).")
        cap = cv.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Error: camera index invalid.")
            return
        window_title = "Gesture Recognition (Webcam)"

    last_time = time.time()
    fps = 0

    try:
        while True:
            if pipeline is not None:
                frames = align.process(pipeline.wait_for_frames())
                color_frame = frames.get_color_frame()
                if not color_frame: continue
                frame = np.asarray(color_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret: break

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                gesture, color = engine.process(res.multi_hand_landmarks, frame.shape)
                for hl in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 150), thickness=2, circle_radius=4),
                        mp_draw.DrawingSpec(color=(0, 200, 100), thickness=2),
                    )
            else:
                gesture, color = engine.process(None, frame.shape)

            draw_overlay(frame, gesture, color)
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = int(0.9 * fps + 0.1 * (1.0 / dt))

            cv.putText(frame, f"FPS: {fps}", (frame.shape[1] - 100, 50),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv.LINE_AA)
            cv.imshow(window_title, frame)

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        if pipeline is not None: pipeline.stop()
        else: cap.release()
        hands.close()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
