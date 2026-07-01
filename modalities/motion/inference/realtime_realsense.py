"""
Real-time Human Motion Recognizer using PyTorch LSTM & MediaPipe Pose.
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
from config import MOTION_COLORS, DEFAULT_CHECKPOINT
from src import MotionEngine

def draw_dashboard(frame, pose_label, motion_label, confidence, fps, landmarks, mp_drawing, mp_pose):
    h, w = frame.shape[:2]
    is_portrait = h > w
    font = cv.FONT_HERSHEY_SIMPLEX

    # Draw Pose Skeleton
    if landmarks:
        mp_drawing.draw_landmarks(
            frame, landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2),
        )

    color = MOTION_COLORS.get(motion_label, (255, 255, 255))

    if is_portrait:
        panel_h = 160
        canvas = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
        canvas[:h, :w] = frame
        cv.rectangle(canvas, (0, h), (w, h + panel_h), (20, 24, 33), -1)
        cv.line(canvas, (0, h), (w, h), (43, 52, 69), 2)

        # Title Row
        cv.putText(canvas, "HRI MOTION ANALYZER", (15, h + 30), font, 0.6, (230, 235, 245), 2, cv.LINE_AA)
        cv.putText(canvas, f"FPS:{fps:.1f}  |  Pose:{pose_label}", (w - 240, h + 30), font, 0.45, (150, 160, 180), 1, cv.LINE_AA)
        cv.line(canvas, (10, h + 42), (w - 10, h + 42), (60, 75, 100), 1)

        # Active Motion Badge
        cv.rectangle(canvas, (10, h + 60), (w - 10, h + 130), color, -1)
        text_color = (255, 255, 255) if np.mean(color) < 120 else (20, 20, 20)
        cv.putText(canvas, f"{motion_label} ({confidence*100:.1f}%)", (25, h + 105), font, 1.0, text_color, 3, cv.LINE_AA)
    else:
        sidebar_w = 320
        canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        cv.rectangle(canvas, (w, 0), (w + sidebar_w, h), (20, 24, 33), -1)
        cv.line(canvas, (w, 0), (w, h), (43, 52, 69), 2)

        cv.putText(canvas, "HRI MOTION ANALYZER", (w + 20, 35), font, 0.65, (230, 235, 245), 2, cv.LINE_AA)
        cv.line(canvas, (w + 20, 48), (w + sidebar_w - 20, 48), (60, 75, 100), 1)

        cv.putText(canvas, f"FPS: {fps:.1f}", (w + 20, 75), font, 0.5, (150, 160, 180), 1, cv.LINE_AA)
        cv.putText(canvas, f"Pose: {pose_label}", (w + 140, 75), font, 0.5, (150, 160, 180), 1, cv.LINE_AA)

        cv.putText(canvas, "ACTIVE MOTION:", (w + 20, 120), font, 0.55, (100, 120, 150), 1, cv.LINE_AA)
        cv.rectangle(canvas, (w + 20, 140), (w + sidebar_w - 20, 210), color, -1)
        text_color = (255, 255, 255) if np.mean(color) < 120 else (20, 20, 20)
        cv.putText(canvas, motion_label, (w + 30, 175), font, 0.7, text_color, 2, cv.LINE_AA)
        cv.putText(canvas, f"Conf: {confidence*100:.1f}%", (w + 30, 200), font, 0.55, text_color, 1, cv.LINE_AA)

    return canvas

def resize_with_aspect_ratio(image, max_dim=960):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)

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
    parser.add_argument("--camera", type=int, default=0, help="Webcam fallback index.")
    args = parser.parse_args()

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    engine = MotionEngine()
    pipeline, align = _try_start_realsense()

    if pipeline is not None:
        print("RealSense camera connected.")
        window_title, cap = "Motion Recognition (RealSense)", None
    else:
        print(f"RealSense fallback webcam (index {args.camera}).")
        cap = cv.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Error: camera index invalid.")
            return
        window_title = "Motion Recognition (Webcam)"

    last_time = time.time()
    fps = 0.0

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

            frame = resize_with_aspect_ratio(frame, max_dim=960)
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            res = pose.process(rgb)

            landmarks = res.pose_landmarks
            motion_state, confidence, pose_class = engine.process(landmarks)

            # Draw adaptive HRI analyzer dashboard
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = int(0.9 * fps + 0.1 * (1.0 / dt))

            canvas = draw_dashboard(
                frame, pose_class, motion_state,
                confidence, fps, landmarks,
                mp_draw, mp_pose
            )

            cv.imshow(window_title, canvas)
            key = cv.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        if pipeline is not None: pipeline.stop()
        else: cap.release()
        pose.close()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
