"""
video.py

Test the motion model on a local .mp4 video file.
Drop-in replacement for realtime.py — same visualisation,
same inference engine, no code changes to core pipeline.

Usage:
    python inference/video.py --video path/to/video.mp4

Optional flags:
    --slow        Play at half speed (easier to read predictions)
    --save        Save annotated output to outputs/<name>-annotated.mp4

Controls:
    Q / ESC       Quit
    SPACE         Pause / resume
    R             Reset inference buffer
    S             Save screenshot of current frame
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import time
import argparse
from collections import deque

# Add src/ to path so we can import our modules
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
ANNOTATED_DIR = os.path.join(PROJECT_ROOT, "outputs")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from inference import MotionInference, MOTION_LABELS, NUM_CLASSES

# ─── Colours (BGR) ────────────────────────────────────────────────────────────
COLOURS = {
    "sitting":       (  0, 200, 255),
    "standing":      (  0, 255,   0),
    "walking":       (255, 200,   0),
    "stepping_back": (  0,   0, 255),
    "buffering":     (180, 180, 180),
}

SKELETON_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

# ─── MediaPipe → NTU joint mapping ───────────────────────────────────────────
MP_TO_NTU = {
    0:  3,   # nose         → head
    11: 4,   # l_shoulder   → left_shoulder
    12: 8,   # r_shoulder   → right_shoulder
    13: 5,   # l_elbow      → left_elbow
    14: 9,   # r_elbow      → right_elbow
    15: 6,   # l_wrist      → left_wrist
    16: 10,  # r_wrist      → right_wrist
    23: 12,  # l_hip        → left_hip
    24: 16,  # r_hip        → right_hip
    25: 13,  # l_knee       → left_knee
    26: 17,  # r_knee       → right_knee
    27: 14,  # l_ankle      → left_ankle
    28: 18,  # r_ankle      → right_ankle
}


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def draw_skeleton(frame, landmarks, world_lms, colour, h, w):
    for a, b in SKELETON_CONNECTIONS:
        lm_a = landmarks[a]
        lm_b = landmarks[b]
        if lm_a.visibility > 0.4 and lm_b.visibility > 0.4:
            pt_a = (int(lm_a.x * w), int(lm_a.y * h))
            pt_b = (int(lm_b.x * w), int(lm_b.y * h))
            cv2.line(frame, pt_a, pt_b, colour, 2, cv2.LINE_AA)
    for idx in range(33):
        lm = landmarks[idx]
        if lm.visibility > 0.4:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pt, 4, colour, -1, cv2.LINE_AA)


def draw_prediction(frame, result, w):
    label  = result.label
    conf   = result.confidence
    colour = COLOURS.get(label, (200, 200, 200))
    text   = f"{label.upper()}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    margin = 10
    cv2.rectangle(frame,
                  (w // 2 - tw // 2 - margin, 8),
                  (w // 2 + tw // 2 + margin, 50),
                  (30, 30, 30), -1)
    cv2.putText(frame, text,
                (w // 2 - tw // 2, 42),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, colour, 2, cv2.LINE_AA)
    if not result.stable:
        cv2.putText(frame, "Buffering  (1 sec warmup)",
                    (w // 2 - 110, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


def draw_prob_bars(frame, probs, h, x=10, y_offset=145):
    bar_w   = 120
    bar_h   = 14
    padding = 4
    y_start = h - y_offset
    for i, (idx, name) in enumerate(MOTION_LABELS.items()):
        p      = float(probs[idx]) if len(probs) == NUM_CLASSES else 0.0
        colour = COLOURS.get(name, (200, 200, 200))
        y0     = y_start + i * (bar_h + padding)
        cv2.rectangle(frame, (x, y0), (x + bar_w, y0 + bar_h),
                      (50, 50, 50), -1)
        filled = int(p * bar_w)
        cv2.rectangle(frame, (x, y0), (x + filled, y0 + bar_h),
                      colour, -1)
        cv2.putText(frame,
                    f"{name[:13]:<13} {p:.2f}",
                    (x + bar_w + 6, y0 + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                    (220, 220, 220), 1, cv2.LINE_AA)


def draw_progress_bar(frame, frame_idx, total_frames, h, w):
    """Video progress bar along the bottom edge."""
    bar_h  = 6
    filled = int((frame_idx / max(total_frames, 1)) * w)
    cv2.rectangle(frame, (0, h - bar_h), (w, h),      (40,  40,  40), -1)
    cv2.rectangle(frame, (0, h - bar_h), (filled, h), (0,  200, 100), -1)
    # Timecode
    fps_vid = 30  # approximate — replaced by actual cap fps below
    secs    = frame_idx / max(fps_vid, 1)
    total_s = total_frames / max(fps_vid, 1)
    cv2.putText(frame,
                f"{int(secs//60):02d}:{int(secs%60):02d} / "
                f"{int(total_s//60):02d}:{int(total_s%60):02d}",
                (w - 120, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def draw_info(frame, fps, frame_idx, total_frames, paused):
    status = "PAUSED" if paused else f"FPS: {fps:.1f}"
    colour = (0, 100, 255) if paused else (200, 200, 200)
    cv2.putText(frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test motion model on a video file.")
    parser.add_argument("--video",  required=True, help="Path to .mp4 video file")
    parser.add_argument("--slow",   action="store_true",
                        help="Half-speed playback")
    parser.add_argument("--save",   action="store_true",
                        help="Save annotated video as output_annotated.mp4")
    parser.add_argument("--headless", action="store_true",
                        help="Process without opening a video window")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "best_model_finetuned.pt")
    print("Loading model...")
    engine = MotionInference(ckpt)

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    # Try new Tasks API first, fall back to solutions API
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
        )
        USE_SOLUTIONS_API = True
        print("MediaPipe: using solutions API")
    except AttributeError:
        # mediapipe >= 0.10.15 Tasks API
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request

        model_path = os.path.join(THIS_DIR, "pose_landmarker_full.task")
        if not os.path.exists(model_path):
            print("Downloading MediaPipe pose model (~30 MB)...")
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_full/float16/latest/"
                   "pose_landmarker_full.task")
            urllib.request.urlretrieve(url, model_path)
            print("Download complete.")

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose = mp_vision.PoseLandmarker.create_from_options(opts)
        USE_SOLUTIONS_API = False
        print("MediaPipe: using Tasks API")

    # ── Video capture ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {args.video}")
    print(f"  Resolution : {vid_w}×{vid_h}")
    print(f"  FPS        : {vid_fps:.1f}")
    print(f"  Frames     : {total_frames}  (~{total_frames/vid_fps:.1f}s)")
    print(f"\nControls: SPACE=pause  R=reset  S=screenshot  Q/ESC=quit\n")

    # ── Output writer (optional) ──────────────────────────────────────────────
    writer = None
    if args.save:
        os.makedirs(ANNOTATED_DIR, exist_ok=True)
        input_name = os.path.splitext(os.path.basename(args.video))[0]
        out_path = os.path.join(ANNOTATED_DIR, f"{input_name}-annotated.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, vid_fps, (vid_w, vid_h))
        print(f"Saving annotated video to: {out_path}")

    # ── Playback timing ───────────────────────────────────────────────────────
    # Wait time between frames (ms). Slow mode = 2× slower.
    frame_delay = int(1000 / vid_fps) * (2 if args.slow else 1)
    frame_delay = max(frame_delay, 1)

    # ── Main loop ─────────────────────────────────────────────────────────────
    fps_buffer = deque(maxlen=30)
    prev_time  = time.time()
    frame_idx  = 0
    paused     = False
    screenshot_ctr = 0

    while True:
        # Handle pause — keep window responsive without reading new frames
        if paused:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                paused = False
            elif key == ord('r'):
                engine.reset()
                print("Buffer reset.")
            continue

        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video.")
            break

        frame_idx += 1
        h, w = frame.shape[:2]

        # ── Pose estimation ───────────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        person_detected = False
        joints_25 = np.zeros((25, 3), dtype=np.float32)

        if USE_SOLUTIONS_API:
            results = pose.process(rgb)
            if results.pose_landmarks:
                person_detected = True
                landmarks       = results.pose_landmarks.landmark
                world_lms       = results.pose_world_landmarks.landmark

                # MediaPipe world landmarks use X-right/Y-down; Kinect/NTU
                # (the training convention) uses X-left/Y-up. Negating X and
                # Y is a 180 deg rotation about Z, so it corrects both axes
                # without mirroring the skeleton.
                for mp_idx, ntu_idx in MP_TO_NTU.items():
                    lm = world_lms[mp_idx]
                    joints_25[ntu_idx] = [-lm.x, -lm.y, lm.z]

                # Approximate spine joints
                joints_25[0] = (joints_25[12] + joints_25[16]) / 2
                joints_25[1] = (joints_25[4]  + joints_25[8])  / 2
                joints_25[2] = joints_25[1] * 0.5 + joints_25[3] * 0.5
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            det      = pose.detect(mp_image)
            if det.pose_landmarks:
                person_detected = True
                landmarks       = det.pose_landmarks[0]
                world_lms       = det.pose_world_landmarks[0]

                # MediaPipe world landmarks use X-right/Y-down; Kinect/NTU
                # (the training convention) uses X-left/Y-up. Negating X and
                # Y is a 180 deg rotation about Z, so it corrects both axes
                # without mirroring the skeleton.
                for mp_idx, ntu_idx in MP_TO_NTU.items():
                    lm = world_lms[mp_idx]
                    joints_25[ntu_idx] = [-lm.x, -lm.y, lm.z]

                joints_25[0] = (joints_25[12] + joints_25[16]) / 2
                joints_25[1] = (joints_25[4]  + joints_25[8])  / 2
                joints_25[2] = joints_25[1] * 0.5 + joints_25[3] * 0.5

        # ── Inference ─────────────────────────────────────────────────────────
        result = engine.update(joints_25)

        # ── Draw ──────────────────────────────────────────────────────────────
        if person_detected:
            colour = COLOURS.get(result.label, (200, 200, 200))
            if USE_SOLUTIONS_API:
                draw_skeleton(frame, landmarks, world_lms, colour, h, w)
            else:
                # Tasks API: convert landmark list to drawable format
                class _LM:
                    def __init__(self, x, y, vis):
                        self.x = x; self.y = y; self.visibility = vis
                fake_lms = [_LM(lm.x, lm.y, lm.presence)
                            for lm in landmarks]
                draw_skeleton(frame, fake_lms, world_lms, colour, h, w)

        # Dark overlay for prob bars
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 155), (275, h - 8),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        draw_prediction(frame, result, w)
        draw_prob_bars(frame, result.probs, h)
        draw_progress_bar(frame, frame_idx, total_frames, h, w)

        # FPS
        now = time.time()
        fps_buffer.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        draw_info(frame, np.mean(fps_buffer), frame_idx, total_frames, paused)

        if not person_detected:
            cv2.putText(frame, "No person detected",
                        (w // 2 - 100, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (100, 100, 255), 1)

        # ── Show / save ───────────────────────────────────────────────────────
        if not args.headless:
            cv2.imshow("HRI Motion Detector — Video Test", frame)
        if writer:
            writer.write(frame)

        # ── Keys ──────────────────────────────────────────────────────────────
        if args.headless:
            continue

        key = cv2.waitKey(frame_delay) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = True
            print(f"Paused at frame {frame_idx}/{total_frames}")
        elif key == ord('r'):
            engine.reset()
            print("Buffer reset.")
        elif key == ord('s'):
            fname = f"screenshot_{screenshot_ctr:03d}.jpg"
            cv2.imwrite(fname, frame)
            screenshot_ctr += 1
            print(f"Screenshot saved: {fname}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        print(f"Annotated video saved: {out_path}")
    if not args.headless:
        cv2.destroyAllWindows()
    if USE_SOLUTIONS_API:
        pose.close()
    print("Done.")


if __name__ == "__main__":
    main()