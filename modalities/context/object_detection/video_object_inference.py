import cv2
import sys
import io
from pathlib import Path
from ultralytics import YOLO

# Set UTF-8 encoding for stdout
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Ensure repo-root imports work
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Video test folder
VIDEOS_TEST_DIR = Path(__file__).resolve().parents[3] / "videos" / "test"

# Video extensions to process
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

# LOAD MODEL
weights_path = Path(__file__).resolve().parent / "yolo11n.pt"

if not weights_path.exists():
    print(f"✗ Model weights not found at: {weights_path}")
    sys.exit(1)

model = YOLO(str(weights_path))
print(f"✓ Model loaded\n")

# FIND ALL VIDEOS
print(f"Searching for videos in: {VIDEOS_TEST_DIR}\n")

if not VIDEOS_TEST_DIR.exists():
    print(f"✗ Video directory not found: {VIDEOS_TEST_DIR}")
    sys.exit(1)

# Find all video files
video_files = []
for ext in VIDEO_EXTENSIONS:
    video_files.extend(VIDEOS_TEST_DIR.rglob(f"*{ext}"))
    video_files.extend(VIDEOS_TEST_DIR.rglob(f"*{ext.upper()}"))

video_files = sorted(list(set(video_files)))

if not video_files:
    print(f"✗ No video files found in: {VIDEOS_TEST_DIR}")
    sys.exit(1)

print(f"✓ Found {len(video_files)} video(s):\n")
for i, vid in enumerate(video_files, 1):
    print(f"  {i}. {vid.name}")

print(f"\n{'-' * 70}")
print("Instructions: Press 'n' for next video, 'q' to quit")
print(f"{'-' * 70}\n")


# PROCESS VIDEOS WITH PREDICTIONS
for video_idx, video_path in enumerate(video_files, 1):
    print(f"\n[{video_idx}/{len(video_files)}] {video_path.name}")
    print(f"{'=' * 70}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"✗ Error: Could not open video")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✓ {width}x{height} @ {fps} fps, {total_frames} frames")
    print("Press 'n' to skip, 'q' to quit\n")

    frame_count = 0
    skip_video = False

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Run inference
        results = model.predict(source=frame, conf=0.5, verbose=False)

        # Annotate frame
        annotated_frame = results[0].plot()

        # Show frame
        cv2.imshow("Object Detection - Live", annotated_frame)

        # Handle key press
        key = cv2.waitKey(int(1000 / fps)) & 0xFF

        if key == ord("n"):
            print(f"Skipped after {frame_count} frames")
            skip_video = True
            break
        elif key == ord("q") or key == 27:
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()

    if not skip_video:
        print(f"✓ Finished ({frame_count} frames)\n")

print(f"\n{'=' * 70}")
print("✓ All videos processed!")
print(f"{'=' * 70}")
