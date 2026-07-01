"""
Gaze estimation on a video file (or a folder of videos), for the HRI context
model. Defaults to scanning the repo's videos/ folder.

Run from this folder (needs the project's gaze_estimator.py):
    python video.py                                    # batch the repo videos/ folder
    python video.py --video myclip.mp4
    python video.py --videos-dir ./my_videos            # batch a different folder

Method: frame -> MediaPipe full-range face detection + crop -> Face Mesh head
pose + iris -> gaze ray, engagement, and whether the user is looking at the robot.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.gaze_estimation.gaze_estimator import GazeEstimator

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# repo_root/modalities/context/gaze_estimation/inference -> repo_root
DEFAULT_VIDEOS_DIR = str(Path(SCRIPT_DIR).parents[3] / "videos")


def collect_videos(videos_dir):
    found = []
    for dirpath, _, filenames in os.walk(videos_dir):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in VIDEO_EXTENSIONS:
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


def build_output_path(video_path, videos_dir, out_root):
    rel = os.path.relpath(video_path, videos_dir)
    out_path = os.path.join(out_root, os.path.splitext(rel)[0] + "_gaze.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


_HINT = "[SPACE] Pause   [N] Next   [P] Prev   [Q] Quit"


def _draw_overlay(frame, paused):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, _HINT, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (220, 220, 220), 1, cv2.LINE_AA)
    if paused:
        text, font, scale, thick = "PAUSED", cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cx, cy = (w - tw) // 2, (h + th) // 2
        cv2.putText(frame, text, (cx + 2, cy + 2), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (cx, cy), font, scale, (0, 220, 255), thick, cv2.LINE_AA)


def _draw_gaze(frame, gaze):
    if not gaze.has_face:
        cv2.putText(frame, "No face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return
    if gaze.gaze_point and gaze.face_bbox:
        cx = (gaze.face_bbox[0] + gaze.face_bbox[2]) // 2
        cy = (gaze.face_bbox[1] + gaze.face_bbox[3]) // 2
        cv2.arrowedLine(frame, (cx, cy), (int(gaze.gaze_point[0]), int(gaze.gaze_point[1])),
                        (0, 255, 0), 2, tipLength=0.2)
    status = "ROBOT" if gaze.looking_at_robot else "AWAY"
    color = (0, 255, 0) if gaze.looking_at_robot else (0, 165, 255)
    cv2.putText(frame, f"yaw {gaze.yaw:+.0f} pitch {gaze.pitch:+.0f} -> {status} "
                f"eng {gaze.engagement:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def process_video(video_path, out_path, estimator, show):
    """Process one video. Returns: "done" | "next" | "prev" | "quit"."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] Cannot open: {video_path}")
        return "next"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    action, paused, display_frame = "done", False, None
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            _draw_gaze(frame, estimator.estimate(frame))
            writer.write(frame)
            display_frame = frame.copy()

        if show and display_frame is not None:
            view = display_frame.copy()
            _draw_overlay(view, paused)
            cv2.imshow("Gaze Estimation", view)
            key = cv2.waitKey(100 if paused else 30) & 0xFF
            if key == ord("q"):
                action = "quit"; break
            elif key == ord("n"):
                action = "next"; break
            elif key == ord("p"):
                action = "prev"; break
            elif key == ord(" "):
                paused = not paused

    cap.release()
    writer.release()
    return action


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--video", default=None, help="Path to a single input video file.")
    src.add_argument("--videos-dir", default=None,
                     help=f"Folder to scan recursively (default: repo videos/ = {DEFAULT_VIDEOS_DIR}).")
    ap.add_argument("--output", default=None, help="Output path (single-file mode only).")
    ap.add_argument("--out-dir", default="outputs", help="Output root for batch mode.")
    ap.add_argument("--no-show", action="store_true", help="Do not open a preview window.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip videos whose output file already exists.")
    args = ap.parse_args()

    print("Initializing gaze estimator...")
    estimator = GazeEstimator()
    out_root = os.path.abspath(args.out_dir)

    # ── Single-file mode ─────────────────────────────────────────────────
    if args.video is not None:
        os.makedirs(out_root, exist_ok=True)
        out_path = args.output or os.path.join(
            out_root, os.path.splitext(os.path.basename(args.video))[0] + "_gaze.mp4")
        print(f"Writing to {out_path}")
        process_video(args.video, out_path, estimator, show=not args.no_show)
        cv2.destroyAllWindows()
        estimator.close()
        print(f"Saved: {out_path}")
        return

    # ── Batch mode (default: repo videos/ folder) ───────────────────────
    videos_dir = os.path.abspath(args.videos_dir or DEFAULT_VIDEOS_DIR)
    videos = collect_videos(videos_dir)
    if not videos:
        print(f"No video files found under: {videos_dir}")
        estimator.close()
        return

    print(f"Found {len(videos)} video(s) under {videos_dir}")
    done = skipped = idx = 0
    while idx < len(videos):
        video_path = videos[idx]
        out_path = build_output_path(video_path, videos_dir, out_root)
        label = os.path.relpath(video_path, videos_dir)
        if args.skip_existing and os.path.exists(out_path):
            print(f"[{idx+1}/{len(videos)}] skip (exists): {label}")
            skipped += 1; idx += 1
            continue
        print(f"[{idx+1}/{len(videos)}] {label}")
        action = process_video(video_path, out_path, estimator, show=not args.no_show)
        cv2.destroyAllWindows()
        if action == "quit":
            print("  Quit by user."); break
        elif action == "prev":
            idx = max(0, idx - 1)
        else:
            done += 1; idx += 1

    cv2.destroyAllWindows()
    estimator.close()
    print(f"\nDone. {done} processed, {skipped} skipped.")


if __name__ == "__main__":
    main()
