"""Gaze estimation on a video file or folder (single sub-model).

Run from repo root or this folder:
    python inference/video.py --input ../../../videos/Classroom
    python inference/video.py --input clip.mp4 --no-show --save

Controls: 'n' next video, space pause, 'q'/ESC quit.
"""
import argparse
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.gaze_estimation.gaze_estimator import GazeEstimator

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def find_videos(root):
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def draw(frame, gaze):
    if not gaze.has_face:
        cv2.putText(frame, "No face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return
    if gaze.gaze_point and gaze.face_bbox:
        cx = (gaze.face_bbox[0] + gaze.face_bbox[2]) // 2
        cy = (gaze.face_bbox[1] + gaze.face_bbox[3]) // 2
        cv2.arrowedLine(frame, (cx, cy), (int(gaze.gaze_point[0]), int(gaze.gaze_point[1])),
                        (0, 255, 0), 2, tipLength=0.2)
    status = "ROBOT" if gaze.looking_at_robot else "AWAY"
    color = (0, 255, 0) if gaze.looking_at_robot else (0, 165, 255)
    cv2.putText(frame, f"yaw {gaze.yaw:+.0f} pitch {gaze.pitch:+.0f} -> {status} "
                f"eng {gaze.engagement:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Video file or folder.")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    videos = find_videos(args.input)
    if not videos:
        print(f"No videos found under: {args.input}")
        sys.exit(1)

    estimator = GazeEstimator()
    print(f"Gaze estimator ready | {len(videos)} video(s)")
    out_dir = Path(__file__).resolve().parents[1] / "runs" / "gaze_video"

    for vi, path in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"  ! could not open {path}")
            continue
        print(f"[{vi}/{len(videos)}] {path.name}")
        writer = None
        if args.save:
            out_dir.mkdir(parents=True, exist_ok=True)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
            writer = cv2.VideoWriter(str(out_dir / f"{path.stem}_gaze.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

        idx, paused, quit_all = 0, False, False
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                if args.stride > 1 and idx % args.stride != 0:
                    continue
                draw(frame, estimator.estimate(frame))
                if writer is not None:
                    writer.write(frame)
                if not args.no_show:
                    cv2.imshow("Gaze Estimation", frame)
            if not args.no_show:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    quit_all = True; break
                if key == ord("n"):
                    break
                if key == ord(" "):
                    paused = not paused
        cap.release()
        if writer is not None:
            writer.release()
        if quit_all:
            break
    estimator.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
