"""Object detection on a video file or folder (single sub-model).

Run from the object_detection/ folder:
    python inference/video.py --input ../../../videos/Kitchen
    python inference/video.py --input clip.mp4 --no-show --save

Controls: 'n' next video, space pause, 'q'/ESC quit.
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector import ContextDetector

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def find_videos(root):
    if os.path.isfile(root):
        return [root]
    found = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                found.append(os.path.join(dirpath, fn))
    return sorted(found)


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

    detector = ContextDetector()
    print(f"Detector ready | {len(videos)} video(s)")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "runs", "object_video")

    for vi, path in enumerate(videos, 1):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"  ! could not open {path}")
            continue
        print(f"[{vi}/{len(videos)}] {os.path.basename(path)}")
        writer = None
        if args.save:
            os.makedirs(out_dir, exist_ok=True)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
            stem = os.path.splitext(os.path.basename(path))[0]
            writer = cv2.VideoWriter(os.path.join(out_dir, f"{stem}_objects.mp4"),
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
                annotated, _, counts, stable = detector.process_frame(frame)
                cv2.putText(annotated, "stable: " + (", ".join(sorted(stable)) if stable else "-"),
                            (10, annotated.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if writer is not None:
                    writer.write(annotated)
                if not args.no_show:
                    cv2.imshow("Object Detection", annotated)
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
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
