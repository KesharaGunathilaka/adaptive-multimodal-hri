import os
import sys
import argparse
import subprocess
from pathlib import Path

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))


def main():
    parser = argparse.ArgumentParser(description="Process all videos in a folder and save annotated outputs")
    parser.add_argument("--input-folder", required=True, help="Folder containing video files")
    parser.add_argument("--output-folder", default="annotated_videos", help="Folder to save annotated videos")
    parser.add_argument("--slow", action="store_true", help="Run playback slower")
    parser.add_argument("--save", action="store_true", help="Save annotated videos")
    args = parser.parse_args()

    input_folder = Path(args.input_folder).expanduser().resolve()
    if not input_folder.exists() or not input_folder.is_dir():
        print(f"ERROR: Input folder not found: {input_folder}")
        sys.exit(1)

    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = sorted([p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() in video_exts])

    if not videos:
        print(f"No supported video files found in {input_folder}")
        sys.exit(0)

    print(f"Found {len(videos)} video(s) in {input_folder}")

    for video_path in videos:
        print(f"\nProcessing: {video_path.name}")
        cmd = [sys.executable, str(Path(PROJECT_ROOT) / "inference" / "video.py"), "--video", str(video_path), "--headless"]
        if args.slow:
            cmd.append("--slow")
        if args.save:
            cmd.append("--save")

        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()
