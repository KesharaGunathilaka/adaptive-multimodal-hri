from pathlib import Path
import sys
import os
import json
import time
from datetime import datetime

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Hardcoded path to videos root directory
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VIDEOS_ROOT = _REPO_ROOT / "videos"

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from custom_detect import detect_context_objects, DetectionSmoother
import config as cfg

# =========================
# CONFIG
# =========================
VIDEO_PATH = "../../../videos/k1/20260509_190437.mp4"


# ─────────────────────────────────────────────
# Drawing helpers (shared with object_run.py)
# ─────────────────────────────────────────────
def draw_detections(frame, result, frame_idx, total_frames, fps):
    """Draw bounding boxes, labels, and an info bar on the frame."""
    overlay = frame.copy()

    for det in result["detections"]:
        cat = det["category"]
        label_text = det["label"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        zone = det["zone"]

        color = cfg.CATEGORY_COLORS.get(cat, cfg.DEFAULT_COLOR)

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.BOX_THICKNESS)

        # label with background
        text = f"{label_text} {conf:.0%} [{zone}]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(
            text, font, cfg.LABEL_FONT_SCALE, cfg.LABEL_THICKNESS
        )

        label_y1 = max(y1 - th - baseline - 6, 0)
        label_y2 = y1
        cv2.rectangle(overlay, (x1, label_y1), (x1 + tw + 6, label_y2), color, -1)
        cv2.addWeighted(
            overlay, cfg.LABEL_BG_ALPHA, frame, 1 - cfg.LABEL_BG_ALPHA, 0, frame
        )
        overlay = frame.copy()

        cv2.putText(
            frame,
            text,
            (x1 + 3, y1 - baseline - 2),
            font,
            cfg.LABEL_FONT_SCALE,
            (255, 255, 255),
            cfg.LABEL_THICKNESS,
            cv2.LINE_AA,
        )

    # --- info bar at top ---
    h, w = frame.shape[:2]
    bar_h = 34
    cv2.rectangle(frame, (0, 0), (w, bar_h), (25, 25, 25), -1)

    timestamp_sec = frame_idx / fps if fps > 0 else 0
    mins, secs = divmod(int(timestamp_sec), 60)
    progress = frame_idx / total_frames * 100 if total_frames > 0 else 0

    counts = result["counts"]
    active = {k: v for k, v in counts.items() if v > 0}
    obj_str = "  |  ".join(f"{k}: {v}" for k, v in sorted(active.items()))
    info = f"[{mins:02d}:{secs:02d}] Frame {frame_idx}/{total_frames} ({progress:.0f}%)   {obj_str}"

    cv2.putText(
        frame,
        info,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    return frame


# ─────────────────────────────────────────────
# Timeline chart generation
# ─────────────────────────────────────────────
def generate_timeline(frame_results, fps, output_path):
    """
    Create a horizontal timeline chart showing when each category
    was detected across the video duration.
    """
    # Collect all categories that appear at least once
    all_categories = sorted(
        {cat for fr in frame_results for cat in fr["counts"] if fr["counts"][cat] > 0}
    )

    if not all_categories:
        print("  No objects detected — skipping timeline chart.")
        return

    # Build per-category binary timeline
    n_frames = len(frame_results)
    timestamps = [fr["timestamp_sec"] for fr in frame_results]

    # Colour map (convert BGR → RGB, normalise to 0-1)
    def bgr_to_rgb_norm(bgr):
        return (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)

    fig, ax = plt.subplots(figsize=(14, max(3, len(all_categories) * 0.7 + 1.5)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bar_height = 0.6

    for row_idx, cat in enumerate(all_categories):
        color_bgr = cfg.CATEGORY_COLORS.get(cat, cfg.DEFAULT_COLOR)
        color_rgb = bgr_to_rgb_norm(color_bgr)

        # Find contiguous spans where category is present
        spans = []
        in_span = False
        span_start = 0

        for i, fr in enumerate(frame_results):
            is_present = fr["counts"].get(cat, 0) > 0
            if is_present and not in_span:
                span_start = timestamps[i]
                in_span = True
            elif not is_present and in_span:
                spans.append((span_start, timestamps[i] - span_start))
                in_span = False

        if in_span:
            spans.append((span_start, timestamps[-1] - span_start + 0.1))

        # Draw spans
        for start, width in spans:
            ax.barh(
                row_idx,
                width,
                left=start,
                height=bar_height,
                color=color_rgb,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.3,
            )

    # Styling
    ax.set_yticks(range(len(all_categories)))
    ax.set_yticklabels(all_categories, fontsize=10, fontweight="bold", color="white")
    ax.set_xlabel("Time (seconds)", fontsize=11, color="white", labelpad=8)
    ax.set_title(
        "Object Detection Timeline",
        fontsize=14,
        fontweight="bold",
        color="white",
        pad=12,
    )
    ax.tick_params(axis="x", colors="white", labelsize=9)
    ax.tick_params(axis="y", colors="white")
    ax.invert_yaxis()  # first category on top

    # Grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle="--", alpha=0.2, color="white")

    # Legend
    patches = []
    for cat in all_categories:
        c = bgr_to_rgb_norm(cfg.CATEGORY_COLORS.get(cat, cfg.DEFAULT_COLOR))
        patches.append(mpatches.Patch(color=c, label=cat))
    ax.legend(
        handles=patches,
        loc="upper right",
        fontsize=8,
        facecolor="#16213e",
        edgecolor="white",
        labelcolor="white",
    )

    for spine in ax.spines.values():
        spine.set_color("#444")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Timeline chart saved: {output_path}")


# ─────────────────────────────────────────────
# JSON output
# ─────────────────────────────────────────────
def save_json_results(frame_results, video_info, output_path):
    """Save structured JSON with metadata + per-frame detections."""
    output = {
        "metadata": {
            "video_file": video_info["path"],
            "total_frames": video_info["total_frames"],
            "fps": round(video_info["fps"], 2),
            "resolution": video_info["resolution"],
            "duration_sec": round(video_info["duration_sec"], 2),
            "model": cfg.MODEL_PATH,
            "confidence_threshold": cfg.CONFIDENCE_THRESHOLD,
            "processed_at": datetime.now().isoformat(),
            "frame_sample_interval": cfg.FRAME_SAMPLE_INTERVAL,
        },
        "summary": _build_summary(frame_results),
        "frames": frame_results,
    }

    with open(str(output_path), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  ✓ JSON results saved: {output_path}")


def _build_summary(frame_results):
    """Aggregate statistics across all frames."""
    total_detections = sum(len(fr["detections"]) for fr in frame_results)
    category_frames = {}  # category → number of frames it appears in
    category_max_conf = {}  # category → max confidence seen

    for fr in frame_results:
        for cat, cnt in fr["counts"].items():
            if cnt > 0:
                category_frames[cat] = category_frames.get(cat, 0) + 1
                for det in fr["detections"]:
                    if det["category"] == cat:
                        prev = category_max_conf.get(cat, 0)
                        category_max_conf[cat] = max(prev, det["confidence"])

    n = len(frame_results) if frame_results else 1
    category_summary = {}
    for cat in sorted(category_frames):
        category_summary[cat] = {
            "frames_present": category_frames[cat],
            "presence_pct": round(category_frames[cat] / n * 100, 1),
            "max_confidence": category_max_conf.get(cat, 0),
        }

    return {
        "total_frames_processed": len(frame_results),
        "total_detections": total_detections,
        "categories_detected": list(sorted(category_frames.keys())),
        "per_category": category_summary,
    }


# ─────────────────────────────────────────────
# Main video processing pipeline
# ─────────────────────────────────────────────
def process_video(video_path: str, show_preview: bool = True):
    """Process a video file and produce JSON results + timeline chart."""

    # Resolve video path: if it doesn't exist, try prepending videos root
    video_path = Path(video_path)
    if not video_path.exists():
        video_path = _VIDEOS_ROOT / video_path

    video_path = str(video_path.resolve())

    print(f"\n{'=' * 60}")
    print(f"  Object Detection — Video Pipeline")
    print(f"{'=' * 60}")
    print(f"  Video : {video_path}")
    print(f"  Model : {cfg.MODEL_PATH}")
    print(f"  Conf  : {cfg.CONFIDENCE_THRESHOLD}")
    print(f"  Sample: every {cfg.FRAME_SAMPLE_INTERVAL} frame(s)")
    print(f"{'=' * 60}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Error: Cannot open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    video_info = {
        "path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "resolution": f"{w}x{h}",
        "duration_sec": duration_sec,
    }

    print(f"  Resolution : {w}x{h}")
    print(f"  FPS        : {fps:.1f}")
    print(f"  Duration   : {duration_sec:.1f}s ({total_frames} frames)")
    print()

    # Ensure output directory
    output_dir = _THIS_DIR / cfg.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_results = []
    frame_idx = 0
    t_start = time.time()

    # Temporal smoother — stabilises detections across frames
    smoother = DetectionSmoother()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames based on sampling interval
        if frame_idx % cfg.FRAME_SAMPLE_INTERVAL != 0:
            continue

        raw_result = detect_context_objects(frame)
        result = smoother.update(raw_result)
        timestamp_sec = round(frame_idx / fps, 3) if fps > 0 else 0

        frame_record = {
            "frame": frame_idx,
            "timestamp_sec": timestamp_sec,
            "counts": result["counts"],
            "detections": result["detections"],
            "zones": result["zones"],
        }
        frame_results.append(frame_record)

        # Progress
        if frame_idx % 50 == 0:
            elapsed = time.time() - t_start
            pct = frame_idx / total_frames * 100
            active = {k: v for k, v in result["counts"].items() if v > 0}
            print(
                f"  [{pct:5.1f}%] Frame {frame_idx}/{total_frames}  "
                f"({elapsed:.0f}s elapsed)  {active}"
            )

        # Live preview
        if show_preview:
            vis = draw_detections(frame.copy(), result, frame_idx, total_frames, fps)
            # Resize for display if very large
            disp_w = min(w, 960)
            if disp_w < w:
                scale = disp_w / w
                vis = cv2.resize(vis, (disp_w, int(h * scale)))
            cv2.imshow("Object Detection - Video", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                print("\n  Interrupted by user (ESC).")
                break

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n  Processing complete: {len(frame_results)} frames in {elapsed:.1f}s")

    # ── Save outputs ──
    print(f"\n  Saving outputs to: {output_dir}/")

    json_path = output_dir / cfg.JSON_OUTPUT_FILE
    save_json_results(frame_results, video_info, json_path)

    timeline_path = output_dir / cfg.TIMELINE_CHART_FILE
    generate_timeline(frame_results, fps, timeline_path)

    print(f"\n{'=' * 60}")
    print(f"  Done!")
    print(f"{'=' * 60}\n")

    return frame_results


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Parse args
    video = VIDEO_PATH
    preview = cfg.SHOW_PREVIEW

    args = sys.argv[1:]
    if "--no-preview" in args:
        preview = False
        args.remove("--no-preview")
    if args:
        video = args[0]

    process_video(video, show_preview=preview)
