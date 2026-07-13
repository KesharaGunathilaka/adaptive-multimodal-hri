"""
Real-world evaluation on the HRI intent dataset (videos/dataset/...).

Runs the deployed inference pipeline (MediaPipe face detection -> tight crop ->
224x224 + ImageNet normalize -> model) on every clip, averages the softmax
probabilities over sampled frames, and scores the clip-level prediction against
the scenario's "Intended Emotion" annotation.

Reports overall metrics, per-class metrics, per-scenario accuracy, a confusion
matrix, and the face-detection rate. Saves CSVs, plots and a markdown report
under reports/evaluation_realworld/<model>/.

Run (from modalities/emotion/):
    python scripts/evaluate_realworld.py                       # deployed MobileNetV2
    python scripts/evaluate_realworld.py --model EfficientNet-B0
    python scripts/evaluate_realworld.py --frames 20 --limit 50   # quick smoke test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import CHECKPOINT_DIR, DEFAULT_MODEL, EMOTION_LABELS, REPORT_DIR, ROOT
from src.engine import get_device
from src.models import ALL_MODELS, safe_name
from src.transforms import get_test_transforms

DATASET_DIR = os.path.abspath(os.path.join(
    ROOT, "..", "..", "videos", "dataset", "hri-multimodal-intent-v1.0.0"))

# Dataset "Intended Emotion" annotation -> RAF-DB class name
INTENDED_TO_LABEL = {
    "neutral": "Neutral", "happy": "Happy", "sad": "Sad", "angry": "Anger",
    "disgust": "Disgust", "surprise": "Surprise", "fear": "Fear",
}


def load_clips(split="all"):
    ann = os.path.join(DATASET_DIR, "annotations")
    clips = pd.read_csv(os.path.join(ann, "splits.csv"))
    scenarios = pd.read_csv(os.path.join(ann, "scenarios.csv"))
    scenarios["scenario"] = scenarios["Scenario ID"]
    clips["scenario"] = clips["scenario_id"].str.split("_").str[0]
    merged = clips.merge(
        scenarios[["scenario", "Intended Emotion"]], on="scenario", how="left")
    merged["true_label"] = merged["Intended Emotion"].str.strip().str.lower().map(
        INTENDED_TO_LABEL)
    missing = merged["true_label"].isna()
    if missing.any():
        print(f"[warn] {missing.sum()} clips have unmapped emotions; skipping them")
        merged = merged[~missing]
    if split != "all":
        merged = merged[merged["split_subject"] == split]
    return merged.reset_index(drop=True)


def sample_frame_indices(frame_count, n):
    if frame_count <= 0:
        return []
    return sorted(set(np.linspace(0, frame_count - 1, min(n, frame_count)).astype(int)))


def detect_best_face(detectors, rgb):
    """Highest-score detection across detectors (full-range first)."""
    for det in detectors:
        results = det.process(rgb)
        if results.detections:
            return max(results.detections, key=lambda d: d.score[0])
    return None


def predict_clip(video_path, model, transform, detectors, device, n_frames):
    """Mean softmax over sampled frames with a detected face.

    Returns (probs[7] or None, frames_read, frames_with_face).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0, 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = sample_frame_indices(frame_count, n_frames)

    tensors, frames_read = [], 0
    for idx in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frames_read += 1
        h, w, _ = frame.shape
        det = detect_best_face(detectors, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if det is None:
            continue
        box = det.location_data.relative_bounding_box
        x, y = max(0, int(box.xmin * w)), max(0, int(box.ymin * h))
        bw, bh = int(box.width * w), int(box.height * h)
        face = frame[y:y + bh, x:x + bw]
        if face.size == 0:
            continue
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        from PIL import Image
        tensors.append(transform(Image.fromarray(rgb)))
    cap.release()

    if not tensors:
        return None, frames_read, 0
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1).mean(dim=0)
    return probs.cpu().numpy(), frames_read, len(tensors)


def plot_confusion(cm, labels, path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Real-world confusion matrix (color = row-normalized, text = counts)")
    plt.xlabel("Predicted")
    plt.ylabel("True (intended emotion)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(ALL_MODELS.keys()))
    ap.add_argument("--checkpoint", default=None,
                    help="Defaults to checkpoints/best_<Model>.pth")
    ap.add_argument("--frames", type=int, default=16,
                    help="Frames sampled evenly per clip (default 16)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N clips (smoke test)")
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"],
                    help="Subject-disjoint split (splits.csv split_subject) to score")
    ap.add_argument("--tag", default=None,
                    help="Report subdirectory name (default: <Model>[_<split>])")
    args = ap.parse_args()

    device = get_device()
    sname = safe_name(args.model)
    ckpt = args.checkpoint or os.path.join(CHECKPOINT_DIR, f"best_{sname}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = ALL_MODELS[args.model](pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    transform = get_test_transforms()
    # Full-range detector first (subjects are 2-5m from the camera in this
    # dataset), close-range as fallback — video.py's close-range-only detector
    # finds a face in only ~20% of these clips.
    detectors = [
        mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.4),
        mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5),
    ]

    clips = load_clips(args.split)
    if args.limit:
        clips = clips.head(args.limit)
    print(f"Model: {args.model}  ({ckpt})")
    print(f"Clips: {len(clips)} (split={args.split})  |  "
          f"frames/clip: {args.frames}  |  device: {device}")

    rows = []
    for i, row in clips.iterrows():
        video_path = os.path.join(DATASET_DIR, row["filepath"])
        probs, frames_read, faces = predict_clip(
            video_path, model, transform, detectors, device, args.frames)
        rec = {
            "clip_id": row["clip_id"], "scenario_id": row["scenario_id"],
            "subject_id": row["subject_id"], "context": row["scenario_id"][:3],
            "true": row["true_label"], "frames_read": frames_read,
            "frames_with_face": faces,
        }
        if probs is not None:
            pred_idx = int(np.argmax(probs))
            rec["pred"] = EMOTION_LABELS[pred_idx]
            rec["confidence"] = float(probs[pred_idx])
            for j, lbl in enumerate(EMOTION_LABELS):
                rec[f"p_{lbl}"] = float(probs[j])
        else:
            rec["pred"] = None
            rec["confidence"] = None
        rows.append(rec)
        if (i + 1) % 100 == 0 or (i + 1) == len(clips):
            print(f"  [{i + 1}/{len(clips)}] processed")

    df = pd.DataFrame(rows)
    default_tag = sname if args.split == "all" else f"{sname}_{args.split}"
    out_dir = os.path.join(REPORT_DIR, "evaluation_realworld", args.tag or default_tag)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "clip_predictions.csv"), index=False)

    detected = df[df["pred"].notna()].copy()
    det_rate = len(detected) / len(df) if len(df) else 0.0
    print(f"\nFace detected in {len(detected)}/{len(df)} clips ({det_rate*100:.1f}%)")

    labels, preds = detected["true"].to_numpy(), detected["pred"].to_numpy()
    present = [l for l in EMOTION_LABELS if (labels == l).any() or (preds == l).any()]
    overall = {
        "Accuracy": accuracy_score(labels, preds),
        "Balanced accuracy": balanced_accuracy_score(labels, preds),
        "Macro-F1": f1_score(labels, preds, average="macro", labels=present),
        "Weighted-F1": f1_score(labels, preds, average="weighted", labels=present),
        "Clips scored": len(detected),
        "Face-detection rate": det_rate,
    }
    print("\n=== Overall (clip-level) ===")
    for k, v in overall.items():
        print(f"  {k}: {v*100:.2f}%" if isinstance(v, float) else f"  {k}: {v}")

    report = classification_report(labels, preds, labels=EMOTION_LABELS,
                                   digits=4, output_dict=True, zero_division=0)
    per_class_df = pd.DataFrame([
        {"Emotion": name, "Precision": report[name]["precision"],
         "Recall": report[name]["recall"], "F1-Score": report[name]["f1-score"],
         "Support": int(report[name]["support"])}
        for name in EMOTION_LABELS
    ])
    print("\n=== Per-class ===")
    print(per_class_df.to_string(index=False))
    per_class_df.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)

    cm = confusion_matrix(labels, preds, labels=EMOTION_LABELS)
    pd.DataFrame(cm, index=EMOTION_LABELS, columns=EMOTION_LABELS).to_csv(
        os.path.join(out_dir, "confusion_matrix.csv"))
    plot_confusion(cm, EMOTION_LABELS, os.path.join(out_dir, "confusion_matrix.png"))

    # Prediction distribution vs. true distribution (shows systematic bias)
    dist = pd.DataFrame({
        "true_count": pd.Series(labels).value_counts().reindex(EMOTION_LABELS, fill_value=0),
        "pred_count": pd.Series(preds).value_counts().reindex(EMOTION_LABELS, fill_value=0),
    })
    dist.to_csv(os.path.join(out_dir, "label_distribution.csv"))

    per_scenario = (detected.assign(correct=labels == preds)
                    .groupby(["scenario_id", "true"])
                    .agg(clips=("clip_id", "count"), accuracy=("correct", "mean"))
                    .reset_index().sort_values("accuracy"))
    per_scenario.to_csv(os.path.join(out_dir, "per_scenario_accuracy.csv"), index=False)
    print("\n=== Per-scenario (worst 10) ===")
    print(per_scenario.head(10).to_string(index=False))

    pd.DataFrame([overall]).to_csv(os.path.join(out_dir, "overall_metrics.csv"), index=False)

    lines = [
        f"# Emotion Model - Real-World Evaluation ({args.model})\n",
        f"- Checkpoint: `{os.path.relpath(ckpt, ROOT)}`",
        f"- Dataset: `{os.path.relpath(DATASET_DIR, ROOT)}` "
        f"({len(df)} clips, {args.frames} frames sampled/clip)",
        f"- Face detected in {len(detected)}/{len(df)} clips ({det_rate*100:.1f}%)\n",
        "## Overall (clip-level, mean softmax over frames)\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {overall['Accuracy']*100:.2f}% |",
        f"| Balanced accuracy | {overall['Balanced accuracy']*100:.2f}% |",
        f"| Macro-F1 | {overall['Macro-F1']*100:.2f}% |",
        f"| Weighted-F1 | {overall['Weighted-F1']*100:.2f}% |\n",
        "## Per-class metrics\n",
        per_class_df.assign(
            Precision=lambda d: (d["Precision"] * 100).round(2),
            Recall=lambda d: (d["Recall"] * 100).round(2),
            **{"F1-Score": lambda d: (d["F1-Score"] * 100).round(2)},
        ).to_markdown(index=False),
        "\n## Prediction vs. true label distribution\n",
        dist.to_markdown(),
        "\n## Notes\n",
        "- Ground truth is the scenario-level *intended* emotion; the subject acts "
        "it while also performing a gesture/motion, so labels are noisier than a "
        "curated face dataset.",
        "- See `per_scenario_accuracy.csv` for which scenarios fail and "
        "`clip_predictions.csv` for per-clip probabilities.",
    ]
    with open(os.path.join(out_dir, "REALWORLD_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
