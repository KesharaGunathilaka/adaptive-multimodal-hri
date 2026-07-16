# CNN-LSTM (MobileNetV2-LSTM) vs. deployed MobileNetV2 — real-world comparison

2026-07-15. Temporal model experiment: `EmotionCNNLSTM` (`src/models.py`,
registry name `MobileNetV2-LSTM`) — MobileNetV2 per-frame features + a
1-layer LSTM (hidden 256) over the clip's frame sequence, 3.8M params /
14.6 MB fp32, unidirectional so it stays streamable on Jetson Orin Nano.

Pipeline (mirrors the deployed model's):
1. **RAF-DB training** — `scripts/train_lstm_rafdb.py`, pseudo-sequences of 4
   augmented views/image, backbone init from `best_MobileNetV2.pth`.
   Best RAF-DB test: **81.0% acc / 73.9 macro-F1** (plain MobileNetV2: 81.9% acc).
2. **Baseline real-world eval** — `scripts/evaluate_realworld.py` (sequence
   models get the ordered frame sequence instead of mean softmax).
3. **Fine-tune on real-world sequences** — `scripts/finetune_lstm_realworld.py`:
   train-split clips as frame sequences (4 random subsets/clip/epoch) + equal
   RAF-DB pseudo-sequences, softened class weights, backbone LR 2e-5 /
   LSTM+head LR 1e-4, selection on val-clip macro-F1 (best 30.7%).
4. **Re-eval** on the held-out test split and on all clips.

## Test split (151 clips, unseen subjects) — clip-level

| Model | Accuracy | Balanced acc | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| MobileNetV2 baseline | 44.68 | 28.28 | 24.03 | 35.87 |
| MobileNetV2 fine-tuned (deployed) | **58.87** | **50.80** | **53.21** | **59.65** |
| EfficientNet-B0 fine-tuned | 57.45 | 47.08 | 49.48 | 57.49 |
| MobileNetV2-LSTM baseline | 44.68 | 35.29 | 35.11 | 42.60 |
| MobileNetV2-LSTM fine-tuned | 56.74 | 47.72 | 50.98 | 57.31 |

## All real-world clips (1270; train/val subjects were fine-tune data)

| Model | Accuracy | Balanced acc | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| MobileNetV2 baseline | 56.73 | 36.73 | 35.34 | 50.85 |
| MobileNetV2 fine-tuned (deployed) | **85.82** | **82.70** | **83.40** | **85.77** |
| MobileNetV2-LSTM baseline | 54.29 | 40.86 | 42.24 | 52.37 |
| MobileNetV2-LSTM fine-tuned | 85.41 | 81.16 | 82.40 | 85.33 |

All-clips numbers for fine-tuned models are optimistic: crops from the same
train/val subjects were in the fine-tune set — the test-split column is the
honest generalization measure.

## Verdict

- Before fine-tuning the LSTM generalizes clearly better than the plain model
  (test macro-F1 35.1 vs 24.0) — multi-view sequence training adds robustness.
- After fine-tuning it does NOT beat the deployed MobileNetV2 (51.0 vs 53.2
  test macro-F1, and essentially tied on all clips) while costing ~8x the
  per-clip compute (every frame goes through the CNN either way, but the
  deployed model can also run single-frame).
- Likely cause: the dataset's "intended emotion" is acted and held for the
  whole clip, so there is little temporal signal for the LSTM to exploit
  beyond what mean-softmax already captures; per-clip labels also give the
  LSTM 8x fewer supervised samples than per-crop fine-tuning.
- **Keep `finetuned_MobileNetV2.pth` deployed.** Checkpoints
  `best_MobileNetV2_LSTM.pth` / `finetuned_MobileNetV2_LSTM.pth` are kept for
  reference; per-run reports live in `MobileNetV2_LSTM_*` subdirectories.
