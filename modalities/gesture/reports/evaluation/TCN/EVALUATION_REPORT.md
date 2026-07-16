# Gesture Model - Stage 4: Evaluation Report (TCN)

Generated: 2026-07-15 13:11 · checkpoint `best_TCN.pth`

## 1. Held-out test split (2883 sequences, subject-wise)

| Metric | Value |
|---|---|
| Accuracy | 92.58% |
| Balanced accuracy | 93.26% |
| Macro-F1 | 92.97% |
| Weighted-F1 | 92.58% |

### Per-class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| idle | 0.94 | 0.93 | 0.94 | 1437 |
| wave | 0.91 | 0.93 | 0.92 | 329 |
| point | 0.95 | 1.00 | 0.98 | 60 |
| thumbs_up | 0.89 | 0.90 | 0.89 | 326 |
| thumbs_down | 0.93 | 0.92 | 0.93 | 331 |
| beckoning | 0.89 | 0.91 | 0.90 | 289 |
| raise_hand | 0.93 | 1.00 | 0.96 | 27 |
| both_hands_up | 0.99 | 0.87 | 0.92 | 84 |

Confusion matrix: `confusion_test.png`

## 2. Test breakdown by source dataset

| Source dataset | Sequences | Accuracy | Macro-F1 |
|---|---|---|---|
| custom | 82 | 78.05% | 51.14% |
| jester | 2357 | 92.19% | 75.74% |
| ntu | 444 | 97.30% | 97.38% |

Macro-F1 here spans only the classes each dataset contributes.

## 3. Live test set

_No live_test sequences found — record them (custom/live_test/<label>/) and re-run extraction + prepare_data. This is the deployment metric (guide §10)._

## 4. Deployment budget

- Single-window forward pass (CPU, this machine): **0.96 ms** — the temporal net is negligible next to MediaPipe landmark extraction, as designed for the Jetson Orin Nano.
