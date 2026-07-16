# Gesture Model - Stage 2: Training Report (TCN)

Generated: 2026-07-15 13:11. Recipe: class-weighted CrossEntropy + label smoothing, AdamW, cosine LR with warmup, early stopping on val macro-F1.

## Configuration

- Model: TCN (2.61 MB)
- Epochs: 100 (patience 15) | Batch size: 256
- LR: 0.001 | Weight decay: 0.0001 | Dropout: 0.3
- Hidden size: 128 | Label smoothing: 0.1
- Window: 32 frames | Feature dim: 185

## Best validation metrics

| Metric | Value |
|---|---|
| Accuracy | 93.24% |
| Balanced accuracy | 92.79% |
| Macro-F1 | 92.77% |
| Weighted-F1 | 93.24% |
| Best epoch | 82 |

- Checkpoint: `checkpoints/best_TCN.pth`
- Training curves: `training_curves.png`

## Next step

```
python scripts/evaluate.py --model TCN
```