# Emotion Model - Stage 2: Training Report (MNASNet1_0)

Trained with the balanced two-stage recipe (weighted CrossEntropy + label smoothing + mixup + cosine LR with warmup + AMP).

## Configuration

- Model: MNASNet1_0 (12.0 MB)
- Epochs: 5 head-only + 25 full fine-tune
- Batch size: 64
- Head LR: 0.001 | Base LR: 0.0001 | Weight decay: 1e-05
- Label smoothing: 0.1 | Mixup alpha: 0.2
- Optimizer: adam

## Best validation metrics

| Metric | Value |
|---|---|
| Accuracy | 66.72% |
| Balanced accuracy | 66.15% |
| Macro-F1 | 58.15% |
| Weighted-F1 | 69.89% |

- Checkpoint: `checkpoints\best_MNASNet1_0.pth`
- Training curves: `training_curves.png`

## Next step

```
python scripts/evaluate.py --model "MNASNet1_0"
```