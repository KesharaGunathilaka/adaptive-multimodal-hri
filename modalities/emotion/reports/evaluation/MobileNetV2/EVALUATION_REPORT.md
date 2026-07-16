# Emotion Model - Stage 4: Evaluation Report (MobileNetV2)

- Checkpoint: `checkpoints/finetuned_MobileNetV2.pth`
- Test images: 3068

## Overall metrics

| Metric | Value |
|---|---|
| Accuracy | 81.91% |
| Balanced accuracy | 76.00% |
| Macro-F1 | 74.32% |
| Weighted-F1 | 82.28% |
| Macro precision | 73.21% |
| Macro recall | 76.00% |

## Per-class metrics

| Emotion   |   Precision |   Recall |   F1-Score |   Support |
|:----------|------------:|---------:|-----------:|----------:|
| Surprise  |       80.97 |    86.63 |      83.7  |       329 |
| Fear      |       62.86 |    59.46 |      61.11 |        74 |
| Disgust   |       45.41 |    58.75 |      51.23 |       160 |
| Happy     |       95.83 |    89.28 |      92.44 |      1185 |
| Sad       |       73.22 |    84.1  |      78.29 |       478 |
| Anger     |       71.43 |    80.25 |      75.58 |       162 |
| Neutral   |       82.78 |    73.53 |      77.88 |       680 |


## Confidence

- Mean confidence on correct predictions: 0.770
- Mean confidence on incorrect predictions: 0.536
- A well-calibrated model is more confident when correct.

## Figures

- `confusion_matrix.png`
- `per_class_metrics.png`
- `confidence_distribution.png`

## Notes

- Minority emotions (Fear, Disgust, Anger) are the hardest; watch their recall in the per-class table and the off-diagonal mass in the confusion matrix (Fear is often confused with Surprise).