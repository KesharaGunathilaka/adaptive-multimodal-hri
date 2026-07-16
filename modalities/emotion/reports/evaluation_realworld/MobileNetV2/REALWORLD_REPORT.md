# Emotion Model - Real-World Evaluation (MobileNetV2)

- Checkpoint: `checkpoints/best_MobileNetV2.pth`
- Dataset: `../../videos/dataset/hri-multimodal-intent-v1.0.0` (1270 clips, 16 frames sampled/clip)
- Face detected in 1234/1270 clips (97.2%)

## Overall (clip-level, mean softmax over frames)

| Metric | Value |
|---|---|
| Accuracy | 56.73% |
| Balanced accuracy | 36.73% |
| Macro-F1 | 35.34% |
| Weighted-F1 | 50.85% |

## Per-class metrics

| Emotion   |   Precision |   Recall |   F1-Score |   Support |
|:----------|------------:|---------:|-----------:|----------:|
| Surprise  |       35.61 |    47.96 |      40.87 |        98 |
| Fear      |        0    |     0    |       0    |        53 |
| Disgust   |       57.14 |     8.16 |      14.29 |        98 |
| Happy     |       67.08 |    73.04 |      69.93 |       293 |
| Sad       |       46.67 |    26.63 |      33.91 |       184 |
| Anger     |       76.92 |     9.9  |      17.54 |       101 |
| Neutral   |       57.85 |    91.4  |      70.86 |       407 |

## Prediction vs. true label distribution

|          |   true_count |   pred_count |
|:---------|-------------:|-------------:|
| Surprise |           98 |          132 |
| Fear     |           53 |            8 |
| Disgust  |           98 |           14 |
| Happy    |          293 |          319 |
| Sad      |          184 |          105 |
| Anger    |          101 |           13 |
| Neutral  |          407 |          643 |

## Notes

- Ground truth is the scenario-level *intended* emotion; the subject acts it while also performing a gesture/motion, so labels are noisier than a curated face dataset.
- See `per_scenario_accuracy.csv` for which scenarios fail and `clip_predictions.csv` for per-clip probabilities.