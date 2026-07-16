# Emotion Model - Real-World Evaluation (MobileNetV2)

- Checkpoint: `checkpoints/finetuned_MobileNetV2.pth`
- Dataset: `../../videos/dataset/hri-multimodal-intent-v1.0.0` (151 clips, 16 frames sampled/clip)
- Face detected in 141/151 clips (93.4%)

## Overall (clip-level, mean softmax over frames)

| Metric | Value |
|---|---|
| Accuracy | 58.87% |
| Balanced accuracy | 50.80% |
| Macro-F1 | 53.21% |
| Weighted-F1 | 59.65% |

## Per-class metrics

| Emotion   |   Precision |   Recall |   F1-Score |   Support |
|:----------|------------:|---------:|-----------:|----------:|
| Surprise  |       75    |    46.15 |      57.14 |        26 |
| Fear      |       50    |    50    |      50    |        10 |
| Disgust   |       83.33 |    38.46 |      52.63 |        13 |
| Happy     |       40    |    33.33 |      36.36 |        12 |
| Sad       |       21.43 |    42.86 |      28.57 |        14 |
| Anger     |       92.31 |    60    |      72.73 |        20 |
| Neutral   |       67.24 |    84.78 |      75    |        46 |

## Prediction vs. true label distribution

|          |   true_count |   pred_count |
|:---------|-------------:|-------------:|
| Surprise |           26 |           16 |
| Fear     |           10 |           10 |
| Disgust  |           13 |            6 |
| Happy    |           12 |           10 |
| Sad      |           14 |           28 |
| Anger    |           20 |           13 |
| Neutral  |           46 |           58 |

## Notes

- Ground truth is the scenario-level *intended* emotion; the subject acts it while also performing a gesture/motion, so labels are noisier than a curated face dataset.
- See `per_scenario_accuracy.csv` for which scenarios fail and `clip_predictions.csv` for per-clip probabilities.