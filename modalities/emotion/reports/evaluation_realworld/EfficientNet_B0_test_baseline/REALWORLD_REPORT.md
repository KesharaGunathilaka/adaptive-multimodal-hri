# Emotion Model - Real-World Evaluation (EfficientNet-B0)

- Checkpoint: `checkpoints/best_EfficientNet_B0.pth`
- Dataset: `../../videos/dataset/hri-multimodal-intent-v1.0.0` (151 clips, 16 frames sampled/clip)
- Face detected in 141/151 clips (93.4%)

## Overall (clip-level, mean softmax over frames)

| Metric | Value |
|---|---|
| Accuracy | 50.35% |
| Balanced accuracy | 38.69% |
| Macro-F1 | 37.27% |
| Weighted-F1 | 48.55% |

## Per-class metrics

| Emotion   |   Precision |   Recall |   F1-Score |   Support |
|:----------|------------:|---------:|-----------:|----------:|
| Surprise  |       52.17 |    46.15 |      48.98 |        26 |
| Fear      |       25    |    60    |      35.29 |        10 |
| Disgust   |       50    |     7.69 |      13.33 |        13 |
| Happy     |       25    |    25    |      25    |        12 |
| Sad       |        0    |     0    |       0    |        14 |
| Anger     |      100    |    45    |      62.07 |        20 |
| Neutral   |       67.8  |    86.96 |      76.19 |        46 |

## Prediction vs. true label distribution

|          |   true_count |   pred_count |
|:---------|-------------:|-------------:|
| Surprise |           26 |           23 |
| Fear     |           10 |           24 |
| Disgust  |           13 |            2 |
| Happy    |           12 |           12 |
| Sad      |           14 |           12 |
| Anger    |           20 |            9 |
| Neutral  |           46 |           59 |

## Notes

- Ground truth is the scenario-level *intended* emotion; the subject acts it while also performing a gesture/motion, so labels are noisier than a curated face dataset.
- See `per_scenario_accuracy.csv` for which scenarios fail and `clip_predictions.csv` for per-clip probabilities.