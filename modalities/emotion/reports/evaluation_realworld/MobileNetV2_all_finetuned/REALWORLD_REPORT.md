# Emotion Model - Real-World Evaluation (MobileNetV2)

- Checkpoint: `checkpoints/finetuned_MobileNetV2.pth`
- Dataset: `../../videos/dataset/hri-multimodal-intent-v1.0.0` (1270 clips, 16 frames sampled/clip)
- Face detected in 1234/1270 clips (97.2%)

## Overall (clip-level, mean softmax over frames)

| Metric | Value |
|---|---|
| Accuracy | 85.82% |
| Balanced accuracy | 82.70% |
| Macro-F1 | 83.40% |
| Weighted-F1 | 85.77% |

## Per-class metrics

| Emotion   |   Precision |   Recall |   F1-Score |   Support |
|:----------|------------:|---------:|-----------:|----------:|
| Surprise  |       81.05 |    78.57 |      79.79 |        98 |
| Fear      |       78.18 |    81.13 |      79.63 |        53 |
| Disgust   |       94.12 |    65.31 |      77.11 |        98 |
| Happy     |       95.18 |    80.89 |      87.45 |       293 |
| Sad       |       73.31 |    94.02 |      82.38 |       184 |
| Anger     |       88.78 |    86.14 |      87.44 |       101 |
| Neutral   |       87.3  |    92.87 |      90    |       407 |

## Prediction vs. true label distribution

|          |   true_count |   pred_count |
|:---------|-------------:|-------------:|
| Surprise |           98 |           95 |
| Fear     |           53 |           55 |
| Disgust  |           98 |           68 |
| Happy    |          293 |          249 |
| Sad      |          184 |          236 |
| Anger    |          101 |           98 |
| Neutral  |          407 |          433 |

## Notes

- Ground truth is the scenario-level *intended* emotion; the subject acts it while also performing a gesture/motion, so labels are noisier than a curated face dataset.
- See `per_scenario_accuracy.csv` for which scenarios fail and `clip_predictions.csv` for per-clip probabilities.