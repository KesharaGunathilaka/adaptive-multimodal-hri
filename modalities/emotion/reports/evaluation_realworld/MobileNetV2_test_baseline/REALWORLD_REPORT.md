# Emotion Model - Real-World Evaluation (MobileNetV2)

- Checkpoint: `checkpoints/best_MobileNetV2.pth`
- Dataset: `../../videos/dataset/hri-multimodal-intent-v1.0.0` (151 clips, 16 frames sampled/clip)
- Face detected in 141/151 clips (93.4%)

## Overall (clip-level, mean softmax over frames)

| Metric | Value |
|---|---|
| Accuracy | 44.68% |
| Balanced accuracy | 28.28% |
| Macro-F1 | 24.03% |
| Weighted-F1 | 35.87% |

## Per-class metrics

| Emotion   |   Precision |   Recall |   F1-Score |   Support |
|:----------|------------:|---------:|-----------:|----------:|
| Surprise  |       42.31 |    42.31 |      42.31 |        26 |
| Fear      |        0    |     0    |       0    |        10 |
| Disgust   |        0    |     0    |       0    |        13 |
| Happy     |       33.33 |    50    |      40    |        12 |
| Sad       |        0    |     0    |       0    |        14 |
| Anger     |      100    |    10    |      18.18 |        20 |
| Neutral   |       52.38 |    95.65 |      67.69 |        46 |

## Prediction vs. true label distribution

|          |   true_count |   pred_count |
|:---------|-------------:|-------------:|
| Surprise |           26 |           26 |
| Fear     |           10 |            2 |
| Disgust  |           13 |            0 |
| Happy    |           12 |           18 |
| Sad      |           14 |            9 |
| Anger    |           20 |            2 |
| Neutral  |           46 |           84 |

## Notes

- Ground truth is the scenario-level *intended* emotion; the subject acts it while also performing a gesture/motion, so labels are noisier than a curated face dataset.
- See `per_scenario_accuracy.csv` for which scenarios fail and `clip_predictions.csv` for per-clip probabilities.