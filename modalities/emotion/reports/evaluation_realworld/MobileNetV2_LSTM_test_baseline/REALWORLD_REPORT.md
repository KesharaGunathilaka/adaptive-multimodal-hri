# Emotion Model - Real-World Evaluation (MobileNetV2-LSTM)

- Checkpoint: `checkpoints/best_MobileNetV2_LSTM.pth`
- Dataset: `../../videos/dataset/hri-multimodal-intent-v1.0.0` (151 clips, 16 frames sampled/clip)
- Face detected in 141/151 clips (93.4%)

## Overall (clip-level, mean softmax over frames)

| Metric | Value |
|---|---|
| Accuracy | 44.68% |
| Balanced accuracy | 35.29% |
| Macro-F1 | 35.11% |
| Weighted-F1 | 42.60% |

## Per-class metrics

| Emotion   |   Precision |   Recall |   F1-Score |   Support |
|:----------|------------:|---------:|-----------:|----------:|
| Surprise  |       63.64 |    26.92 |      37.84 |        26 |
| Fear      |       28.57 |    60    |      38.71 |        10 |
| Disgust   |      100    |     7.69 |      14.29 |        13 |
| Happy     |       66.67 |    33.33 |      44.44 |        12 |
| Sad       |        6.9  |    14.29 |       9.3  |        14 |
| Anger     |      100    |    20    |      33.33 |        20 |
| Neutral   |       56.52 |    84.78 |      67.83 |        46 |

## Prediction vs. true label distribution

|          |   true_count |   pred_count |
|:---------|-------------:|-------------:|
| Surprise |           26 |           11 |
| Fear     |           10 |           21 |
| Disgust  |           13 |            1 |
| Happy    |           12 |            6 |
| Sad      |           14 |           29 |
| Anger    |           20 |            4 |
| Neutral  |           46 |           69 |

## Notes

- Ground truth is the scenario-level *intended* emotion; the subject acts it while also performing a gesture/motion, so labels are noisier than a curated face dataset.
- See `per_scenario_accuracy.csv` for which scenarios fail and `clip_predictions.csv` for per-clip probabilities.