# Emotion Model - Real-World Evaluation (MobileNetV2-LSTM)

- Checkpoint: `checkpoints/best_MobileNetV2_LSTM.pth`
- Dataset: `../../videos/dataset/hri-multimodal-intent-v1.0.0` (1270 clips, 16 frames sampled/clip)
- Face detected in 1234/1270 clips (97.2%)

## Overall (clip-level, mean softmax over frames)

| Metric | Value |
|---|---|
| Accuracy | 54.29% |
| Balanced accuracy | 40.86% |
| Macro-F1 | 42.24% |
| Weighted-F1 | 52.37% |

## Per-class metrics

| Emotion   |   Precision |   Recall |   F1-Score |   Support |
|:----------|------------:|---------:|-----------:|----------:|
| Surprise  |       49.49 |    50    |      49.75 |        98 |
| Fear      |       11.69 |    16.98 |      13.85 |        53 |
| Disgust   |       42.86 |    18.37 |      25.71 |        98 |
| Happy     |       85.38 |    49.83 |      62.93 |       293 |
| Sad       |       34.56 |    25.54 |      29.38 |       184 |
| Anger     |       61.02 |    35.64 |      45    |       101 |
| Neutral   |       56.15 |    89.68 |      69.06 |       407 |

## Prediction vs. true label distribution

|          |   true_count |   pred_count |
|:---------|-------------:|-------------:|
| Surprise |           98 |           99 |
| Fear     |           53 |           77 |
| Disgust  |           98 |           42 |
| Happy    |          293 |          171 |
| Sad      |          184 |          136 |
| Anger    |          101 |           59 |
| Neutral  |          407 |          650 |

## Notes

- Ground truth is the scenario-level *intended* emotion; the subject acts it while also performing a gesture/motion, so labels are noisier than a curated face dataset.
- See `per_scenario_accuracy.csv` for which scenarios fail and `clip_predictions.csv` for per-clip probabilities.