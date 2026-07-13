# Emotion Model - Real-World Evaluation (EfficientNet-B0)

- Checkpoint: `checkpoints/finetuned_EfficientNet_B0.pth`
- Dataset: `../../videos/dataset/hri-multimodal-intent-v1.0.0` (151 clips, 16 frames sampled/clip)
- Face detected in 141/151 clips (93.4%)

## Overall (clip-level, mean softmax over frames)

| Metric | Value |
|---|---|
| Accuracy | 57.45% |
| Balanced accuracy | 47.08% |
| Macro-F1 | 49.48% |
| Weighted-F1 | 57.49% |

## Per-class metrics

| Emotion   |   Precision |   Recall |   F1-Score |   Support |
|:----------|------------:|---------:|-----------:|----------:|
| Surprise  |       78.57 |    42.31 |      55    |        26 |
| Fear      |       28.57 |    20    |      23.53 |        10 |
| Disgust   |       87.5  |    53.85 |      66.67 |        13 |
| Happy     |       31.58 |    50    |      38.71 |        12 |
| Sad       |       11.76 |    14.29 |      12.9  |        14 |
| Anger     |      100    |    60    |      75    |        20 |
| Neutral   |       64.06 |    89.13 |      74.55 |        46 |

## Prediction vs. true label distribution

|          |   true_count |   pred_count |
|:---------|-------------:|-------------:|
| Surprise |           26 |           14 |
| Fear     |           10 |            7 |
| Disgust  |           13 |            8 |
| Happy    |           12 |           19 |
| Sad      |           14 |           17 |
| Anger    |           20 |           12 |
| Neutral  |           46 |           64 |

## Notes

- Ground truth is the scenario-level *intended* emotion; the subject acts it while also performing a gesture/motion, so labels are noisier than a curated face dataset.
- See `per_scenario_accuracy.csv` for which scenarios fail and `clip_predictions.csv` for per-clip probabilities.