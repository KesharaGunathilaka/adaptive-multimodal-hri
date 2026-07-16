# Emotion Model - Real-World Evaluation (EfficientNet-B0)

- Checkpoint: `checkpoints/best_EfficientNet_B0.pth`
- Dataset: `../../videos/dataset/hri-multimodal-intent-v1.0.0` (1270 clips, 16 frames sampled/clip)
- Face detected in 1234/1270 clips (97.2%)

## Overall (clip-level, mean softmax over frames)

| Metric | Value |
|---|---|
| Accuracy | 56.48% |
| Balanced accuracy | 43.71% |
| Macro-F1 | 43.52% |
| Weighted-F1 | 54.32% |

## Per-class metrics

| Emotion   |   Precision |   Recall |   F1-Score |   Support |
|:----------|------------:|---------:|-----------:|----------:|
| Surprise  |       36.36 |    65.31 |      46.72 |        98 |
| Fear      |       17.02 |    15.09 |      16    |        53 |
| Disgust   |       29.17 |    14.29 |      19.18 |        98 |
| Happy     |       77.78 |    57.34 |      66.01 |       293 |
| Sad       |       32.77 |    21.2  |      25.74 |       184 |
| Anger     |       81.82 |    44.55 |      57.69 |       101 |
| Neutral   |       62.65 |    88.21 |      73.27 |       407 |

## Prediction vs. true label distribution

|          |   true_count |   pred_count |
|:---------|-------------:|-------------:|
| Surprise |           98 |          176 |
| Fear     |           53 |           47 |
| Disgust  |           98 |           48 |
| Happy    |          293 |          216 |
| Sad      |          184 |          119 |
| Anger    |          101 |           55 |
| Neutral  |          407 |          573 |

## Notes

- Ground truth is the scenario-level *intended* emotion; the subject acts it while also performing a gesture/motion, so labels are noisier than a curated face dataset.
- See `per_scenario_accuracy.csv` for which scenarios fail and `clip_predictions.csv` for per-clip probabilities.