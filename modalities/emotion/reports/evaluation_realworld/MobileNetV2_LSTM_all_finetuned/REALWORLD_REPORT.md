# Emotion Model - Real-World Evaluation (MobileNetV2-LSTM)

- Checkpoint: `checkpoints/finetuned_MobileNetV2_LSTM.pth`
- Dataset: `../../videos/dataset/hri-multimodal-intent-v1.0.0` (1270 clips, 16 frames sampled/clip)
- Face detected in 1234/1270 clips (97.2%)

## Overall (clip-level, mean softmax over frames)

| Metric | Value |
|---|---|
| Accuracy | 85.41% |
| Balanced accuracy | 81.16% |
| Macro-F1 | 82.40% |
| Weighted-F1 | 85.33% |

## Per-class metrics

| Emotion   |   Precision |   Recall |   F1-Score |   Support |
|:----------|------------:|---------:|-----------:|----------:|
| Surprise  |       74.04 |    78.57 |      76.24 |        98 |
| Fear      |       86.96 |    75.47 |      80.81 |        53 |
| Disgust   |      100    |    58.16 |      73.55 |        98 |
| Happy     |       96.37 |    81.57 |      88.35 |       293 |
| Sad       |       71.14 |    95.11 |      81.4  |       184 |
| Anger     |       86.14 |    86.14 |      86.14 |       101 |
| Neutral   |       87.73 |    93.12 |      90.35 |       407 |

## Prediction vs. true label distribution

|          |   true_count |   pred_count |
|:---------|-------------:|-------------:|
| Surprise |           98 |          104 |
| Fear     |           53 |           46 |
| Disgust  |           98 |           57 |
| Happy    |          293 |          248 |
| Sad      |          184 |          246 |
| Anger    |          101 |          101 |
| Neutral  |          407 |          432 |

## Notes

- Ground truth is the scenario-level *intended* emotion; the subject acts it while also performing a gesture/motion, so labels are noisier than a curated face dataset.
- See `per_scenario_accuracy.csv` for which scenarios fail and `clip_predictions.csv` for per-clip probabilities.