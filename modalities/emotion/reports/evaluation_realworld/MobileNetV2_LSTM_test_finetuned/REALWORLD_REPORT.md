# Emotion Model - Real-World Evaluation (MobileNetV2-LSTM)

- Checkpoint: `checkpoints/finetuned_MobileNetV2_LSTM.pth`
- Dataset: `../../videos/dataset/hri-multimodal-intent-v1.0.0` (151 clips, 16 frames sampled/clip)
- Face detected in 141/151 clips (93.4%)

## Overall (clip-level, mean softmax over frames)

| Metric | Value |
|---|---|
| Accuracy | 56.74% |
| Balanced accuracy | 47.72% |
| Macro-F1 | 50.98% |
| Weighted-F1 | 57.31% |

## Per-class metrics

| Emotion   |   Precision |   Recall |   F1-Score |   Support |
|:----------|------------:|---------:|-----------:|----------:|
| Surprise  |       73.33 |    42.31 |      53.66 |        26 |
| Fear      |       40    |    20    |      26.67 |        10 |
| Disgust   |      100    |    30.77 |      47.06 |        13 |
| Happy     |       63.64 |    58.33 |      60.87 |        12 |
| Sad       |       18.75 |    42.86 |      26.09 |        14 |
| Anger     |      100    |    55    |      70.97 |        20 |
| Neutral   |       61.9  |    84.78 |      71.56 |        46 |

## Prediction vs. true label distribution

|          |   true_count |   pred_count |
|:---------|-------------:|-------------:|
| Surprise |           26 |           15 |
| Fear     |           10 |            5 |
| Disgust  |           13 |            4 |
| Happy    |           12 |           11 |
| Sad      |           14 |           32 |
| Anger    |           20 |           11 |
| Neutral  |           46 |           63 |

## Notes

- Ground truth is the scenario-level *intended* emotion; the subject acts it while also performing a gesture/motion, so labels are noisier than a curated face dataset.
- See `per_scenario_accuracy.csv` for which scenarios fail and `clip_predictions.csv` for per-clip probabilities.