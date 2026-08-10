# Emotion Model - Stage 1: Model Comparison Report

Candidate ImageNet-pretrained backbones trained with the identical balanced two-stage recipe (weighted CE + label smoothing + mixup + cosine warmup) and ranked by **macro-F1** within a **20 MB** deployment budget.

- Search length: 5 head-only + 20 full-finetune epochs/model
- Batch size: 64
- Selection metric: macro-F1 (every emotion weighted equally)

## Results

| model             |   params_m |   size_mb | within_budget   |   accuracy |   balanced_acc |   f1_weighted |   macro_f1 |   gpu_ms |   cpu_ms |   train_time_s |
|:------------------|-----------:|----------:|:----------------|-----------:|---------------:|--------------:|-----------:|---------:|---------:|---------------:|
| EfficientNet-B0   |       4.02 |     15.48 | True            |      77.05 |          75.61 |         78.13 |      69.76 |     1.75 |     9.59 |          298.3 |
| MobileNetV3-Large |       4.21 |     16.16 | True            |      71.97 |          71.36 |         73.88 |      63.75 |     1.33 |     4.39 |          295.5 |
| MobileNetV2       |       2.23 |      8.65 | True            |      69.56 |          70.02 |         71.94 |      61.34 |     1.09 |    11.47 |          292.6 |
| MNASNet1_0        |       3.11 |     12.01 | True            |      66.98 |          65.55 |         69.5  |      58.28 |     1.03 |    91.19 |          288.4 |


## Recommendation from this table alone: EfficientNet-B0

- Best macro-F1 among models within the 20 MB budget.
- Size 15.48 MB, macro-F1 69.76%, accuracy 77.05%.

> Note: these numbers come from a shortened search on RAF-DB. The final reported accuracy comes
> from the full training run (Stage 2) — but even the full RAF-DB run does not decide deployment;
> see below.

## Superseded by real-world validation — MobileNetV2 is the actual deployed model

RAF-DB is curated, close-up portrait photography. This project's own footage has subjects 2–5 m
from camera with 40–90 px faces — a different distribution. Backbone selection therefore cannot
stop at RAF-DB accuracy: each full-RAF-DB-trained candidate (`best_<Arch>.pth`) was fine-tuned
once more on real face crops from `data/final_merged` (identical recipe for every architecture —
`scripts/46_finetune_emotion_backbone.py`) and re-scored on the same held-out real-world test
clips (`scripts/45_compare_emotion_backbones.py`, 856 clips, clip-level mean-softmax,
`split=="test"`, `headline_eval` rows).

| Model | RAF-DB-only acc / macro-F1 | Real-data fine-tuned acc / macro-F1 |
|---|---|---|
| **MobileNetV2** | 43.22% / 35.83% | **77.45% / 66.81%** |
| MobileNetV3-Large | 34.11% / 28.38% | 69.51% / 61.00% |
| EfficientNet-B0 | 37.85% / 31.82% | 66.12% / 56.80% |

Under this deployment-realistic criterion the RAF-DB ranking does not just weaken — it **inverts**.
MobileNetV2 beats EfficientNet-B0 by +11.3 points accuracy / +10.0 points macro-F1 after fine-tuning,
despite being the smallest model in the table above (2.23M params vs 4.02M). MobileNetV2 was
selected on this real-world fine-tuned performance, not RAF-DB benchmark accuracy. Full numbers:
`results/realworld_eval_merged/emotion_backbone_comparison.json`.

## Next step

```
python scripts/train.py --model "MobileNetV2"
```