# Fusion v1 transfer to `data/final` — classroom

Generated 2026-07-27 05:14 · frozen checkpoint `jetson_deploy\fusion\fusion_attn.pt` · **no retraining** · view realsense_480p

Fusion v1 reported 0.939 clip accuracy on `data/old`. The emotion and motion models had training overlap with those clips, so the `held-out` column below — the 2026-07-25 takes only — is the number that estimates real generalisation.

| Cue(s) masked | overall | train-design | ceiling | test-design | ceiling | held-out only |
|---|---|---|---|---|---|---|
| none | 0.7949 | 0.887 | 0.9774 | 0.3875 | 0.9 | 0.4775 |
| emotion | 0.5426 | 0.6102 | 0.7528 | 0.2437 | 0.8 | 0.2941 |
| gesture | 0.6313 | 0.678 | 0.6907 | 0.425 | 0.9 | 0.474 |
| motion | 0.7177 | 0.822 | 0.9774 | 0.2562 | 0.9 | 0.4152 |
| context | 0.7638 | 0.863 | 0.9774 | 0.325 | 0.9 | 0.4567 |
| emotion+gesture | 0.3237 | 0.3799 | 0.4068 | 0.075 | 0.4 | 0.0692 |

## Per V3 row (no masking beyond the table's own)

|   v3_row | split   | source            |   n |   acc | true_intent   | top_pred   |
|---------:|:--------|:------------------|----:|------:|:--------------|:-----------|
|       18 | train   | raw_take_20260725 |  16 | 0     | F09           | F01        |
|       25 | test    | raw_take_20260725 |  16 | 0     | F04           | F01        |
|       26 | test    | raw_take_20260725 |  16 | 0     | F05           | F02        |
|       27 | test    | raw_take_20260725 |  16 | 0.125 | F06           | F07        |
|       23 | test    | raw_take_20260725 |  16 | 0.188 | F02           | F09        |
|       17 | train   | raw_take_20260725 |  17 | 0.235 | F08           | F04        |
|       30 | test    | raw_take_20260725 |  16 | 0.25  | F09           | F06        |
|       29 | test    | raw_take_20260725 |  16 | 0.312 | F08           | F04        |
|       20 | train   | raw_take_20260725 |  16 | 0.438 | F10           | F10        |
|       21 | train   | raw_take_20260725 |  16 | 0.5   | F10           | F10        |
|       31 | test    | raw_take_20260725 |  16 | 0.5   | F10           | F10        |
|       19 | train   | raw_take_20260725 |  16 | 0.562 | F09           | F09        |
|       22 | test    | raw_take_20260725 |  16 | 0.688 | F01           | F01        |
|       12 | train   | curated_clip      |  54 | 0.778 | F06           | F06        |
|       10 | train   | curated_clip      |  42 | 0.786 | F05           | F05        |
|       28 | test    | raw_take_20260725 |  16 | 0.812 | F07           | F07        |
|        8 | train   | curated_clip      |  61 | 0.951 | F04           | F04        |
|       11 | train   | curated_clip      |  54 | 0.963 | F05           | F05        |
|        2 | train   | curated_clip      |  54 | 0.981 | F01           | F01        |
|       15 | train   | raw_take_20260725 |  16 | 1     | F07           | F07        |
|        4 | train   | raw_take_20260725 |  16 | 1     | F02           | F02        |
|        3 | train   | curated_clip      |  40 | 1     | F02           | F02        |
|        7 | train   | curated_clip      |  70 | 1     | F04           | F04        |
|        5 | train   | curated_clip      |  59 | 1     | F03           | F03        |
|        1 | train   | curated_clip      |  42 | 1     | F01           | F01        |
|       16 | train   | curated_clip      |  55 | 1     | F08           | F08        |
|       13 | train   | raw_take_20260725 |  16 | 1     | F06           | F06        |
|       14 | train   | curated_clip      |  48 | 1     | F07           | F07        |
|       24 | test    | raw_take_20260725 |  16 | 1     | F03           | F03        |