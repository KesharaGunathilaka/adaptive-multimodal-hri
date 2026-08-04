# Unimodal results on `data/final_merged` — complete dataset (classroom + kitchen, train + test)

Generated 2026-08-04 22:43 · 2869 clips · 62 V3 rows · contexts: classroom, kitchen

Clip-level = mean of softmax over windows where the model fired, then argmax. **headline** additionally excludes row #58's 24 test clips that are row #49's train footage re-used with masked cues (`docs/DECISIONS.md` 2026-08-03, `headline_eval=False`).

## 1. Headline (test split, headline_eval only)

| Modality | Clips | Clip acc | Clip macro-F1 | Window acc | Window macro-F1 | Obs. rate |
|---|---|---|---|---|---|---|
| emotion | 856 | 0.7617 | 0.6568 | 0.8489 | 0.8361 | 1.0 |
| gesture | 877 | 0.8643 | 0.8355 | 0.9497 | 0.9458 | 0.999 |
| motion | 979 | 0.665 | 0.5885 | 0.6469 | 0.5763 | 0.999 |
| context | 892 | 0.9709 | 0.4915 | 0.9592 | 0.389 | 1.0 |

## 2. Train / val / test (all test clips, incl. #58's 24)

| Modality | train acc | train F1 | val acc | val F1 | test acc | test F1 |
|---|---|---|---|---|---|---|
| emotion | 0.9589 | 0.9596 | 0.7524 | 0.7479 | 0.7617 | 0.6568 |
| gesture | 1.0 | 1.0 | 0.9966 | 0.9977 | 0.8643 | 0.8355 |
| motion | 0.655 | 0.5605 | 0.6708 | 0.6823 | 0.671 | 0.5905 |
| context | 0.9556 | 0.4841 | 0.9467 | 0.6444 | 0.9716 | 0.4919 |

## 3. Held-out vs fine-tune-overlap clips (the number that matters)

`curated_clip` = migrated from `data/old` — emotion and motion were fine-tuned on (a subset of) these; their number here is not a generalisation estimate. The `raw_take_*` sessions are genuinely unseen by every model.

| Modality | curated_clip acc (n) | F1 | raw_take (all) acc (n) | F1 |
|---|---|---|---|---|
| emotion | 0.9504 (n=1108) | 0.9373 | 0.8163 (n=1562) | — |
| gesture | 0.9929 (n=991) | 0.9892 | 0.9289 (n=1590) | — |
| motion | 0.8617 (n=1106) | 0.7795 | 0.5372 (n=1761) | — |
| context | 0.9733 (n=1049) | 0.4892 | 0.9516 (n=1674) | — |

## 4. By context (classroom vs kitchen — now both complete)

| Modality | classroom acc | F1 | kitchen acc | F1 |
|---|---|---|---|---|
| emotion | 0.8792 (n=1432) | 0.8704 | 0.8635 (n=1238) | 0.8468 |
| gesture | 0.9476 (n=1375) | 0.9382 | 0.9602 (n=1206) | 0.966 |
| motion | 0.7314 (n=1534) | 0.6068 | 0.5829 (n=1333) | 0.5545 |
| context | 0.9993 (n=1424) | 0.4998 | 0.9169 (n=1299) | 0.1913 |

## 5. By resolution class (confound: also correlated with context, N8/N9)

| Modality | 1080p | 480p | 4k | 576p | 720p |
|---|---|---|---|---|---|
| emotion | 0.8513 (n=269) | 0.9275 (n=1283) | 0.7731 (n=454) | 0.7895 (n=76) | 0.8469 (n=588) |
| gesture | 0.9244 (n=291) | 0.966 (n=1177) | 0.9301 (n=472) | 0.9474 (n=76) | 0.9628 (n=565) |
| motion | 0.5279 (n=305) | 0.7509 (n=1337) | 0.595 (n=516) | 0.7368 (n=76) | 0.5861 (n=633) |
| context | 0.9965 (n=287) | 0.985 (n=1264) | 0.921 (n=481) | 0.9079 (n=76) | 0.9285 (n=615) |

## 6. Designed-missing rows — did the cue come out missing?

| Modality | Clips | Observation rate | Per scenario |
|---|---|---|---|
| emotion | 199 | 1.0 | S22_F01=1.0, S25_F04=1.0, S49_F01=1.0, S58_F06=1.0 |
| gesture | 286 | 1.0 | S09_F04=1.0, S12_F06=1.0, S23_F02=1.0, S43_F06=1.0, S53_F02=1.0, S58_F06=1.0 |
| context | 146 | 1.0 | S06_F03=1.0, S24_F03=1.0, S56_F04=1.0 |

## 7.1 emotion detail

**Per class (clip-level, all test)**

|          |   precision |   recall |    f1 |   support |
|:---------|------------:|---------:|------:|----------:|
| Surprise |       0.713 |    1     | 0.832 |        72 |
| Fear     |       0.7   |    0.847 | 0.766 |       215 |
| Disgust  |       0.877 |    0.822 | 0.849 |       321 |
| Happy    |       0.925 |    0.886 | 0.905 |       403 |
| Sad      |       0.89  |    0.858 | 0.874 |       614 |
| Anger    |       0.835 |    0.974 | 0.899 |       384 |
| Neutral  |       0.948 |    0.835 | 0.888 |       661 |

**Confusion (rows = truth)**

|               |   Surprise |   Fear |   Disgust |   Happy |   Sad |   Anger |   Neutral |
|:--------------|-----------:|-------:|----------:|--------:|------:|--------:|----------:|
| true_Surprise |         72 |      0 |         0 |       0 |     0 |       0 |         0 |
| true_Fear     |         20 |    182 |         0 |       2 |     1 |       0 |        10 |
| true_Disgust  |          0 |      2 |       264 |       0 |    11 |      41 |         3 |
| true_Happy    |          1 |     12 |         1 |     357 |     5 |      17 |        10 |
| true_Sad      |          0 |     24 |        24 |      22 |   527 |      10 |         7 |
| true_Anger    |          1 |      2 |         7 |       0 |     0 |     374 |         0 |
| true_Neutral  |          7 |     38 |         5 |       5 |    48 |       6 |       552 |

**Worst scenarios**

| scenario_dir   |   v3_row | intent   | split   | context   | source            | gt_emotion   |   n |   acc |
|:---------------|---------:|:---------|:--------|:----------|:------------------|:-------------|----:|------:|
| S42_F06        |       42 | F06      | val     | kitchen   | raw_take_20260707 | Disgust      |  11 | 0     |
| S40_F05        |       40 | F05      | val     | kitchen   | raw_take_20260728 | Neutral      |  11 | 0     |
| S53_F02        |       53 | F02      | test    | kitchen   | raw_take_20260728 | Fear         |  10 | 0     |
| S41_F05        |       41 | F05      | train   | kitchen   | raw_take_20260728 | Neutral      |  15 | 0.133 |
| S16_F08        |       16 | F08      | val     | classroom | curated_clip      | Disgust      |   7 | 0.143 |
| S60_F08        |       60 | F08      | test    | kitchen   | raw_take_20260728 | Disgust      |   6 | 0.167 |
| S17_F08        |       17 | F08      | val     | classroom | raw_take_20260725 | Disgust      |  12 | 0.333 |
| S46_F08        |       46 | F08      | val     | kitchen   | curated_clip      | Disgust      |  12 | 0.333 |
| S18_F01        |       18 | F01      | test    | classroom | raw_take_20260725 | Happy        |  48 | 0.354 |
| S39_F04        |       39 | F04      | val     | kitchen   | raw_take_20260728 | Sad          |  11 | 0.364 |

## 7.2 gesture detail

**Per class (clip-level, all test)**

|               |   precision |   recall |    f1 |   support |
|:--------------|------------:|---------:|------:|----------:|
| idle          |       0.954 |    0.947 | 0.951 |       457 |
| wave          |       0.906 |    1     | 0.951 |       378 |
| point         |       0.921 |    0.913 | 0.917 |       241 |
| thumbs_up     |       0.996 |    0.878 | 0.933 |       254 |
| thumbs_down   |       0.932 |    0.991 | 0.96  |       429 |
| beckoning     |       0.97  |    0.986 | 0.978 |       291 |
| raise_hand    |       1     |    0.839 | 0.913 |       218 |
| both_hands_up |       1     |    0.997 | 0.998 |       313 |

**Confusion (rows = truth)**

|                    |   idle |   wave |   point |   thumbs_up |   thumbs_down |   beckoning |   raise_hand |   both_hands_up |
|:-------------------|-------:|-------:|--------:|------------:|--------------:|------------:|-------------:|----------------:|
| true_idle          |    433 |      2 |      15 |           0 |             2 |           5 |            0 |               0 |
| true_wave          |      0 |    378 |       0 |           0 |             0 |           0 |            0 |               0 |
| true_point         |      0 |      0 |     220 |           0 |            21 |           0 |            0 |               0 |
| true_thumbs_up     |     16 |      2 |       2 |         223 |             7 |           4 |            0 |               0 |
| true_thumbs_down   |      2 |      0 |       1 |           1 |           425 |           0 |            0 |               0 |
| true_beckoning     |      2 |      0 |       1 |           0 |             1 |         287 |            0 |               0 |
| true_raise_hand    |      0 |     35 |       0 |           0 |             0 |           0 |          183 |               0 |
| true_both_hands_up |      1 |      0 |       0 |           0 |             0 |           0 |            0 |             312 |

**Worst scenarios**

| scenario_dir   |   v3_row | intent   | split   | context   | source            | gt_gesture   |   n |   acc |
|:---------------|---------:|:---------|:--------|:----------|:------------------|:-------------|----:|------:|
| S57_F05        |       57 | F05      | test    | kitchen   | raw_take_20260706 | idle         |  12 | 0     |
| S25_F04        |       25 | F04      | test    | classroom | raw_take_20260725 | raise_hand   |  52 | 0.327 |
| S55_F03        |       55 | F03      | test    | kitchen   | raw_take          | point        |   6 | 0.333 |
| S59_F07        |       59 | F07      | test    | kitchen   | raw_take          | thumbs_up    |   6 | 0.333 |
| S57_F05        |       57 | F05      | test    | kitchen   | raw_take_20260602 | idle         |  16 | 0.625 |
| S59_F07        |       59 | F07      | test    | kitchen   | raw_take_20260728 | thumbs_up    |   6 | 0.667 |
| S27_F06        |       27 | F06      | test    | classroom | raw_take_20260725 | point        |  53 | 0.755 |
| S28_F07        |       28 | F07      | test    | classroom | raw_take_20260725 | thumbs_up    |  55 | 0.8   |
| S29_F08        |       29 | F08      | test    | classroom | raw_take_20260725 | thumbs_up    |  49 | 0.816 |
| S54_F02        |       54 | F02      | test    | kitchen   | raw_take_20260728 | idle         |   6 | 0.833 |

## 7.3 motion detail

**Per class (clip-level, all test)**

|               |   precision |   recall |    f1 |   support |
|:--------------|------------:|---------:|------:|----------:|
| sitting       |       0.916 |    0.846 | 0.88  |       967 |
| standing      |       0.51  |    0.496 | 0.503 |       706 |
| walking       |       0.593 |    0.801 | 0.682 |       790 |
| stepping_back |       0.443 |    0.243 | 0.314 |       404 |

**Confusion (rows = truth)**

|                    |   sitting |   standing |   walking |   stepping_back |
|:-------------------|----------:|-----------:|----------:|----------------:|
| true_sitting       |       818 |         82 |        67 |               0 |
| true_standing      |        42 |        350 |       217 |              97 |
| true_walking       |        27 |        104 |       633 |              26 |
| true_stepping_back |         6 |        150 |       150 |              98 |

**Worst scenarios**

| scenario_dir   |   v3_row | intent   | split   | context   | source            | gt_motion     |   n |   acc |
|:---------------|---------:|:---------|:--------|:----------|:------------------|:--------------|----:|------:|
| S04_F02        |        4 | F02      | val     | classroom | raw_take_20260725 | stepping_back |  12 |     0 |
| S04_F02        |        4 | F02      | train   | classroom | raw_take_20260725 | stepping_back |  36 |     0 |
| S13_F06        |       13 | F06      | train   | classroom | raw_take_20260725 | standing      |  36 |     0 |
| S35_F02        |       35 | F02      | train   | kitchen   | curated_clip      | stepping_back |  32 |     0 |
| S31_F10        |       31 | F10      | test    | classroom | raw_take_20260725 | stepping_back |  52 |     0 |
| S34_F02        |       34 | F02      | train   | kitchen   | curated_clip      | stepping_back |  40 |     0 |
| S33_F01        |       33 | F01      | train   | kitchen   | raw_take_20260728 | standing      |  12 |     0 |
| S33_F01        |       33 | F01      | train   | kitchen   | raw_take_20260603 | standing      |  12 |     0 |
| S40_F05        |       40 | F05      | val     | kitchen   | raw_take_20260728 | standing      |  11 |     0 |
| S34_F02        |       34 | F02      | val     | kitchen   | curated_clip      | stepping_back |   6 |     0 |

## 7.4 context detail

**Per class (clip-level, all test)**

|             |   precision |   recall |    f1 |   support |
|:------------|------------:|---------:|------:|----------:|
| classroom   |       0.981 |    0.999 | 0.99  |      1424 |
| kitchen     |       1     |    0.917 | 0.957 |      1299 |
| hospital    |       0     |    0     | 0     |         0 |
| cloth_store |       0     |    0     | 0     |         0 |
| museum      |       0     |    0     | 0     |         0 |

**Confusion (rows = truth)**

|                  |   classroom |   kitchen |   hospital |   cloth_store |   museum |
|:-----------------|------------:|----------:|-----------:|--------------:|---------:|
| true_classroom   |        1423 |         0 |          1 |             0 |        0 |
| true_kitchen     |          27 |      1191 |         76 |             2 |        3 |
| true_hospital    |           0 |         0 |          0 |             0 |        0 |
| true_cloth_store |           0 |         0 |          0 |             0 |        0 |
| true_museum      |           0 |         0 |          0 |             0 |        0 |

**Worst scenarios**

| scenario_dir   |   v3_row | intent   | split   | context   | source            | gt_context   |   n |   acc |
|:---------------|---------:|:---------|:--------|:----------|:------------------|:-------------|----:|------:|
| S52_F01        |       52 | F01      | val     | kitchen   | raw_take_20260727 | kitchen      |  12 | 0.167 |
| S52_F01        |       52 | F01      | train   | kitchen   | raw_take_20260727 | kitchen      |  18 | 0.444 |
| S40_F05        |       40 | F05      | train   | kitchen   | raw_take_20260727 | kitchen      |  10 | 0.5   |
| S47_F08        |       47 | F08      | train   | kitchen   | raw_take_20260727 | kitchen      |  16 | 0.5   |
| S50_F10        |       50 | F10      | val     | kitchen   | raw_take_20260728 | kitchen      |  12 | 0.5   |
| S48_F01        |       48 | F01      | train   | kitchen   | curated_clip      | kitchen      |  38 | 0.658 |
| S50_F10        |       50 | F10      | train   | kitchen   | raw_take_20260728 | kitchen      |  37 | 0.73  |
| S60_F08        |       60 | F08      | test    | kitchen   | raw_take_20260727 | kitchen      |  29 | 0.793 |
| S53_F02        |       53 | F02      | test    | kitchen   | raw_take_20260727 | kitchen      |  29 | 0.828 |
| S38_F04        |       38 | F04      | train   | kitchen   | curated_clip      | kitchen      |  29 | 0.828 |