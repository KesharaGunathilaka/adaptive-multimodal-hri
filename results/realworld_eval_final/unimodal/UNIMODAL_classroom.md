# Unimodal results on `data/final` — classroom

Generated 2026-07-27 08:42 · 1440 clips · views: phone_1080p, phone_4k, realsense_480p

Clip-level = mean of softmax over the windows where the model fired, then argmax. Rows the V3 table marks `[missing]` for a cue are excluded from that cue's accuracy and reported separately.

## 1. Headline

| Modality | Clips scored | Clip acc | Clip macro-F1 | Window acc | Window macro-F1 | Windows observed |
|---|---|---|---|---|---|---|
| emotion | 1296 | 0.8017 | 0.7173 | 0.7676 | 0.6906 | 1.0 |
| gesture | 1345 | 0.797 | 0.7998 | 0.7887 | 0.7913 | 0.999 |
| motion | 1439 | 0.7248 | 0.6387 | 0.72 | 0.632 | 0.999 |
| context | 1391 | 0.9993 | 0.4998 | 0.9984 | 0.4996 | 1.0 |

## 2. Train-design vs test-design rows

| Modality | train acc | train macro-F1 | test acc | test macro-F1 |
|---|---|---|---|---|
| emotion | 0.8363 | 0.775 | 0.7009 | 0.4359 |
| gesture | 0.8407 | 0.7333 | 0.7057 | 0.6208 |
| motion | 0.7718 | 0.6973 | 0.6295 | 0.4326 |
| context | 1.0 | 1.0 | 0.9977 | 0.4994 |

## 2b. Held-out vs training-overlap clips

`curated_clip` rows were migrated from `data/old` — the emotion model was fine-tuned on them, so its number there is not a generalisation estimate. `raw_take_20260725` is the genuinely unseen collection.

| Modality | curated_clip acc | macro-F1 | raw_take_20260725 acc | macro-F1 |
|---|---|---|---|---|
| emotion | 0.9689 (n=579) | 0.9578 | 0.6667 (n=717) | 0.5157 |
| gesture | 0.8817 (n=524) | 0.7479 | 0.743 (n=821) | 0.738 |
| motion | 0.9204 (n=578) | 0.8792 | 0.5935 (n=861) | 0.4231 |
| context | 1.0 (n=579) | 1.0 | 0.9988 (n=812) | 0.4997 |

## 3. By camera view (clip-level)

| Modality | phone_1080p | phone_4k | realsense_480p |
|---|---|---|---|
| emotion | 0.6835 (n=237) | 0.6736 (n=239) | 0.8732 (n=820) |
| gesture | 0.7179 (n=273) | 0.7455 (n=275) | 0.8419 (n=797) |
| motion | 0.5263 (n=285) | 0.6167 (n=287) | 0.8258 (n=867) |
| context | 0.9963 (n=269) | 1.0 (n=270) | 1.0 (n=852) |

## 4. Designed-missing rows — was the cue really unobservable?

Observation rate is the fraction of windows where the model still produced an output. Low is what the design asks for.

| Modality | Clips | Observation rate | Per scenario |
|---|---|---|---|
| emotion | 144 | 0.999 | S22_F01=1.0, S25_F04=1.0, S30_F09=0.997 |
| gesture | 94 | 1.0 | S12_F06=1.0, S23_F02=1.0 |
| context | 49 | 1.0 | S24_F03=1.0 |

## 5.1 emotion detail

**Per class (clip-level)**

|          |   precision |   recall |    f1 |   support |
|:---------|------------:|---------:|------:|----------:|
| Surprise |       0.411 |    0.975 | 0.578 |        40 |
| Fear     |       0.808 |    0.239 | 0.368 |        88 |
| Disgust  |       0.75  |    0.644 | 0.693 |       149 |
| Happy    |       0.888 |    0.838 | 0.862 |       198 |
| Sad      |       0.671 |    0.737 | 0.702 |       205 |
| Anger    |       0.919 |    0.931 | 0.925 |       245 |
| Neutral  |       0.873 |    0.911 | 0.892 |       371 |

**Confusion (rows = truth)**

|               |   Surprise |   Fear |   Disgust |   Happy |   Sad |   Anger |   Neutral |
|:--------------|-----------:|-------:|----------:|--------:|------:|--------:|----------:|
| true_Surprise |         39 |      0 |         0 |       1 |     0 |       0 |         0 |
| true_Fear     |         48 |     21 |         0 |       2 |     1 |       0 |        16 |
| true_Disgust  |          0 |      0 |        96 |       0 |    36 |      12 |         5 |
| true_Happy    |          4 |      2 |         1 |     166 |     9 |       3 |        13 |
| true_Sad      |          0 |      0 |        20 |      15 |   151 |       5 |        14 |
| true_Anger    |          0 |      1 |        10 |       1 |     4 |     228 |         1 |
| true_Neutral  |          4 |      2 |         1 |       2 |    24 |       0 |       338 |

**Worst scenarios**

| scenario_dir   |   v3_row | intent   | split_design   | source            | gt_emotion   |   n |   acc | top_error   |
|:---------------|---------:|:---------|:---------------|:------------------|:-------------|----:|------:|:------------|
| S23_F02        |       23 | F02      | test           | raw_take_20260725 | Fear         |  40 | 0.05  | Surprise×25 |
| S18_F09        |       18 | F09      | train          | raw_take_20260725 | Happy        |  48 | 0.333 | Neutral×13  |
| S04_F02        |        4 | F02      | train          | raw_take_20260725 | Fear         |  48 | 0.396 | Surprise×23 |
| S17_F08        |       17 | F08      | train          | raw_take_20260725 | Disgust      |  49 | 0.449 | Sad×17      |
| S29_F08        |       29 | F08      | test           | raw_take_20260725 | Disgust      |  45 | 0.6   | Sad×16      |
| S20_F10        |       20 | F10      | train          | raw_take_20260725 | Sad          |  48 | 0.604 | Disgust×10  |
| S21_F10        |       21 | F10      | train          | raw_take_20260725 | Sad          |  48 | 0.646 | Neutral×5   |
| S31_F10        |       31 | F10      | test           | raw_take_20260725 | Sad          |  48 | 0.667 | Neutral×6   |
| S19_F09        |       19 | F09      | train          | raw_take_20260725 | Neutral      |  49 | 0.755 | Sad×7       |
| S24_F03        |       24 | F03      | test           | raw_take_20260725 | Neutral      |  49 | 0.796 | Sad×7       |

Coverage: 0 clip(s) never observed; per-view observation {'phone_1080p': 1.0, 'phone_4k': 1.0, 'realsense_480p': 1.0}

## 5.2 gesture detail

**Per class (clip-level)**

|               |   precision |   recall |    f1 |   support |
|:--------------|------------:|---------:|------:|----------:|
| idle          |       0.576 |    0.898 | 0.702 |       186 |
| wave          |       0.84  |    0.558 | 0.671 |       283 |
| point         |       0.623 |    0.878 | 0.729 |        49 |
| thumbs_up     |       0.966 |    0.755 | 0.848 |       151 |
| thumbs_down   |       0.953 |    0.859 | 0.904 |       213 |
| beckoning     |       0.816 |    0.785 | 0.8   |       107 |
| raise_hand    |       0.698 |    0.82  | 0.754 |       172 |
| both_hands_up |       0.995 |    0.989 | 0.992 |       184 |

**Confusion (rows = truth)**

|                    |   idle |   wave |   point |   thumbs_up |   thumbs_down |   beckoning |   raise_hand |   both_hands_up |
|:-------------------|-------:|-------:|--------:|------------:|--------------:|------------:|-------------:|----------------:|
| true_idle          |    167 |      0 |       0 |           0 |             4 |           0 |           15 |               0 |
| true_wave          |     87 |    158 |       0 |           1 |             0 |           0 |           37 |               0 |
| true_point         |      2 |      0 |      43 |           1 |             3 |           0 |            0 |               0 |
| true_thumbs_up     |      7 |      0 |       9 |         114 |             1 |          18 |            1 |               1 |
| true_thumbs_down   |      7 |      0 |      13 |           2 |           183 |           0 |            8 |               0 |
| true_beckoning     |     18 |      0 |       4 |           0 |             1 |          84 |            0 |               0 |
| true_raise_hand    |      1 |     30 |       0 |           0 |             0 |           0 |          141 |               0 |
| true_both_hands_up |      1 |      0 |       0 |           0 |             0 |           1 |            0 |             182 |

**Worst scenarios**

| scenario_dir   |   v3_row | intent   | split_design   | source            | gt_gesture   |   n |   acc | top_error     |
|:---------------|---------:|:---------|:---------------|:------------------|:-------------|----:|------:|:--------------|
| S30_F09        |       30 | F09      | test           | raw_take_20260725 | wave         |  48 | 0.25  | idle×35       |
| S25_F04        |       25 | F04      | test           | raw_take_20260725 | raise_hand   |  48 | 0.375 | wave×29       |
| S19_F09        |       19 | F09      | train          | raw_take_20260725 | wave         |  49 | 0.49  | idle×25       |
| S18_F09        |       18 | F09      | train          | raw_take_20260725 | wave         |  48 | 0.5   | idle×24       |
| S24_F03        |       24 | F03      | test           | raw_take_20260725 | beckoning    |  49 | 0.531 | idle×18       |
| S10_F05        |       10 | F05      | train          | curated_clip      | idle         |  42 | 0.619 | raise_hand×14 |
| S02_F01        |        2 | F01      | train          | curated_clip      | thumbs_up    |  54 | 0.63  | beckoning×18  |
| S22_F01        |       22 | F01      | test           | raw_take_20260725 | wave         |  48 | 0.688 | raise_hand×13 |
| S01_F01        |        1 | F01      | train          | curated_clip      | wave         |  42 | 0.714 | raise_hand×12 |
| S13_F06        |       13 | F06      | train          | raw_take_20260725 | wave         |  48 | 0.729 | raise_hand×12 |

Coverage: 1 clip(s) never observed; per-view observation {'phone_1080p': 1.0, 'phone_4k': 1.0, 'realsense_480p': 0.998}

## 5.3 motion detail

**Per class (clip-level)**

|               |   precision |   recall |    f1 |   support |
|:--------------|------------:|---------:|------:|----------:|
| sitting       |       0.991 |    0.838 | 0.908 |       629 |
| standing      |       0.385 |    0.461 | 0.42  |       232 |
| walking       |       0.635 |    0.822 | 0.716 |       427 |
| stepping_back |       0.763 |    0.384 | 0.511 |       151 |

**Confusion (rows = truth)**

|                    |   sitting |   standing |   walking |   stepping_back |
|:-------------------|----------:|-----------:|----------:|----------------:|
| true_sitting       |       527 |         57 |        45 |               0 |
| true_standing      |         5 |        107 |       112 |               8 |
| true_walking       |         0 |         66 |       351 |              10 |
| true_stepping_back |         0 |         48 |        45 |              58 |

**Worst scenarios**

| scenario_dir   |   v3_row | intent   | split_design   | source            | gt_motion     |   n |   acc | top_error   |
|:---------------|---------:|:---------|:---------------|:------------------|:--------------|----:|------:|:------------|
| S04_F02        |        4 | F02      | train          | raw_take_20260725 | stepping_back |  48 | 0     | standing×48 |
| S26_F05        |       26 | F05      | test           | raw_take_20260725 | sitting       |  48 | 0.021 | standing×46 |
| S13_F06        |       13 | F06      | train          | raw_take_20260725 | standing      |  48 | 0.042 | walking×41  |
| S25_F04        |       25 | F04      | test           | raw_take_20260725 | stepping_back |  48 | 0.062 | walking×45  |
| S31_F10        |       31 | F10      | test           | raw_take_20260725 | standing      |  48 | 0.167 | walking×40  |
| S21_F10        |       21 | F10      | train          | raw_take_20260725 | standing      |  48 | 0.188 | walking×31  |
| S12_F06        |       12 | F06      | train          | curated_clip      | walking       |  54 | 0.278 | standing×38 |
| S20_F10        |       20 | F10      | train          | raw_take_20260725 | sitting       |  48 | 0.646 | walking×13  |
| S15_F07        |       15 | F07      | train          | raw_take_20260725 | sitting       |  48 | 0.75  | walking×10  |
| S22_F01        |       22 | F01      | test           | raw_take_20260725 | walking       |  48 | 0.812 | standing×8  |

Coverage: 1 clip(s) never observed; per-view observation {'phone_1080p': 1.0, 'phone_4k': 1.0, 'realsense_480p': 0.998}

## 5.4 context detail

**Per class (clip-level)**

|             |   precision |   recall |   f1 |   support |
|:------------|------------:|---------:|-----:|----------:|
| classroom   |           1 |    0.999 |    1 |      1391 |
| kitchen     |           0 |    0     |    0 |         0 |
| hospital    |           0 |    0     |    0 |         0 |
| cloth_store |           0 |    0     |    0 |         0 |
| museum      |           0 |    0     |    0 |         0 |

**Confusion (rows = truth)**

|                  |   classroom |   kitchen |   hospital |   cloth_store |   museum |
|:-----------------|------------:|----------:|-----------:|--------------:|---------:|
| true_classroom   |        1390 |         0 |          1 |             0 |        0 |
| true_kitchen     |           0 |         0 |          0 |             0 |        0 |
| true_hospital    |           0 |         0 |          0 |             0 |        0 |
| true_cloth_store |           0 |         0 |          0 |             0 |        0 |
| true_museum      |           0 |         0 |          0 |             0 |        0 |

**Worst scenarios**

| scenario_dir   |   v3_row | intent   | split_design   | source            | gt_context   |   n |   acc | top_error   |
|:---------------|---------:|:---------|:---------------|:------------------|:-------------|----:|------:|:------------|
| S23_F02        |       23 | F02      | test           | raw_take_20260725 | classroom    |  40 | 0.975 | hospital×1  |
| S01_F01        |        1 | F01      | train          | curated_clip      | classroom    |  42 | 1     |             |
| S03_F02        |        3 | F02      | train          | curated_clip      | classroom    |  40 | 1     |             |
| S02_F01        |        2 | F01      | train          | curated_clip      | classroom    |  54 | 1     |             |
| S05_F03        |        5 | F03      | train          | curated_clip      | classroom    |  59 | 1     |             |
| S07_F04        |        7 | F04      | train          | curated_clip      | classroom    |  70 | 1     |             |
| S08_F04        |        8 | F04      | train          | curated_clip      | classroom    |  61 | 1     |             |
| S04_F02        |        4 | F02      | train          | raw_take_20260725 | classroom    |  48 | 1     |             |
| S11_F05        |       11 | F05      | train          | curated_clip      | classroom    |  54 | 1     |             |
| S12_F06        |       12 | F06      | train          | curated_clip      | classroom    |  54 | 1     |             |

Coverage: 0 clip(s) never observed; per-view observation {'phone_1080p': 1.0, 'phone_4k': 1.0, 'realsense_480p': 1.0}