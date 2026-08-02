# Experiment ledger

23 runs · exported from `mlruns.db` by `scripts/25_export_runs.py`.

`mlruns.db` is per-machine and gitignored; **this file is the cross-machine record**. Full parameter/metric set in `EXPERIMENTS.csv`.

## 01_baselines

| experiment   | run_name              | dataset   | split_kind   | cues   | model            | aggregation   |   test_clip_acc |   test_clip_macro_f1 |   val_clip_acc |
|:-------------|:----------------------|:----------|:-------------|:-------|:-----------------|:--------------|----------------:|---------------------:|---------------:|
| 01_baselines | old__rule_based       | old       | people       | real   | rule_based       | window_vote   |          0.6951 |               0.4521 |         0.7875 |
| 01_baselines | old__unimodal_emotion | old       | people       | real   | unimodal_emotion | window_vote   |          0.7317 |               0.4364 |         0.425  |
| 01_baselines | old__unimodal_gesture | old       | people       | real   | unimodal_gesture | window_vote   |          0.4268 |               0.2269 |         0.75   |
| 01_baselines | old__unimodal_motion  | old       | people       | real   | unimodal_motion  | window_vote   |          0.5122 |               0.2337 |         0.4125 |
| 01_baselines | old__unimodal_context | old       | people       | real   | unimodal_context | window_vote   |          0.1463 |               0.1502 |         0.3    |
| 01_baselines | old__concat_mlp       | old       | people       | real   | concat_mlp       | window_vote   |          0.9309 |               0.6214 |         0.9417 |

## 02_fusion

| experiment   | run_name         | dataset   | split_kind   | cues   | model            | aggregation   |   test_clip_acc |   test_clip_macro_f1 |   val_clip_acc |
|:-------------|:-----------------|:----------|:-------------|:-------|:-----------------|:--------------|----------------:|---------------------:|---------------:|
| 02_fusion    | old__attn_base   | old       | people       | real   | attention_fusion | window_vote   |          0.9431 |               0.6279 |         0.9625 |
| 02_fusion    | old__attn_do     | old       | people       | real   | attention_fusion | window_vote   |          0.9431 |               0.6291 |         0.9542 |
| 02_fusion    | old__attn_do_jit | old       | people       | real   | attention_fusion | window_vote   |          0.9512 |               0.6293 |         0.9625 |
| 02_fusion    | old__attn_full   | old       | people       | real   | attention_fusion | window_vote   |          0.9431 |               0.6493 |         0.9417 |
| 02_fusion    | old__attn_robust | old       | people       | real   | attention_fusion | window_vote   |          0.9431 |               0.6496 |         0.9333 |

## 03_diagnostics

| experiment     | run_name                               | dataset   | split_kind   | cues   | model            | aggregation            |   test_clip_acc |   test_clip_macro_f1 |   val_clip_acc |
|:---------------|:---------------------------------------|:----------|:-------------|:-------|:-----------------|:-----------------------|----------------:|---------------------:|---------------:|
| 03_diagnostics | old__masking_sweep                     | old       | people       | real   | attention_vs_mlp | window_vote            |         nan     |              nan     |            nan |
| 03_diagnostics | old__window_sweep                      | old       | people       | real   | attention_fusion | window                 |         nan     |              nan     |            nan |
| 03_diagnostics | classroom__rules_oracle                | final     | scenarios    | oracle | rule_based       | clip_mean_4s           |           0.9   |              nan     |            nan |
| 03_diagnostics | classroom__fusion_oracle               | final     | scenarios    | oracle | attention_fusion | clip_mean_4s           |           0.5   |              nan     |            nan |
| 03_diagnostics | classroom__rules_real                  | final     | scenarios    | real   | rule_based       | clip_mean_4s           |           0.494 |                0.473 |            nan |
| 03_diagnostics | classroom__fusion_real                 | final     | scenarios    | real   | attention_fusion | clip_mean_4s           |           0.325 |                0.242 |            nan |
| 03_diagnostics | clipvswin__train_window_vote           | final     | scenarios    | real   | attention_fusion | window_vote            |           0.342 |                0.284 |            nan |
| 03_diagnostics | clipvswin__train_window_infer_clipmean | final     | scenarios    | real   | attention_fusion | window_train_clip_mean |           0.331 |                0.269 |            nan |
| 03_diagnostics | clipvswin__train_window_infer_clipmax  | final     | scenarios    | real   | attention_fusion | window_train_clip_max  |           0.309 |                0.246 |            nan |
| 03_diagnostics | clipvswin__clip_mean                   | final     | scenarios    | real   | attention_fusion | clip_mean              |           0.374 |                0.32  |            nan |
| 03_diagnostics | clipvswin__clip_max                    | final     | scenarios    | real   | attention_fusion | clip_max               |           0.338 |                0.289 |            nan |
| 03_diagnostics | clipvswin__clip_peak                   | final     | scenarios    | real   | attention_fusion | clip_peak              |           0.373 |                0.308 |            nan |
