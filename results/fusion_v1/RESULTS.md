# Attention fusion results (features_v1, actor-disjoint test)

## Ablations (3 seeds, test clip-level)

| config | clip acc | clip macro-F1 |
|---|---|---|
| attn_base | 0.943�0.011 | 0.628�0.033 |
| attn_do | 0.943�0.025 | 0.629�0.040 |
| attn_do_jit | 0.951�0.017 | 0.629�0.038 |
| attn_full | 0.943�0.006 | 0.649�0.005 |
| attn_robust | 0.943�0.021 | 0.650�0.015 |

## T03 masking sweep (test clip acc, attention=attn_robust best seed)

| masked | attention | concat-MLP |
|---|---|---|
| none | 0.951 | 0.915 |
| emotion | 0.671 | 0.744 |
| gesture | 0.695 | 0.878 |
| motion | 0.793 | 0.805 |
| context | 0.780 | 0.780 |
| emotion+gesture | 0.305 | 0.439 |
| emotion+motion | 0.451 | 0.549 |
| emotion+context | 0.537 | 0.622 |
| gesture+motion | 0.561 | 0.537 |
| gesture+context | 0.732 | 0.817 |
| motion+context | 0.732 | 0.658 |

## Robustness follow-up (3 seeds, test clip acc)

attn_exclude = attention, missing cues excluded via key-padding mask;
mlp_do = concat-MLP trained with the same dropout/jitter/recombination.

| masked | attn_exclude | mlp_do |
|---|---|---|
| none | 0.939±0.000 | 0.927±0.020 |
| emotion | 0.654±0.029 | 0.658±0.026 |
| gesture | 0.720±0.060 | 0.805±0.053 |
| motion | 0.809±0.032 | 0.833±0.015 |
| context | 0.776±0.006 | 0.776±0.006 |
| emotion+gesture | 0.297±0.064 | 0.297±0.047 |
| emotion+motion | 0.476±0.065 | 0.504±0.055 |
| emotion+context | 0.537±0.000 | 0.545±0.006 |
| gesture+motion | 0.618±0.051 | 0.683±0.053 |
| gesture+context | 0.752±0.038 | 0.732±0.030 |
| motion+context | 0.740±0.006 | 0.720±0.000 |
