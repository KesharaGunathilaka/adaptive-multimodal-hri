# Baseline results (features_v1, window-level & clip-level)

val/test are actor-disjoint subject splits of the 22 recorded TRAIN scenarios; the V3 test scenarios are not recorded yet. F10 has no supervised rows (S28 pooled).

| model | test win acc | test win F1 | test clip acc | test clip F1 |
|---|---|---|---|---|
| rule_based | 0.666 | 0.480 | 0.695 | 0.452 |
| unimodal_emotion | 0.638 | 0.371 | 0.732 | 0.436 |
| unimodal_gesture | 0.420 | 0.238 | 0.427 | 0.227 |
| unimodal_motion | 0.473 | 0.222 | 0.512 | 0.234 |
| unimodal_context | 0.130 | 0.136 | 0.146 | 0.150 |
| concat_mlp (3 seeds) | 0.934±0.004 | 0.629±0.005 | 0.931±0.011 | 0.621±0.011 |
