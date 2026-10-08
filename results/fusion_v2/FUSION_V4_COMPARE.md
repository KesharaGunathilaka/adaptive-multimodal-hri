# Phase 3+4 — v4 fusion: representation comparison

Taxonomy v4 labels; self-attention fusion; exhaustive v4 cue-combination augmentation (66 of 80 combinations, every TEST combination excluded). Recorded test n=773, 3 seeds.

**v4 rules on measured cues: 0.7193**

| representation | test acc | synthetic rows |
|---|---|---|
| softmax_super dp0.4 maskedsel | 0.6516 ± 0.0132 | 19800 |
| softmax_super dp0.4 plainsel | 0.6529 ± 0.0062 | 19800 |
| softmax_super dp0.1 plainsel | 0.6580 ± 0.0064 | 19800 |
| softmax_super dp0.0 plainsel | 0.6598 ± 0.0059 | 19800 |
| embed dp0.4 maskedsel | 0.6028 ± 0.0417 | 19800 |
| embed dp0.4 plainsel | 0.6330 ± 0.0096 | 19800 |
| embed dp0.1 plainsel | 0.6464 ± 0.0032 | 19800 |
| embed dp0.0 plainsel | 0.6464 ± 0.0049 | 19800 |
| two_stage dp0.4 maskedsel | 0.6235 ± 0.0412 | 19800 |
| two_stage dp0.4 plainsel | 0.6507 ± 0.0021 | 19800 |
| two_stage dp0.1 plainsel | 0.6567 ± 0.0065 | 19800 |
| two_stage dp0.0 plainsel | 0.6498 ± 0.0086 | 19800 |

**Best: softmax_super dp0.0 plainsel — 0.6598 ± 0.0059**
