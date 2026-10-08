# T03 missing-cue robustness + headline fusion vs rules

v4 taxonomy, softmax-super representation, full-rubric augmentation, 5 seeds (majority vote), recorded test n=773.

## Training-time cue dropout sweep

The Phase-5 model used dropout 0.0 and therefore never learned to run without a cue. `sel` is the checkpoint-selection criterion: *plain* = full-cue val accuracy, *masked* = mean of full-cue and each single-cue-masked val accuracy.

| dropout | sel | full-cue | emotion masked | gesture masked | context masked |
|---|---|---|---|---|---|
| 0.0 | plain | 0.8189 | 0.4360 | 0.4450 | 0.8163 |
| 0.3 | plain | 0.8215 | 0.4269 | 0.3467 | 0.8189 |
| 0.3 (real-only) | plain | 0.8176 | 0.4709 | 0.3648 | 0.8215 |
| 0.5 (real-only) | plain | 0.8241 | 0.4761 | 0.3583 | 0.8241 |
| 0.7 (real-only) | plain | 0.8215 | 0.4748 | 0.4437 | 0.8241 |
| 0.5 (real-only) | masked | 0.8137 | 0.4748 | 0.3169 | 0.8124 |

| **rules** | - | **0.7646** | 0.4308 | 0.4062 | 0.7646 |
| **ceiling** | - | 1.0000 | 0.7633 | 0.7128 | 1.0000 |

The ceiling is the best any predictor can reach with that cue hidden (majority intent per remaining ground-truth cue signature): with emotion hidden, `thumbs_up` is F01 or F07 and nothing else separates them.

## Recommended configuration

dropout **0.7** applied to recorded rows only, selection **plain**

| condition | fusion | rules | ceiling | fusion as % of ceiling |
|---|---|---|---|---|
| full-cue (T01) | **0.8215** | 0.7646 | 1.0000 | 82.1% |
| emotion masked (T03) | **0.4748** | 0.4308 | 0.7633 | 62.2% |
| gesture masked (T03) | **0.4437** | 0.4062 | 0.7128 | 62.3% |
| context masked (T03) | **0.8241** | 0.7646 | 1.0000 | 82.4% |

## Headline: fusion vs rule-based

| metric | fusion | rules | delta |
|---|---|---|---|
| Overall accuracy (T01/T05) | **0.8215** | 0.7646 | +0.0569 |
| T03 emotion lost | **0.4748** | 0.4308 | +0.0440 |
| T03 gesture lost | **0.4437** | 0.4062 | +0.0375 |
| T03 context lost | **0.8241** | 0.7646 | +0.0595 |
| **mean over all four conditions** | **0.6410** | 0.5915 | +0.0495 |
