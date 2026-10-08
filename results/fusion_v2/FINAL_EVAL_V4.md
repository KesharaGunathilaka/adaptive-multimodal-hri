# Phase 5 - v4 fusion, full T01-T05 evaluation

softmax-super representation (emotion superclasses / gesture / context), no motion token, cue-dropout 0.0, plain val selection, 10 seeds. Recorded test n=773.

## Augmentation protocol (reported both ways)

The only conflict family in the test set is `hostile+thumbs_up` (146 of 773 clips) and it appears NOWHERE in the recorded training data. Excluding test cue-combinations from augmentation therefore makes T02 a test of transfer to a conflict combination never seen in any form. Enumerating the full authored rubric is how a deployed system is actually built - no test clip, scenario or actor is used, only the combination is enumerated and labelled by the rubric, and materialised from real TRAIN cue vectors.

| protocol | combos | synth rows | per-seed acc | ensemble | T02 conflict |
|---|---|---|---|---|---|
| `strict` | 66 | 19800 | 0.6577 +/- 0.0076 | 0.6624 | 0.1027 |
| `full_rubric` | 80 | 24000 | 0.8142 +/- 0.0073 | 0.8215 | 0.9384 |

v4 rules on the same clips: **0.7646** (T02 conflict 0.8973).

T03 has no rows as a *slice*: clips with a designed-missing cue cannot receive a v4 ground-truth label (the rubric needs that cue), so missing-cue behaviour is measured by the masking sweep in each protocol below. T04's context-dependent branch (`raise_hand`) has no test clips, so context generalization is reported as the per-context breakdown.

---

## `strict` protocol - detail

Augmentation over 66 combinations (19800 synthetic rows); 10 seeds. Per-seed **0.6577 +/- 0.0076**; ensemble **0.6624**; v4 rules **0.7646**.

| slice | n | fusion | rules | delta |
|---|---|---|---|---|
| **Overall (T05 - rules comparison)** | 773 | 0.6624 | 0.7646 | -0.1022 |
| T01 full-cue fusion | 773 | 0.6624 | 0.7646 | -0.1022 |
| T02 cue conflict | 146 | 0.1027 | 0.8973 | -0.7945 |
| T04 context-dependent rows | 0 | - | - | - |
| - classroom | 309 | 0.5761 | 0.7994 | -0.2233 |
| - kitchen | 377 | 0.7056 | 0.7798 | -0.0743 |
| Person-disjoint (unseen actor) | 165 | 0.6364 | 0.8000 | -0.1636 |
| Person-overlap | 608 | 0.6694 | 0.7549 | -0.0855 |

**T03 - masking sweep** (each cue forced missing at eval)

| masked cue | fusion | rules |
|---|---|---|
| emotion | 0.4334 | 0.4308 |
| gesture | 0.2561 | 0.4062 |
| context | 0.6624 | 0.7646 |

**Per-class recall**

| intent | n | fusion | rules |
|---|---|---|---|
| F01 | 146 | 0.856 | 0.664 |
| F02 | 38 | 0.947 | 0.974 |
| F03 | 95 | 0.853 | 0.684 |
| F04 | 80 | 0.762 | 0.725 |
| F05 | 80 | 0.750 | 0.662 |
| F06 | 53 | 0.736 | 0.755 |
| F07 | 146 | 0.103 | 0.897 |
| F08 | 41 | 0.732 | 0.780 |
| F10 | 94 | 0.691 | 0.830 |

McNemar vs rules: fusion-only-correct **57**, rules-only-correct **136**, p = **1.246e-08** (significant).

---

## `full_rubric` protocol - detail

Augmentation over 80 combinations (24000 synthetic rows); 10 seeds. Per-seed **0.8142 +/- 0.0073**; ensemble **0.8215**; v4 rules **0.7646**.

| slice | n | fusion | rules | delta |
|---|---|---|---|---|
| **Overall (T05 - rules comparison)** | 773 | 0.8215 | 0.7646 | +0.0569 |
| T01 full-cue fusion | 773 | 0.8215 | 0.7646 | +0.0569 |
| T02 cue conflict | 146 | 0.9384 | 0.8973 | +0.0411 |
| T04 context-dependent rows | 0 | - | - | - |
| - classroom | 309 | 0.8738 | 0.7994 | +0.0744 |
| - kitchen | 377 | 0.7931 | 0.7798 | +0.0133 |
| Person-disjoint (unseen actor) | 165 | 0.8121 | 0.8000 | +0.0121 |
| Person-overlap | 608 | 0.8240 | 0.7549 | +0.0691 |

**T03 - masking sweep** (each cue forced missing at eval)

| masked cue | fusion | rules |
|---|---|---|
| emotion | 0.4398 | 0.4308 |
| gesture | 0.4127 | 0.4062 |
| context | 0.8215 | 0.7646 |

**Per-class recall**

| intent | n | fusion | rules |
|---|---|---|---|
| F01 | 146 | 0.808 | 0.664 |
| F02 | 38 | 0.947 | 0.974 |
| F03 | 95 | 0.832 | 0.684 |
| F04 | 80 | 0.762 | 0.725 |
| F05 | 80 | 0.750 | 0.662 |
| F06 | 53 | 0.792 | 0.755 |
| F07 | 146 | 0.938 | 0.897 |
| F08 | 41 | 0.780 | 0.780 |
| F10 | 94 | 0.745 | 0.830 |

McNemar vs rules: fusion-only-correct **58**, rules-only-correct **14**, p = **1.646e-07** (significant).

---
