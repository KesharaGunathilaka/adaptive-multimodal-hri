# Recombination pool purity — are the synthetic cues realistic?

Generated 2026-08-08 06:10 · `data/final_merged`.

**Purity** = the fraction of a ground-truth bucket's per-clip vectors that the cue model actually argmaxes to that bucket's own class. Recombination draws from the **train** column; deployment sees the **test** column. A gap between them is distribution shift injected straight into fusion's training data.

## Overall

| Cue | Fine-tuned on train? | Train | Val | Test | Train − Test |
|---|---|---|---|---|---|
| gesture | **yes** | 1.000 | 0.997 | 0.864 | **+0.136** |
| emotion | **yes** | 0.959 | 0.752 | 0.762 | **+0.197** |
| motion | no | 0.655 | 0.671 | 0.671 | **-0.016** |

The pattern lines up exactly with which models were promoted on 2026-08-04: **emotion and gesture were fine-tuned on the train split, and their pools are correspondingly unrealistic; motion was not promoted (its fine-tune regressed 4/4 times) and its pool is the only honest one.** Gesture's train purity is a perfect 1.000 — every single training clip is classified correctly — against 0.864 at test.

## Per class

### gesture

| Class | Train | Val | Test |
|---|---|---|---|
| idle | 1.000 | 1.000 | 0.850 |
| wave | 1.000 | 1.000 | 1.000 |
| point | 1.000 | 1.000 | 0.779 |
| thumbs_up | 1.000 | — | 0.845 |
| thumbs_down | 1.000 | 0.982 | 0.966 |
| beckoning | 1.000 | 1.000 | 0.954 |
| raise_hand | 1.000 | 1.000 | 0.327 |
| both_hands_up | 1.000 | 1.000 | 0.981 |

### emotion

| Class | Train | Val | Test |
|---|---|---|---|
| Surprise | 1.000 | 1.000 | — |
| Fear | 1.000 | 1.000 | 0.727 |
| Disgust | 0.961 | 0.365 | 0.811 |
| Happy | 0.979 | 0.913 | 0.729 |
| Sad | 0.985 | 0.617 | 0.729 |
| Anger | 0.989 | 0.983 | 0.953 |
| Neutral | 0.903 | 0.795 | 0.632 |

### motion

| Class | Train | Val | Test |
|---|---|---|---|
| sitting | 0.899 | 0.848 | 0.708 |
| standing | 0.427 | 0.474 | 0.648 |
| walking | 0.746 | 0.831 | 0.837 |
| stepping_back | 0.219 | 0.583 | 0.164 |

## Why this matters

1. **Fusion is trained to over-trust gesture and emotion.** During training those cues are essentially never wrong; at deployment they are wrong 13.6% and 23.8% of the time. This independently predicts the cue-attribution result (`CUE_ATTRIBUTION.md`: gesture 31.8% > emotion 29.5% > motion 17.6% > context 10.1%) — the model weights the cues in precisely the order of how clean their training pools were, not how reliable they actually are.
2. **It explains the oracle/real gap.** `PERCEPTION_BAND.md` finds fusion at 0.9373 on clean cues but 0.7133 on real ones. A model trained almost exclusively on clean cues is expected to be brittle to realistic noise.
3. **`raise_hand` is the worst case**: train purity 1.000, test 0.327. Fusion has never once seen a mistaken `raise_hand` vector in training. Row #25 (the only `raise_hand` test row, and the entire flip subset of T04 — `CONTEXT_COUNTERFACTUAL.md`) scores 0.31/0.19 for fusion/rules.

## What to do about it

- **Do not simply switch the pools to val.** Val is actor-disjoint but shares scenarios with train, so gesture purity there is still 0.997 — it measures actor shift, not the scenario shift the test split measures. Val pools would fix emotion (0.959 → 0.752) but barely touch gesture.
- **The sound fix is out-of-fold prediction**: refit each cue model K times, holding out a different scenario group each time, and pool the held-out predictions. That yields pool vectors carrying the error rate the model has on unseen scenarios, which is what deployment sees. Cost is K unimodal fine-tunes, not a re-extraction.
- **A cheaper approximation** is calibrated noise injection: perturb pool vectors until bucket purity matches a held-out estimate of test-time purity. `WindowDataset`'s `jitter_sigma` already does something in this spirit, but it is class-agnostic and not calibrated to per-class confusion.
- Either way this is a **recombination-design** issue, not a fusion-architecture one, and it sits squarely in the perception band that `PERCEPTION_BAND.md` identifies as holding all the remaining headroom.