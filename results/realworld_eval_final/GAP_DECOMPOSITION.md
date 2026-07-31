# Where the 0.37-vs-0.90 gap actually goes (classroom test rows)

**Date:** 2026-07-28 · **Host:** WIN-3060 · **Data:** `data/final`, classroom,
RealSense 480p, 868 clips / 29 recorded V3 rows. Fusion trained on the
**train-design rows only**, evaluated on the **10 test-design rows** (160 clips).
Aggregation: 4 s mean-pooled windows, stride 1 s (see §"protocol").

This supersedes the priority ordering in `ASSESSMENT.md` §5. The measurement
that made the difference is an **oracle test**: run the *same* fusion
architecture on the V3 table's **true cue labels** (one-hot) instead of the
perception models' predictions. That separates "the cues are bad" from "the
fusion model cannot generalise".

---

## 1. The decomposition

| Configuration | Test clip acc | What it isolates |
|---|---|---|
| **Ceiling** (label ambiguity) | **0.900** | rows #22/#30 are the identical tuple; direction is unobservable |
| **Rule-based + oracle cues** | **0.900** | the rubric is fully expressible — hits the ceiling exactly |
| **Learned fusion + oracle cues** | **0.500** ±0.000 | ← **loses 0.40 here** |
| **Learned fusion + real cues** | **0.325** ±0.018 | ← loses a further 0.175 to perception |

```
0.90  ceiling ─────────────────────────────────────────── rules+oracle = 0.90 ✔
  │
  │   0.40   FUSION GENERALISATION COST   ← dominant, and previously unmeasured
  ▼
0.50  fusion+oracle
  │
  │   0.175  PERCEPTION COST
  ▼
0.325 fusion+real
```

**The dominant term is not perception.** Even with perfect cues the learned
fusion head reaches only 0.50, while a hand-written rubric on the same perfect
cues reaches 0.90. Fixing emotion and motion (ASSESSMENT §5 items 2–3) can
recover at most the 0.175 band.

## 2. Why the learned model loses 0.40

The test rows are, *by design*, cue combinations that never appear in training
(that is the G3 claim). The fusion head sees roughly **19 distinct cue tuples**
during training and is then asked to classify **10 unseen tuples**. A network
trained on 19 tuples memorises tuple→intent; it does not infer the compositional
rule behind them. This is a systematicity/compositional-generalisation failure,
not a capacity or optimisation problem — note the oracle run has **±0.000 seed
variance**, i.e. all three seeds converge to the same wrong answers.

The per-row table shows it plainly (mean of 3 seeds):

| V3 row | Cue tuple (abridged) | Intent | oracle | real |
|---|---|---|---|---|
| #26 | neutral, both_hands_up, sit | F05 | **0.00** | 0.00 |
| #23 | fear, [missing] gesture, walk | F02 | **0.00** | 0.00 |
| #27 | angry, point, walk | F06 | 0.33 | 0.00 |
| #22 | [missing] emotion, wave, walk | F01 | 0.33 | 0.00 |
| #25 | [missing] emotion, raise_hand, step back | F04 | 0.67 | 0.00 |
| #24 | neutral, beckoning, walk | F03 | 1.00 | 0.02 |
| #31 | sad, none, stand | F10 | 1.00 | 0.77 |

- **#26 and #23 fail even with perfect cues.** Every training row with
  `both_hands_up` is alarmed or standing (F02/F07), so the model answers F02 for
  the seated neutral stretch. The rubric says otherwise; the training tuples do
  not teach it. Row #23 is the same story for fear.
- **#24 is the opposite failure**: perfect with oracle cues (1.00), destroyed by
  perception (0.02). That row *is* a perception problem.

So the two costs hit **different rows**, which is why a single headline number
hid the structure.

## 3. What this changes

`ASSESSMENT.md` §5 predicted "retrain fusion on `data/final` → 0.388 → high-0.8s".
Retraining was done here and the result is **0.325–0.374**, not high-0.8s. The
prediction assumed the 18 unseen rows were unseen *by accident*; 10 of them are
test rows and are unseen **by design**, so retraining cannot make them seen
without leakage.

**Revised priority:**

| # | Action | Targets | Expected |
|---|---|---|---|
| 1 | **Rubric-driven cue recombination** — synthesise training samples spanning the *combinatorial cue space* using V3 §2.6, with real cue vectors as the source of noise | the 0.40 band | the only lever that addresses compositional generalisation |
| 2 | Fine-tune emotion (Fear/Disgust/Sad) and motion (`stepping_back`, `standing`) on the 2026-07-25 takes | the 0.175 band, rows #24/#25 | recovers perception cost |
| 3 | Direction cue as a 5th token (`pose_img` already cached) | the 0.10 band | ceiling 0.90 → 1.00; fixes #22 vs #30 |
| 4 | Collision-aware dropout, confidence-thresholded masking | label noise, train/deploy mismatch | correctness, not headline |

Item 1 first — it is both the largest term and a prerequisite for a meaningful
error analysis of items 2–3.

## 4. The honesty question item 1 raises

If fusion is trained on rubric-generated samples, it is being **taught the
rubric** rather than discovering it from data. The thesis claim must be stated
accordingly. The defensible framing, supported by the numbers we already have:

- **Rules + perfect cues = 0.90** (ceiling) but **rules + real cues** are brittle
  — on `data/old` the rule baseline scored 0.695 while learned fusion scored
  0.951 on the same real cues.
- Therefore the learned model's contribution is **robustness to noisy, uncertain
  and missing cues**, not semantic discovery.
- Rubric-driven augmentation gives the model the *semantics* it cannot induce
  from 19 tuples, and training on **real (noisy) cue vectors** gives it the
  *robustness* rules lack.

That is a coherent and honest contribution — "rule knowledge + learned noise
robustness beats either alone" — and it is directly testable: compare
rules-on-real-cues vs fusion-on-real-cues vs fusion-with-rubric-augmentation, all
on the same test rows.

## 5. Protocol

- Aggregation: per clip, 4 s windows with 1 s stride, **mean-pooled** cue
  probabilities (the deployment-matched representation chosen 2026-07-28; see
  the clip-vs-window study below).
- Split: V3 `split_design`; val = 20 % of *train takes*, grouped by
  `(scenario_dir, take_index)` so the 3 simultaneous camera views of one take
  never straddle splits.
- 3 seeds, `AttentionFusion(missing_mode="exclude")`, clip-level majority vote.
- Oracle cues: one-hot of `emotion_v3` / `gesture_v3` / `motion_v3` from the
  table; designed-missing cues left as `obs=0`.

### Companion result — clip-level vs window-level (2026-07-28)

Same data, same splits, 3 seeds, evaluated on the same 475 test clips:

| Approach | clip acc | macro-F1 |
|---|---|---|
| train=window, infer=majority vote | 0.342 ±0.028 | 0.284 |
| train=window, infer=clip mean | 0.331 ±0.022 | 0.269 |
| train=window, infer=clip max | 0.309 ±0.013 | 0.246 |
| **train=clip mean, infer=clip mean** | **0.374 ±0.030** | **0.320** |
| train=clip max | 0.338 ±0.032 | 0.289 |
| train=clip peak | 0.373 ±0.021 | 0.308 |

And the per-cue reading of the intended label:

| Cue | per-window | clip mean | clip max | clip vote |
|---|---|---|---|---|
| emotion | 0.771 | 0.802 | 0.803 | 0.803 |
| gesture | 0.788 | 0.797 | 0.805 | 0.796 |
| motion | 0.719 | 0.725 | 0.733 | 0.725 |

Conclusions: aggregating over a clip **does** read cues better (+1.4 to +3.2
points) and clip-level training is the better of the two (+3.2 accuracy,
+3.6 macro-F1), but **never mix** train-on-windows with infer-on-clips (worse
than either consistent choice). Mean ≈ peak > max; mean is used. Clip-level
training uses 784 samples instead of 10,946 and still wins, indicating the extra
windows were largely redundant. Both pipelines are retained for comparison.
