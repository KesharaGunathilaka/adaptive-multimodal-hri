# Stage 7 — Evaluation protocol

**Goal of this stage:** define what counts as evidence, so that every number in
the thesis has a stated question behind it. Most of the project's hard-won
lessons are in this document — including two places where an earlier, more
flattering protocol was replaced by a harsher and more honest one.

---

## 7.1 The three questions any number must answer

Before quoting any accuracy figure in this project, state:

| Axis | Options | Why it matters |
|---|---|---|
| **Which dataset** | `data/old` (22 scenarios) / `data/final` (all V3 rows) | different label coverage |
| **What is held out** | **people** / **scenarios (cue tuples)** | changes the difficulty enormously |
| **Which cues** | **real** (model outputs) / **oracle** (table labels) | oracle is a diagnostic, never a deployable result |

A single model scored **0.951** and **0.325** on these axes. Same weights, same
code — different question. See `05_baselines.md` §5.0 for the unified table.

---

## 7.2 The unit of prediction: clip-level is the headline

Window-level accuracy is **not** the headline metric, for a structural reason:
windows within one clip overlap in time and show the same person doing the same
thing, so they are **highly correlated**. Treating 15 windows as 15 independent
samples flatters the model.

Therefore:
- **Clip-level accuracy** (aggregate a clip's windows into one prediction) is the
  reported metric.
- **Window-level accuracy** is secondary and slightly optimistic.

How to aggregate was itself measured (2026-07-28, `GAP_DECOMPOSITION.md` §5):

| Approach | Test clip acc | macro-F1 |
|---|---|---|
| train=window, infer=majority vote | 0.342 ± 0.028 | 0.284 |
| train=window, infer=clip mean | 0.331 ± 0.022 | 0.269 |
| **train=clip mean, infer=clip mean** | **0.374 ± 0.030** | **0.320** |
| train=clip max | 0.338 ± 0.032 | 0.289 |
| train=clip peak | 0.373 ± 0.021 | 0.308 |

Three rules follow, and they are now project policy:
1. **Aggregate over ~4 s, mean-pooled.** Matches the deployment buffer and beats
   window-level training.
2. **Never mix** — training on windows and inferring on clips is worse than either
   consistent choice. Train/serve aggregation must match.
3. **Both pipelines are retained** for comparison (user decision, 2026-07-28).

Aggregating also measurably improves how the *cues* are read (emotion
0.771→0.802, gesture 0.788→0.805, motion 0.719→0.733), which is the perception
half of the same effect.

---

## 7.3 Metrics: accuracy and macro-F1, always both

- **Accuracy** — fraction of clips correct. Easy to read, but dominated by common
  classes.
- **Macro-F1** — the mean of per-class F1, treating all 10 intents equally. This
  is the metric that notices when a rare intent is never predicted.

Both are always reported because they disagree in an informative way. On
`data/old` the fusion model scored 0.951 accuracy but only 0.629 macro-F1 — the
gap was almost entirely **F10 having zero training rows** (its only scenario, S28,
was in the recombination pool), which capped macro-F1 at 0.9. Accuracy barely
noticed; macro-F1 exposed it. Cue recombination raised macro-F1 to 0.649 without
moving accuracy — the correct signature for that augmentation.

**Reporting standard:** headline numbers are mean ± std over **≥3 seeds**. A
single-seed number is not reported as a result.

---

## 7.4 The test cases T01–T05

Defined in the V3 table §2.4. Current measurability:

| Code | Test | Status |
|---|---|---|
| T01 | Full-cue fusion accuracy | ✅ measured |
| T02 | Cue-conflict resolution (emotion vs gesture) | ⚠️ now recorded in `data/final`; not yet reported |
| T03 | Missing-cue robustness | ✅ measured **as simulated masking** — see §7.5 |
| T04 | Context generalization (meaning flips by environment) | ⚠️ kitchen only 10/31 rows recorded |
| T05 | Rule-based baseline comparison | ✅ measured — and the result **reversed** on `data/final` (§5.3) |

---

## 7.5 T03 — the masking sweep, and an honesty problem

**The sweep:** evaluate the model with each modality force-masked, then each pair,
on the same test set. Results and the `[MISSING]`-token negative result are in
`06_fusion_model.md` §6.3.

Two findings worth carrying into the thesis:

1. **Masking emotion costs ~28 points** — by far the largest single drop,
   consistent with emotion being the most reliable cue (Stage 2).
2. **Always report a masking number against its ceiling.** The 2026-07-27
   ambiguity analysis (`scripts/17_cue_ambiguity.py`) computes what the *label
   table itself* permits under each masking. Under emotion masking the train-row
   ceiling is 0.753; a model at ~0.67 is close to optimal, not badly broken. A
   masking number without its ceiling is uninterpretable.

**The honesty problem** (`04_missing_cues.md` §4): the rows *designed* to have a
missing cue did not come out missing. On rows whose scenario says the face is
hidden behind a book, the emotion model still finds a face in **99.7 %** of
windows; on hands-occupied rows the gesture model fires **100 %** of the time,
because it only needs a pose. So:

- T03 as currently measured is **simulated (flag-driven) masking**, not real
  sensor failure. It remains a valid experiment, but the thesis **must not**
  describe it as real occlusion.
- There is a train/deploy mismatch: training masks these rows from the table, but
  on the Jetson nothing sets `obs=0` — the detector succeeds. Confidence-
  thresholded masking closes this (Stage 8 §8.6).

---

## 7.6 The ceiling — evaluation against what is possible

Fusion cannot beat its own label table. Where two rows share a cue tuple but
disagree on intent, the best possible policy answers the majority intent.
Classroom, weighted by clips on disk:

| Cue(s) masked | train ceiling | test ceiling |
|---|---|---|
| none | 0.977 | **0.900** |
| context | 0.977 | 0.900 |
| motion | 0.977 | 0.900 |
| emotion | 0.753 | 0.800 |
| gesture | 0.691 | 0.900 |
| emotion + gesture | 0.407 | 0.400 |

**Even with all four cues, the classroom test set caps at 0.90.** Rows #22 (F01
greeting) and #30 (F09 farewell) are the identical tuple — classroom, emotion
`[MISSING]`, wave, walk — differing only in walking *toward the robot* vs *toward
the door*. **Direction** is the disambiguator and no model outputs it. That is a
coin flip by construction, and it is the single cleanest motivation figure for
adding a fifth cue.

Verified independently: a hand-coded rule system given **oracle cues** scores
exactly **0.900** on test rows — hitting the ceiling to three decimals, which
both validates the ceiling computation and proves the rule baseline is not a
strawman.

---

## 7.7 The gap decomposition — the most important evaluation result

Running the *same* fusion architecture on oracle cues separates two failure modes
that a single accuracy number conflates:

```
0.900  ceiling ────────────────────── rules + oracle cues = 0.900 ✔
  │
  │  0.40   FUSION GENERALISATION COST   ← dominant
  ▼
0.500  fusion + oracle cues
  │
  │  0.175  PERCEPTION COST
  ▼
0.325  fusion + real cues
```

**The bottleneck is reasoning, not perception.** Even with perfect cues the
learned head reaches 0.500 while a hand-written rubric reaches 0.900. The oracle
run had **±0.000 seed variance** — all seeds converge on the same wrong answers,
so this is systematic.

This overturned the priority ordering in `results/realworld_eval_final/ASSESSMENT.md`
§5, which predicted that retraining fusion on `data/final` would reach "high-0.8s".
That prediction was tested and is wrong (0.325–0.374): the 10 test rows are unseen
**by design**, so retraining cannot make them seen without leakage.

Per-row, the two costs hit **different rows** — which is why one headline number
hid the structure:

| V3 row | oracle | real | Diagnosis |
|---|---|---|---|
| #26 (neutral, both_hands_up, sit → F05) | 0.00 | 0.00 | fails with perfect cues → **reasoning** |
| #23 (fear, missing gesture, walk → F02) | 0.00 | 0.00 | **reasoning** (safety-critical) |
| #24 (neutral, beckoning, walk → F03) | 1.00 | 0.02 | perfect with oracle → **perception** |

Row #26 is instructive: every training row containing `both_hands_up` is alarmed
or standing (F02/F07), so the model answers F02 for a seated neutral stretch. The
rubric says otherwise; the *training tuples* do not teach it.

---

## 7.8 G3 spotlight — the qualitative exhibit

Per-row predictions for same-gesture families where emotion or context flips the
meaning (`scripts/09_g3_spotlight.py` → `results/fusion_v1/G3_SPOTLIGHT.md`). On
`data/old`, **13 of 14 cases** resolved correctly — e.g. thumbs-down mapping to
F04/F07/F08/F01 across sad/angry/disgust/happy.

This is a *demonstration*, not a metric: small per-cell sample counts (some rows
have 2–3 test clips) mean it illustrates the mechanism rather than proving it.
Label it as such in the thesis.

---

## 7.9 The window-size sweep

Re-running Stage 3's Pass 2 at 0.5×/1×/2× lookback spans (cheap — no video
re-decode):

| Span scale | gesture / motion span | Test clip acc |
|---|---|---|
| ×0.5 | 1.07 s / 1.0 s | **0.976** |
| ×1.0 | 2.13 s / 2.0 s | 0.951 |
| ×2.0 | 4.27 s / 4.0 s | 0.756 |

**Shorter is better, and cheaper.** ×2 exceeds typical clip length and collapses.
The deployed model nevertheless keeps ×1 because those spans match the unimodal
engines' *validated* buffers; ×0.5 is the first thing to try if Jetson latency
needs headroom. Note this sweep varies each model's **input span** — a different
question from Stage 7.2's *output aggregation* window.

---

## 7.10 Safety evaluation — F02

Emergency detection is evaluated separately because a miss is a safety failure,
not an accuracy point (`scripts/11_policy_demo.py`):

| Metric | Value |
|---|---|
| Clip-level "emergency fired" | **22/22 (1.000)** |
| Window-level F02 recall | 0.968 |
| False-emergency rate on non-F02 windows | 0.013 |

Measured on `data/old` test subjects with the deployed policy (τ_emergency = 0.30).
**This must be re-measured on `data/final`** — the gap analysis shows row #23
(fear + motion blur → F02) currently scores 0.188, so the safety claim does not
transfer unexamined.

---

## 7.11 Reproducibility

Every experiment writes config + seed + metrics under `results/`. The feature
table carries `manifest.json` with the **sha256 of every checkpoint** that
produced it, plus class orders and window parameters — so any result can be
traced to the exact weights behind it, on either machine.

---

### Open points you might want to change
- **T02 and T04 are now recordable** on `data/final` and should be reported.
- **Re-run the F02 safety evaluation** on `data/final` before any safety claim.
- **Kitchen is 10/31 rows** — every `data/final` number in this project is
  classroom-only until that is closed.
- **Report every masking number against its ceiling**, not in isolation.
