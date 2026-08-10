# Phase 1 — robustness battery: where fusion can actually beat rules

Generated 2026-08-06 19:32 · `data/final_merged` · headline test clips · 3-seed ensemble.

## Why this study exists

Headline accuracy is fusion **0.7191** vs rules **0.712** — a tie. That is a property of how the labels were built, not a failed hypothesis: each clip's intent comes from its V3 row, and that row's intent is a function of its cue tuple, so `y = rule_intent(true_cues)`. A rule system given true cues is therefore **Bayes-optimal by construction** — confirmed independently by the gap decomposition (rules+oracle = 1.000, exactly the ceiling). Rules can only lose points to *perception* error, never to reasoning error, because their reasoning defines correctness. Recombination then trains fusion on `rule_intent()` labels, making it a distillation student of that same teacher. Clean-data accuracy was never a fair arena.

These three tests probe the regimes a hand-written argmax rubric structurally cannot handle.

## A — Does the rule baseline get an unfair context advantage?

`merged_gap.rule_predict` hands rules the clip's TRUE context while fusion must infer it from the CLIP classifier.

| Rule variant | Clip acc | Macro-F1 |
|---|---|---|
| true context (as previously reported) | 0.712 | 0.6414 |
| predicted context (fair) | 0.712 | 0.6414 |

Only **0 clips** change. The advantage is real but nearly worthless — because `rule_intent` branches on context in **exactly one place** (`raise_hand`: classroom vs kitchen). **That is the finding**: the rubric is effectively a *three-cue* system. Hand-authoring context-dependence across every branch is hard, so it was never written — a structural limit of rule systems that fusion does not share (it can condition on context everywhere, for free). It also means T04-style context claims cannot be strongly supported by *this* rule baseline in its current form.

## B — Degradation: the G2 experiment

Gaussian noise on log-probabilities (same functional form as training jitter), renormalised, applied to the **identical** corrupted table for every system at each sigma. Chosen over temperature smoothing because a monotonic rescale leaves `argmax` untouched — rules would be trivially invariant and the comparison rigged.

`full` = deployed recipe (trained with dropout+jitter, so partly adapted to this noise — a legitimate advantage of learned systems, but flagged). `recomb_only` = no dropout, no jitter in training, so its robustness is architectural rather than trained-in.

| sigma | Rules | Fusion `full` | Fusion `recomb_only` |
|---|---|---|---|
| 0.0 | 0.712 | 0.716 | 0.715 |
| 0.15 | 0.713 | 0.713 | 0.7038 |
| 0.3 | 0.7038 | 0.6956 | 0.6966 |
| 0.5 | 0.6823 | 0.6772 | 0.6772 |
| 0.75 | 0.6333 | 0.6425 | 0.6445 |
| 1.0 | 0.5996 | 0.6016 | 0.6108 |
| 1.5 | 0.5036 | 0.4964 | 0.5046 |
| 2.0 | 0.4065 | 0.4168 | 0.4178 |

From sigma 0 → 2.0: rules 0.712 → 0.4065 (**-0.3055**), fusion `full` 0.716 → 0.4168 (**-0.2992**), fusion `recomb_only` 0.715 → 0.4178 (**-0.2972**).

Read the *slopes*, not the intercepts: the intercept is the tie we already knew about; the slope is whether discarding confidence (`argmax`) costs you when cues get unreliable.

## C — F02 emergency: a capability rules cannot have

Rules emit one hard label with no confidence, so they occupy a **single point** in precision/recall space. Fusion emits a distribution, so the emergency threshold is tunable — the operating point can be moved toward recall, which is what a safety-critical HRI system actually needs.

True F02 clips in the headline test set: **121**.

- **Rules (single point):** precision 0.481, recall 0.727 (tp=88, fp=95, fn=33) — **not adjustable**
- **Fusion recall at rules' precision:** 0.6446280991735537
- **Fusion precision at rules' recall:** None

| tau | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|
| 0.02 | 0.672 | 0.645 | 78 | 38 | 43 |
| 0.05 | 0.720 | 0.595 | 72 | 28 | 49 |
| 0.08 | 0.726 | 0.570 | 69 | 26 | 52 |
| 0.11 | 0.739 | 0.562 | 68 | 24 | 53 |
| 0.14 | 0.744 | 0.554 | 67 | 23 | 54 |
| 0.17 | 0.756 | 0.537 | 65 | 21 | 56 |
| 0.2 | 0.756 | 0.537 | 65 | 21 | 56 |
| 0.23 | 0.774 | 0.537 | 65 | 19 | 56 |
| 0.26 | 0.771 | 0.529 | 64 | 19 | 57 |
| 0.29 | 0.780 | 0.529 | 64 | 18 | 57 |
| 0.32 | 0.790 | 0.529 | 64 | 17 | 57 |
| 0.35 | 0.800 | 0.529 | 64 | 16 | 57 |
| 0.38 | 0.810 | 0.529 | 64 | 15 | 57 |
| 0.41 | 0.810 | 0.529 | 64 | 15 | 57 |
| 0.44 | 0.818 | 0.521 | 63 | 14 | 58 |
| 0.47 | 0.816 | 0.512 | 62 | 14 | 59 |
| 0.5 | 0.824 | 0.504 | 61 | 13 | 60 |
| 0.53 | 0.847 | 0.504 | 61 | 11 | 60 |
| 0.56 | 0.847 | 0.504 | 61 | 11 | 60 |
| 0.59 | 0.857 | 0.496 | 60 | 10 | 61 |
| 0.62 | 0.857 | 0.496 | 60 | 10 | 61 |
| 0.65 | 0.855 | 0.488 | 59 | 10 | 62 |
| 0.68 | 0.881 | 0.488 | 59 | 8 | 62 |
| 0.71 | 0.879 | 0.479 | 58 | 8 | 63 |
| 0.74 | 0.879 | 0.479 | 58 | 8 | 63 |
| 0.77 | 0.877 | 0.471 | 57 | 8 | 64 |
| 0.8 | 0.877 | 0.471 | 57 | 8 | 64 |
| 0.83 | 0.869 | 0.438 | 53 | 8 | 68 |
| 0.86 | 0.895 | 0.421 | 51 | 6 | 70 |
| 0.89 | 0.907 | 0.405 | 49 | 5 | 72 |
| 0.92 | 0.922 | 0.388 | 47 | 4 | 74 |
| 0.95 | 0.955 | 0.347 | 42 | 2 | 79 |

**Correction to the a priori hypothesis this test was built to check**: the "fusion has a tunable safety dial, rules don't" argument is true in principle but does NOT hold empirically here. Fusion's recall CEILING across the entire threshold sweep — even at the most permissive tau=0.02 — is **0.645**, which never reaches rules' fixed-point recall of **0.727**. The dial exists, but it cannot be turned far enough: some true-F02 clips get low F02 probability from the model at every threshold, which no amount of threshold-lowering fixes. This is a training-time deficiency (the model itself under-predicts F02), not a deployment-time thresholding question — consistent with the earlier gap-decomposition finding that F02 rows fail even with ORACLE cues (`GAP_DECOMPOSITION_MERGED.md`) and the scenario report's finding that fusion loses to rules specifically on the two F02-with-missing-gesture rows (`SCENARIO_TEST_REPORT.md`). **F02 recall is a genuine, real weakness of fusion relative to rules, not an artefact of an untuned threshold.** Candidate fix: class-weighted loss favouring F02 recall during fusion training (the same intervention that worked for emotion/gesture unimodal fine-tuning), not evaluated here.