# v4 Fusion engine - final result matrices

**Model.** Self-attention fusion over three cue tokens (emotion superclass, gesture, context), trained on the v4 perception-aware taxonomy with full-rubric augmentation and cue-dropout 0.7 on recorded rows. 10 seeds, majority vote.
**Baseline.** The v4 rule cascade applied to the same measured cues.
**Test set.** 773 recorded clips, held out by scenario; no synthetic data is evaluated.

---

## A. Overall results

*What this shows: headline performance of the fusion model against the rule-based baseline on the whole test set.*

| metric | fusion | rules | difference |
|---|---|---|---|
| **Accuracy** | **0.8202** | 0.7646 | **+0.0556** |
| Macro-F1 (all classes equally weighted) | **0.7968** | 0.7385 | +0.0583 |
| Accuracy on an unseen actor | **0.8182** | 0.8000 | +0.0182 |
| Correct clips | 634 / 773 | 591 / 773 | +43 |

- Per-seed accuracy **0.8182 ± 0.0061** (95% CI ±0.0038), min 0.8072 / max 0.8279 - the model is stable across random initialisations.
- **Statistical significance:** fusion is right where rules are wrong on **56** clips; rules right where fusion is wrong on **13**. McNemar exact p = **1.68e-07** (significant at α=0.05).

---

## B. Per test case (scenario)

*What this shows: one row per recorded test scenario, with the cue signature the actors performed. `Cue signature` is emotion-superclass / gesture / context.*

| scenario | cue signature | intent | clips | fusion | rules | diff |
|---|---|---|---|---|---|---|
| #18 | positive / wave / classroom | F01 | 48 | 0.812 | 0.542 | +0.271 |
| #24 | neutral / beckoning / None | F03 | 53 | 0.868 | 0.604 | +0.264 |
| #26 | neutral / both_hands_up / classroom | F05 | 52 | 0.981 | 0.885 | +0.096 |
| #27 | hostile / point / classroom | F06 | 53 | 0.774 | 0.755 | +0.019 |
| #28 | hostile / thumbs_up / classroom | F07 | 55 | 0.909 | 0.873 | +0.036 |
| #29 | hostile / thumbs_up / classroom | F07 | 49 | 0.980 | 0.939 | +0.041 |
| #31 | low / idle / classroom | F10 | 52 | 0.731 | 0.788 | -0.058 |
| #32 | positive / thumbs_up / kitchen | F01 | 54 | 0.852 | 0.833 | +0.019 |
| #54 | alarm / idle / kitchen | F02 | 38 | 0.947 | 0.974 | -0.026 |
| #55 | positive / point / kitchen | F03 | 42 | 0.810 | 0.786 | +0.024 |
| #56 | low / beckoning / None | F04 | 34 | 0.559 | 0.529 | +0.029 |
| #57 | neutral / idle / kitchen | F05 | 28 | 0.321 | 0.250 | +0.071 |
| #59 | hostile / thumbs_up / kitchen | F07 | 42 | 0.881 | 0.881 | +0.000 |
| #60 | hostile / thumbs_down / kitchen | F08 | 41 | 0.805 | 0.780 | +0.024 |
| #61 | low / wave / kitchen | F01 | 44 | 0.795 | 0.591 | +0.205 |
| #62 | low / idle / kitchen | F10 | 42 | 0.762 | 0.881 | -0.119 |
| #63 | low / thumbs_down / kitchen | F04 | 46 | 0.870 | 0.870 | +0.000 |

Fusion scores higher on **12** scenarios, lower on **3**, equal on **2**.

---

## C. Per intent

*What this shows: how reliably each of the nine intents is recognised. Recall = of all clips that truly are this intent, the share correctly identified. Precision = of all clips predicted as this intent, the share that really are.*

| intent | meaning | clips | fusion recall | fusion prec. | fusion F1 | rules F1 |
|---|---|---|---|---|---|---|
| **F01** | Greeting / positive acknowledgement | 146 | 0.822 | 0.976 | **0.892** | 0.789 |
| **F02** | Emergency / danger | 38 | 0.947 | 0.720 | **0.818** | 0.565 |
| **F03** | Task assistance request | 95 | 0.842 | 0.721 | **0.777** | 0.695 |
| **F04** | Help request | 80 | 0.738 | 0.756 | **0.747** | 0.730 |
| **F05** | Engaged / busy - do not interrupt | 80 | 0.750 | 0.732 | **0.741** | 0.757 |
| **F06** | Requests passage / space | 53 | 0.774 | 0.672 | **0.719** | 0.635 |
| **F07** | Frustration / agitation | 146 | 0.925 | 0.944 | **0.934** | 0.913 |
| **F08** | Break / relief request | 41 | 0.805 | 0.611 | **0.695** | 0.667 |
| **F10** | Discouraged / disengaged | 94 | 0.745 | 0.986 | **0.848** | 0.897 |

---

## D. Per test type

*What this shows: performance on each research test type. These overlap (a clip can belong to several), so they do not sum to the total.*

| test | what it measures | clips | fusion | rules | diff |
|---|---|---|---|---|---|
| **T01** | Full-cue fusion: all cues present, joint reasoning required | 773 | 0.8202 | 0.7646 | +0.0556 |
| **T02** | Cue conflict: emotion contradicts the gesture (e.g. sarcastic thumbs-up) | 146 | 0.9247 | 0.8973 | +0.0274 |
| **T04** | Context generalization: same cues, different environment | 773 | 0.8202 | 0.7646 | +0.0556 |
| **T05** | Rule-baseline comparison: the whole test set | 773 | 0.8202 | 0.7646 | +0.0556 |

**T03 - missing-cue robustness.** Each cue is removed at evaluation time and the model must decide from what is left. The *ceiling* is the best score any method could reach with that cue hidden: with emotion gone, a thumbs-up is genuinely either friendly (F01) or sarcastic (F07) and nothing remaining can separate them.

| cue removed | fusion | rules | ceiling | fusion vs ceiling |
|---|---|---|---|---|
| emotion | **0.4787** | 0.4308 | 0.7633 | 63% of achievable |
| gesture | **0.4334** | 0.4062 | 0.7128 | 61% of achievable |
| context | **0.8189** | 0.7646 | 1.0000 | 82% of achievable |
