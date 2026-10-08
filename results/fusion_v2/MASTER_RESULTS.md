# v4 Fusion Engine — complete results

**System.** Self-attention fusion over three cue tokens (emotion superclass /
gesture / context), built on the v4 perception-aware intent taxonomy, trained with
full-rubric cue-combination augmentation and cue-dropout 0.7 on recorded rows.
Ten random seeds, majority vote. The four cue models are frozen throughout.

**Evaluation.** 773 held-out recorded clips (scenario-disjoint). Baseline is the
v4 rule cascade on identical cues. Synthetic data is never evaluated.

---

## 1. Headline

| metric | fusion | rules | difference |
|---|---|---|---|
| **Accuracy** | **0.8202** | 0.7646 | **+0.0556** |
| **Macro-F1** | **0.7968** | 0.7385 | +0.0583 |
| Accuracy, unseen actor | **0.8182** | 0.8000 | +0.0182 |
| Correct clips | 634 / 773 | 591 / 773 | +43 |

Per-seed **0.8182 ± 0.0061** (95% CI ±0.0038), min 0.8072 / max 0.8279.
**McNemar exact p = 1.68e-07** — fusion right where rules fail on 56 clips, the
reverse on 13.

---

## 2. Comparison against published fusion methods

Identical labels, features, augmentation, seeds and test clips; only the fusion
mechanism differs.

| method | accuracy | macro-F1 | per-seed | vs ours (p) |
|---|---|---|---|---|
| unimodal — context only | 0.1268 | 0.040 | — | 8.4e-140 |
| unimodal — gesture only | 0.4075 | 0.250 | — | 3.9e-71 |
| unimodal — emotion only | 0.4191 | 0.275 | — | 7.4e-73 |
| late fusion (product of experts) | 0.5653 | 0.424 | — | 1.2e-40 |
| gradient-boosted trees | 0.7050 | 0.690 | ±0.005 | 1.4e-16 |
| **rule cascade** | 0.7646 | 0.739 | — | 1.7e-07 |
| channel attention (SE) | 0.7853 | 0.751 | ±0.004 | 6.6e-05 |
| GMU (Arevalo et al. 2017) | 0.7969 | 0.763 | ±0.004 | 9.1e-04 |
| early fusion (concat-MLP) | 0.8008 | 0.772 | ±0.005 | 1.5e-03 |
| LMF (Liu et al. 2018) | 0.8137 | 0.784 | ±0.009 | 0.267 |
| cross-attention | 0.8176 | 0.792 | ±0.009 | 0.804 |
| **self-attention (ours)** | **0.8202** | **0.7968** | ±0.006 | — |

**Reading.** Fusion is necessary: the best single cue reaches 0.419 and naive late
fusion 0.565, so joint reasoning is worth ~+0.40 absolute. Our model beats every
classical and gated baseline significantly. It does **not** beat cross-attention or
low-rank fusion — those are statistically tied — so no claim of architectural
superiority is made. It is selected for the best macro-F1 and the tightest seed
variance among the top three.

---

## 3. Taxonomy ablation — where the gain comes from

Model, augmentation budget, seeds and protocol fixed; only the label space varies.

| taxonomy | what changed | fusion | rules | ceiling | vs v4 |
|---|---|---|---|---|---|
| v3 (original) | 4 cues, fine emotion, anger/disgust split | 0.7624 | 0.7653 | 0.9519 | **−0.0671** (p=1.4e-05) |
| **v4 (ours)** | gesture-first, superclasses, no motion | **0.8294** | 0.7886 | 0.9402 | — |
| v4 − gesture split | F07/F08 back on anger-vs-disgust | 0.7259 | 0.7697 | 0.9257 | **−0.1035** (p=5.9e-15) |
| v4 − alarm merge | only Fear triggers emergency | 0.8324 | 0.7828 | 0.9402 | +0.0029 (p=0.727) |
| v4 + motion | re-admit motion as a discriminator | 0.8120 | 0.7799 | 0.9650 | −0.0175 (p=0.017) |

**Reading.** The taxonomy contributes **+0.067**; the architecture does not (§2).
Splitting frustration/break-request on *gesture polarity* rather than the
confusable anger-vs-disgust distinction is worth **+0.104** — the single largest
effect measured. Excluding motion is worth +0.018, and note that admitting motion
*raises* the theoretical ceiling (0.965) while *lowering* achieved accuracy: an
unreliable cue adds information the model cannot use.

**Null result, reported as such:** merging Fear+Surprise into an "alarm"
superclass has no effect (+0.003, p=0.727). It was hypothesised to help and does
not, because Surprise is rare in the recorded footage.

---

## 4. Results by research test type

| test | what it measures | clips | fusion | rules | diff |
|---|---|---|---|---|---|
| **T01** | Full-cue joint reasoning | 773 | **0.8202** | 0.7646 | +0.056 |
| **T02** | Cue conflict (e.g. sarcastic thumbs-up) | 146 | **0.9247** | 0.8973 | +0.027 |
| **T03** | Missing-cue robustness | see below | | | |
| **T04** | Context generalization, unseen environments | 118 | **0.9831** | 0.9407 | +0.042 |
| **T05** | Rule-baseline comparison | 773 | **0.8202** | 0.7646 | +0.056 |

### T03 — missing-cue robustness

Each cue is removed at evaluation. *Ceiling* = the best any method could achieve
with that cue hidden (with emotion gone, a thumbs-up is genuinely either friendly
or sarcastic and nothing remaining separates them).

| cue removed | fusion | rules | ceiling | fusion vs ceiling |
|---|---|---|---|---|
| emotion | **0.4787** | 0.4308 | 0.7633 | 63% of achievable |
| gesture | **0.4334** | 0.4062 | 0.7128 | 61% of achievable |
| context | **0.8189** | 0.7646 | 1.0000 | 82% of achievable |

Fusion beats rules under every cue loss. Losing context is nearly free
(0.820 → 0.819); losing emotion or gesture is costly, but roughly a third of that
loss is irreducible.

### T04 — unseen environments (Hospital / Museum)

Trained only on Classroom and Kitchen; these clips are environments the model has
never seen, processed through the same cue pipeline as training.

| slice | clips | fusion | rules | diff |
|---|---|---|---|---|
| **Novel environments** | 118 | **0.9831** | 0.9407 | +0.042 |
| — Hospital | 50 | 0.9800 | 0.9000 | +0.080 |
| — Museum | 68 | 0.9853 | 0.9706 | +0.015 |
| Control: AI Kitchen (known env.) | 11 | 1.0000 | 1.0000 | 0.000 |

The control isolates the cost of novelty at **−0.017**. These clips are cleaner
than recorded footage (all cues observed), so 0.983 evidences *context transfer*,
not general accuracy.

---

## 5. Results per intent

| intent | meaning | clips | recall | precision | F1 | rules F1 |
|---|---|---|---|---|---|---|
| **F01** | Greeting / acknowledgement | 146 | 0.822 | 0.976 | **0.892** | 0.789 |
| **F02** | Emergency / danger | 38 | **0.947** | 0.720 | **0.818** | 0.565 |
| **F03** | Task assistance request | 95 | 0.842 | 0.721 | **0.777** | 0.695 |
| **F04** | Help request | 80 | 0.738 | 0.756 | **0.747** | 0.730 |
| **F05** | Busy — do not interrupt | 80 | 0.750 | 0.732 | 0.741 | 0.757 |
| **F06** | Requests passage / space | 53 | 0.774 | 0.672 | **0.719** | 0.635 |
| **F07** | Frustration / agitation | 146 | 0.925 | 0.944 | **0.934** | 0.913 |
| **F08** | Break / relief request | 41 | 0.805 | 0.611 | **0.695** | 0.667 |
| **F10** | Discouraged / disengaged | 94 | 0.745 | 0.986 | 0.848 | 0.897 |

Fusion leads on **7 of 9** intents by F1. The largest gain is the safety-critical
**F02 (0.818 vs 0.565)**: recall 0.947 with precision 0.720 means it over-triggers
rather than misses — the correct direction for an emergency class.

---

## 6. Results per test case (scenario)

Cue signature = emotion superclass / gesture / context.

| scenario | cue signature | intent | clips | fusion | rules | diff |
|---|---|---|---|---|---|---|
| #18 | positive / wave / classroom | F01 | 48 | 0.812 | 0.542 | **+0.271** |
| #24 | neutral / beckoning / — | F03 | 53 | 0.868 | 0.604 | **+0.264** |
| #26 | neutral / both_hands_up / classroom | F05 | 52 | 0.981 | 0.885 | +0.096 |
| #27 | hostile / point / classroom | F06 | 53 | 0.774 | 0.755 | +0.019 |
| #28 | hostile / thumbs_up / classroom | F07 | 55 | 0.909 | 0.873 | +0.036 |
| #29 | hostile / thumbs_up / classroom | F07 | 49 | 0.980 | 0.939 | +0.041 |
| #31 | low / idle / classroom | F10 | 52 | 0.731 | 0.788 | −0.058 |
| #32 | positive / thumbs_up / kitchen | F01 | 54 | 0.852 | 0.833 | +0.019 |
| #54 | alarm / idle / kitchen | F02 | 38 | 0.947 | 0.974 | −0.026 |
| #55 | positive / point / kitchen | F03 | 42 | 0.810 | 0.786 | +0.024 |
| #56 | low / beckoning / — | F04 | 34 | 0.559 | 0.529 | +0.029 |
| #57 | neutral / idle / kitchen | F05 | 28 | 0.321 | 0.250 | +0.071 |
| #59 | hostile / thumbs_up / kitchen | F07 | 42 | 0.881 | 0.881 | 0.000 |
| #60 | hostile / thumbs_down / kitchen | F08 | 41 | 0.805 | 0.780 | +0.024 |
| #61 | low / wave / kitchen | F01 | 44 | 0.795 | 0.591 | **+0.205** |
| #62 | low / idle / kitchen | F10 | 42 | 0.762 | 0.881 | −0.119 |
| #63 | low / thumbs_down / kitchen | F04 | 46 | 0.870 | 0.870 | 0.000 |

Fusion higher on **12** scenarios, lower on **3**, equal on **2**. The largest
wins are cases where a single-cue rule takes the wrong branch (#18, #24, #61).
#57 is the weakest case for both methods — with neutral affect and no gesture
there is very little to key on.

---

## 7. Robustness

### Leave-one-actor-out cross-validation

Each actor held out entirely; trained on the others.

| held-out actor | clips | fusion | rules | diff |
|---|---|---|---|---|
| P01 | 700 | 0.9657 | 0.9300 | +0.036 |
| P02 | 655 | 0.9221 | 0.8947 | +0.028 |
| P04 | 414 | 0.8188 | 0.7802 | +0.039 |
| P06 | 530 | 0.9075 | 0.8453 | +0.062 |
| **mean** | | **0.9036** | 0.8625 | **+0.041** |

Fusion wins for every held-out person. These numbers exceed the 0.820 headline
because LOAO holds out *people* while the same *scenarios* remain in training via
other actors — it measures person-generalization, the headline measures
scenario-generalization.

### Cue-reliability sweep

A cue's measured vector is replaced with probability *p* by a real vector from a
different ground-truth class. As *p* rises both methods approach chance, so the
*ratio* is the comparable quantity.

| corrupted | p=0.0 | p=0.2 | p=0.4 |
|---|---|---|---|
| emotion | 1.138× | 1.117× | 1.118× |
| gesture | 1.138× | 1.126× | 1.133× |
| all cues | 1.138× | 1.102× | 1.117× |

Fusion holds a stable ~1.12–1.14× proportional advantage at every perception
quality level — the advantage is not an artefact of one operating point.

---

## 8. What the model attends to

CLS-token attention over the cue tokens, per intent (rows sum to 1).

| intent | emotion | gesture | context |
|---|---|---|---|
| F02 Emergency | **0.747** | 0.127 | 0.126 |
| F10 Disengaged | 0.505 | 0.351 | 0.144 |
| F07 Frustration | 0.449 | 0.437 | 0.114 |
| F04 Help request | 0.426 | 0.460 | 0.114 |
| F01 Greeting | 0.396 | 0.471 | 0.134 |
| F05 Busy | 0.379 | 0.475 | 0.146 |
| F03 Task request | 0.371 | **0.510** | 0.119 |
| F06 Give space | 0.365 | 0.465 | 0.170 |
| F08 Break request | 0.363 | 0.472 | 0.165 |

Intents the taxonomy defines by emotion overriding gesture draw **0.486** mean
emotion attention against **0.393** for gesture-led intents (**+0.093**). F02
(emergency) is the clearest case at 0.747 — alarm overrides whatever the hands are
doing, which is the rubric's first and most safety-critical rule. F08 is the
exception, leaning on the distinctive thumbs-down gesture instead. Attention
describes the forward pass and is consistent with the intended structure; it is
not proof of it.

---

## 9. Cue-model reliability (the design input)

Measured on held-out footage; the basis for every taxonomy decision.

| cue | fine-grained accuracy | notes |
|---|---|---|
| gesture | 0.864 | most reliable → primary discriminator |
| context | 0.995 | reliable but binary → modifier only |
| emotion | 0.762 | moderate; Neutral→Sad 14.6%, Disgust→Anger 12.5% |
| motion | 0.435 | unusable — standing→walking 44%, stepping-back recall 0.149 |

**Contamination warning.** 57% of the training split is footage the cue models
were fine-tuned on, where emotion reads 0.973 against 0.762 at deployment. Cue
reliability must be characterised on held-out data or every downstream design
decision is made against inflated numbers.

---

## 10. Caveats carried into the manuscript

1. **Augmentation protocol.** 0.820 uses the complete authored rubric. The strict
   protocol, which withholds every test cue-combination, gives 0.658 — the test
   set's only conflict family appears in no training data, so the strict number
   measures transfer to an unseen composition. Both are reported; every figure
   states which it uses.
2. **Label provenance.** The intent table was authored by hand before recording,
   and the rubric formalises it, so the rule baseline is strong by construction.
   Fusion's advantage is specifically robustness to perception error, conflict
   handling and context transfer — not a better intent model. A human-annotation
   study (200 clips, 2–3 annotators) is prepared to supply non-circular ground
   truth.
3. **AI-generated T04 clips are cleaner** than recorded footage; 0.983 evidences
   context transfer, not general accuracy.
4. **Known data gaps.** Scenarios #24 and #56 lack context annotation; #57 is
   near-unrecognisable for both methods; AI scenario S67's folder intent (F08)
   disagrees with its v4 label (F07) by design.
