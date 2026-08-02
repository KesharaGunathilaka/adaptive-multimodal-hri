# Stage 5 — Baselines: the evidence for (and against) fusion

**Goal of this stage:** establish what the fusion model must beat, and *why*
beating it is meaningful. Without baselines, "fusion gets 0.93" is a number with
no argument attached. With them it becomes the G1 claim: *weighing all four cues
jointly beats any single cue and beats rule-based logic.*

Code: `fusion/baselines/` · runner: `scripts/04_run_baselines.py`

**Read the honesty note in §5.4 before quoting any number from this stage.** The
baseline comparison came out one way on `data/old` and the *opposite* way on
`data/final`, and understanding why is the most important methodological result
in the project so far.

---

## 5.0 One table to orient yourself

Every baseline number in this document varies along **three** axes, and mixing
them up is the fastest way to misread the results:

1. **Which dataset** — `data/old` or `data/final`
2. **What was held out** — *people* or *scenarios (cue combinations)*
3. **Which cues** — *real* (model predictions, noisy) or *oracle* (the table's
   true labels as one-hot; a diagnostic, not achievable in deployment)

| Dataset | Held out | Cues | System | Test score |
|---|---|---|---|---|
| `data/old` | people | real | rule-based | 0.695 |
| `data/old` | people | real | best unimodal (emotion) | 0.732 |
| `data/old` | people | real | concat-MLP | 0.931 |
| `data/old` | people | real | **attention fusion** | **0.951** ✅ |
| `data/final` | scenarios | **oracle** | rule-based | **0.900** (= ceiling) |
| `data/final` | scenarios | **oracle** | attention fusion | 0.500 |
| `data/final` | scenarios | real | **rule-based** | **0.494** ✅ |
| `data/final` | scenarios | real | attention fusion | 0.325 |

**Why "test" means two different things.** On `data/old` all 22 recorded
scenarios were V3 *train* rows — the V3 test scenarios did not exist yet — so the
only available holdout was **people**. `data/final` (the 2026-07-25 shoot) added
the test rows, so the holdout is now **scenarios**, which is the far harder
question the table was designed to ask.

**One-sentence summary:** fusion wins when the situation is familiar and the cues
are noisy; rules win when the situation is new. Rows 4 and 8 are the same model.

---

## 5.1 The four baseline families

All consume the same cue vectors from Stage 3 and predict an intent (F01–F10),
so every comparison is apples-to-apples.

### (a) Rule-based — the "traditional HRI" system
Hand-coded `if/else` over the **argmax** cue labels, mirroring the V3 table's
labeling rubric (§2.6) as closely as possible: *thumbs_down + angry → F07;
thumbs_down + disgust → F08; beckoning + sad → F04; both_hands_up + fear →
F02 …* (`fusion/baselines/rule_based.py`).

This is the system the thesis argues against, so it must be implemented
*fairly* — a deliberately weak strawman would invalidate the comparison. One
concession is forced on it: **the rubric needs a direction cue that no model
outputs**, so where the rubric says "wave toward robot → F01, wave toward exit →
F09" the rule takes the more common reading. That ambiguity is not a bug in the
implementation; it is the direction gap, and it caps *any* method at 0.90.

### (b) Unimodal — one cue at a time
A small logistic regression per modality, mapping that modality's probabilities
alone to an intent. Answers "how much of the intent is recoverable from emotion
alone / gesture alone / …". These are the numbers the G1 claim is measured
against.

### (c) Concat-MLP — naive fusion
Concatenate all four probability vectors (+ the four observed flags) into one
28-dim input → 128 → 64 → 10. No attention, no modality structure. This is the
*floor for fusion*: if the attention model cannot beat a plain MLP on the same
inputs, the architecture is not earning its complexity.

### (d) Oracle variants — diagnostics, not baselines
Added 2026-07-28. The same rule-based and learned models, but fed the **table's
true cue labels** (one-hot) instead of the models' predictions. These do not
compete with fusion; they *decompose the error* into "bad cues" vs "bad
reasoning" (Stage 6 / `GAP_DECOMPOSITION.md`).

---

## 5.2 Results on `data/old` — where fusion wins convincingly

Actor-disjoint test subjects (82 clips), clip-level majority vote, 3 seeds.
Here "test" means **unseen people performing the same 22 scenarios**.

| Model | Clip accuracy | Clip macro-F1 |
|---|---|---|
| Rule-based | 0.695 | 0.452 |
| Unimodal — emotion | 0.732 | 0.436 |
| Unimodal — motion | 0.512 | 0.234 |
| Unimodal — gesture | 0.427 | 0.227 |
| Unimodal — context | 0.146 | 0.150 |
| **Concat-MLP** | **0.931 ± 0.011** | 0.621 ± 0.011 |
| **Attention fusion** | **0.951 ± 0.017** | 0.629 ± 0.038 |

Three readings:

1. **G1 holds strongly here.** Fusion (0.951) beats the best single cue (0.732)
   by ~22 points and the rule system (0.695) by ~26.
2. **Context alone is near-useless (0.146)** — and that is the *right* result. A
   kitchen hosts many intents; context only has value as a *conditioner* of other
   cues. Its uselessness alone is part of the fusion argument.
3. **Why learned fusion beats rules here:** it learns to *correct systematic
   perception errors*. The motion model almost never outputs `stepping_back` in
   the kitchen, but it fails in a *consistent* way; rules are fooled, while the
   MLP learns "this probability pattern + disgust + kitchen = actually a step
   back → F08". Rules see argmax labels and cannot recover.

The macro-F1 (~0.62) is deflated for a mechanical reason: **F10 had zero training
rows** (its only scenario, S28, was in the recombination pool), so F10 scored 0
and capped macro-F1 at 0.9.

---

## 5.3 Results on `data/final` — where the comparison reverses

Classroom, RealSense view, 4 s mean-pooled windows, 3 seeds. Here "test" means
the **V3 test rows** — unseen *cue combinations*, not just unseen people.

| Model | Train rows | **Test rows** |
|---|---|---|
| Rule-based + **oracle** cues | 0.977 | **0.900** ← hits the ceiling exactly |
| Rule-based + **real** cues | 0.758 | **0.494** |
| Learned fusion + **oracle** cues | not measured | **0.500** |
| Learned fusion + **real** cues | not measured † | **0.325** |

† `ASSESSMENT.md` reports 0.887 on train rows, but that is the **frozen fusion v1
model** (trained on `data/old`), not the model retrained here — the two are not
comparable, so the cell is left blank rather than filled with a number from a
different run. Train-row accuracy is in any case a weak diagnostic: these rows
were in training.

**On the honest test set the rule-based baseline beats the learned fusion model
(0.494 vs 0.325).** The G1 claim, as measured on `data/final`, currently **does
not hold**.

This is not a contradiction of §5.2 — it is the same two systems being asked a
different question:

| | `data/old` test | `data/final` test |
|---|---|---|
| What is held out | **people** | **cue combinations** |
| Winner | learned fusion (0.951 vs 0.695) | **rules** (0.494 vs 0.325) |
| Why | fusion corrects perception noise on *familiar* tuples | rules generalise compositionally *by construction* |

The rule system encodes the rubric explicitly, so an unseen tuple costs it
nothing. The learned model sees ~19 tuples in training and cannot extrapolate the
compositional rule to 10 unseen ones — it memorises tuple→intent. Meanwhile the
rule system's weakness is unchanged: perception noise drops it from 0.900
(oracle) to 0.494 (real), a 40-point fall the learned model does not suffer as
steeply in relative terms.

**Each method has exactly the strength the other lacks:**

| | compositional generalisation | noise robustness |
|---|---|---|
| Rules | ✅ by construction | ❌ brittle to argmax errors |
| Learned fusion | ❌ needs to have seen the tuple | ✅ learns systematic error patterns |

That observation is the argument for the fix proposed in
`GAP_DECOMPOSITION.md`: **rubric-driven cue recombination** — synthesise training
samples spanning the combinatorial cue space (supplying the semantics rules have)
using *real, noisy cue vectors* as the source (preserving the robustness learning
gives). The target is a model that has both columns ticked.

---

## 5.4 Honesty notes (do not skip these when writing up)

1. **Do not quote the `data/old` baseline table as evidence for G1 without the
   qualifier.** Its test split holds out *people*, not situations. It is a valid
   result about subject-generalisation and an invalid one about the harder claim.
2. **The current honest headline is that rules beat fusion on unseen
   combinations.** Reporting this is not a failure — it is the finding that
   motivates the augmentation contribution, and it is far better to discover it
   ourselves than to have an examiner find it.
3. **If fusion is later trained on rubric-generated samples, it is being *taught*
   the rubric**, not discovering it. The claim then becomes "rule knowledge +
   learned noise-robustness beats either alone", which is supported by the two
   tables above and is directly testable.
4. **The rule baseline must stay honest.** It is implemented from the table's own
   rubric and hits the theoretical ceiling (0.900) when given perfect cues —
   proof that it is not a strawman. Any future change to it must preserve that
   property.

---

## 5.5 What Stage 5 produced

```
fusion/baselines/
├── common.py        shared loading, splits, clip-vote, metrics
├── rule_based.py    the V3 rubric as if/else over argmax cues
├── learned.py       per-modality logistic regression + ConcatMLP
├── results.json     data/old baseline numbers
└── RESULTS.md       the data/old table

results/realworld_eval_final/GAP_DECOMPOSITION.md   the data/final picture
```

**In one sentence:** the baselines established that on familiar cue combinations
learned fusion beats rules by ~26 points, and that on *unseen* combinations the
relationship inverts — which localises the project's central problem and defines
what the fusion model must be given in order to win on both.

---

### Open points you might want to change
- **Concat-MLP has not been re-measured on `data/final`.** Worth adding for
  completeness — the attention-vs-MLP comparison should be redone on the honest
  split before any architecture claim is made.
- **Kitchen is only 10 of 31 rows recorded**, so §5.3 is classroom-only.
- **The rule baseline could be strengthened** once a direction cue exists; its
  ceiling would rise from 0.90 to 1.00 and the comparison should be re-run.
