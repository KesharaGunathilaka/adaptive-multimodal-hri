# How the v4 fusion engine reached 0.82 — a step-by-step guide

A plain-language account of the method, the reasoning behind each decision, and
the evidence for it. Written so it can be read straight into the thesis
methodology chapter.

**Final outcome:** intent accuracy **0.8202** against a rule-based baseline of
**0.7646** on 773 held-out recorded clips (+0.056, McNemar p = 1.7e-07), with the
four cue models left completely untouched.

---

## The starting position

We had four already-trained perception models — emotion, gesture, motion,
context — and needed a fusion model that turns their outputs into one of nine
human-intent classes. The models were **frozen**: not allowed to retrain them.

The previous design (v3) had plateaued at about 0.69, below the rule-based
baseline, and a long series of model-side fixes had each bought 0.01–0.02:
architecture sweeps, class weighting, focal loss, augmentation rebalancing, an
auxiliary cue-classification objective. None closed the gap.

**The key realisation:** if a dozen model changes all fail by the same margin,
the problem is probably not the model. So instead of tuning harder, we went back
and asked a more basic question — *is the thing we are asking the model to
predict actually predictable from the cues we have?*

---

## Step 1 — Measure what the cue models can really discriminate

*(script `90_cue_reliability.py`)*

Before designing anything we measured, per cue: accuracy, the full confusion
matrix, and which class pairs get mixed up.

**Two findings changed everything.**

**(a) The training-split numbers are inflated.** 57% of the training clips (888
of 1547) are footage the cue models were themselves fine-tuned on. Measured
there, emotion looks 97% accurate. Measured on held-out data it is 76%.

| cue | on fine-tuned footage | on held-out data |
|---|---|---|
| emotion | 0.973 | **0.762** |
| gesture | 1.000 | **0.864** |
| motion | 0.845 | **0.671** (0.399 on some footage) |
| context | 0.988 | **0.995** |

Had we designed against the training numbers we would have assumed 97% emotion
accuracy where deployment delivers 76% — and every downstream decision would
have been wrong.

**(b) Motion is unusable, and the confusions were not the ones we assumed.**
Motion scores 0.435 on honest footage: it calls a standing person "walking" 44%
of the time, and detects stepping-back correctly only 15% of the time. Even
collapsing it to static/approach/retreat only reaches 0.507.

Meanwhile the emotion errors that actually matter are **Neutral→Sad (14.6%)** and
**Disgust→Anger (12.5%)** — not the Fear/Surprise pair we had assumed from an
earlier, contaminated measurement.

---

## Step 2 — Redesign the intent taxonomy around those measurements

*(script `91_taxonomy_v4.py`)*

This is the step that produced the result. The old rubric decided intents like
this:

- `thumbs_up` + **anger** → frustration, but `thumbs_up` + **disgust** → break request
- `idle` + **sad** → disengaged, but `idle` + **neutral** → busy
- anything + **walking** vs not-walking → different intents

Every one of those boundaries sits exactly on a distinction the cue models get
wrong. The taxonomy was, in effect, asking the perception stack for information
it does not have.

**The v4 design rules:**

1. **Key intents on the most reliable cue first.** Gesture (0.864) leads,
   emotion second, context as a modifier.
2. **Never let an unreliable cue decide alone.** Motion (0.435) is removed from
   the rubric entirely.
3. **Use emotion superclasses, not fine classes.** `alarm` = {Fear, Surprise},
   `hostile` = {Anger, Disgust}, plus positive / low / neutral. The two most
   damaging confusions now happen *inside* a superclass, so they no longer change
   the answer.
4. **Split similar intents on the reliable cue.** Frustration (F07) vs break
   request (F08) is now decided by **thumbs-up vs thumbs-down** — a gesture
   distinction the model gets right — instead of anger vs disgust, which it does
   not.
5. **Keep every intent actionable.** Intents were only merged if they map to the
   same robot action, so accuracy was not bought by collapsing classes.

**A go/no-go gate before any training.** We computed, on the validation split,
the best score *any* method could reach with these cues, and what the rule
cascade scores under the new rubric:

| | ceiling | rule baseline |
|---|---|---|
| old taxonomy | ~0.90 | 0.700 |
| **new taxonomy (v4)** | **0.951** | **0.827** |

The redesign lifted the rule baseline by **+0.127 before a single model was
trained**. That is the honest source of most of the final gain.

---

## Step 3 — Build the training data

*(script `92_fusion_v4.py`)*

**Relabel, do not re-record.** Existing footage was re-labelled under the new
rubric using the cue annotations already in the dataset. No new recording.

**Fill the combination gaps synthetically.** The intent depends on a
*combination* of cues, and the recorded scenarios only cover a fraction of the
possible combinations — so the model would never see most of them. We therefore
enumerate the cue-combination space (5 emotion superclasses × 8 gestures × 2
contexts = 80 combinations) and build training examples for each by sampling
**real measured cue vectors** from training clips of the matching class.

Nothing is invented: the cue measurements are real, taken from real recordings.
Only the *pairing* is new, and the label comes from the rubric.

This is the single largest data-side lever — removing it drops accuracy from
about 0.65 to 0.51.

**Two protocols, both reported.** The test set's only cue-conflict family
(sarcastic thumbs-up) appears in no recorded training clip.
- *Strict*: exclude every test cue-combination from augmentation → 0.658. This
  asks the model to invent a rule it has never seen in any form.
- *Full rubric*: enumerate the complete authored rubric → 0.820. No test clip,
  scenario or actor is used; only the combination is enumerated.

The second is how a deployed system is actually built — you define the intent
rules, then teach the model every case you defined. Both numbers belong in the
thesis, with the protocol stated.

---

## Step 4 — The fusion model

*(script `92_fusion_v4.py`)*

Self-attention over three cue tokens (emotion superclass, gesture, context). Each
cue is projected to a shared space, the tokens attend to each other, and a
classification head reads out the intent. Missing cues are removed from attention
entirely rather than filled with a placeholder.

We compared three ways of representing each cue and found them **statistically
tied** (~0.65 at that stage): the cue models' own softmax, PCA-reduced penultimate
embeddings, and a two-stage variant that re-classifies the embedding first. We
kept the softmax form for simplicity; the embedding approach is not disadvantaged.

We also confirmed empirically that adding a motion token changes nothing
(0.6520 → 0.6516), which validates dropping motion from the rubric.

---

## Step 5 — Fix missing-cue robustness

*(script `96_t03_improve.py`)*

The first model was trained with no cue-dropout, so it had never operated without
a cue: hiding emotion collapsed it from 0.82 to 0.44.

The obvious fix — train with cue-dropout — **made things worse** (gesture-hidden
fell to 0.347). The reason is subtle and worth stating: when a cue is hidden, the
correct answer is the most likely intent *given the remaining cues*, which depends
on how often each combination actually occurs. Synthetic rows are spread evenly
over all 80 combinations, so dropping cues on them teaches the model a flat,
unrealistic prior.

**Fix: apply cue-dropout only to recorded rows**, which carry the true
distribution, while synthetic rows keep all cues for combination coverage.

| | full cue | emotion hidden | gesture hidden |
|---|---|---|---|
| no dropout | 0.8189 | 0.4360 | 0.4450 |
| dropout on everything | 0.8215 | 0.4269 | **0.3467** |
| **dropout on recorded rows only** | **0.8215** | **0.4748** | 0.4437 |

**Always compare against the achievable ceiling.** With emotion hidden, a
thumbs-up is genuinely either friendly or sarcastic and nothing left can separate
them — the best any method could do is 0.763, not 1.0. Fusion reaches 63% of that.

---

## Step 6 — Evaluate

*(scripts `95_eval_v4.py`, `97_final_matrices.py`)*

Test-type membership is **derived from each clip's cue signature** rather than
read from a tag column, so the evaluation cannot silently drift from the taxonomy.
Reported with 10 random seeds (majority vote), against the rule baseline on
identical clips, with McNemar significance testing.

**Results.**

| | fusion | rules |
|---|---|---|
| Accuracy | **0.8202** | 0.7646 |
| Macro-F1 | **0.7968** | 0.7385 |
| Unseen actor | **0.8182** | 0.8000 |
| Cue conflict (T02) | **0.9247** | 0.8973 |
| Emergency class F1 | **0.818** | 0.565 |

Fusion wins 12 of 17 test scenarios and 7 of 9 intents, and beats rules under
every single-cue loss.

---

## Why it worked — the one-sentence version

**We stopped asking the model to make distinctions the perception stack cannot
make, and instead defined the intents around the distinctions it can.**

The gain breaks down roughly as: taxonomy redesign **+0.13**, combination-coverage
augmentation **+0.15** over no augmentation, missing-cue fix **+0.04** on degraded
conditions, and model/representation tuning **~0.02**. The model was never the
bottleneck.

---

## What to state honestly in the thesis

1. **The augmentation protocol.** 0.820 uses the full authored rubric; the strict
   held-out-combination protocol gives 0.658. Name which one each number uses.
2. **The labels are rubric-derived.** Intent labels come from the cue rubric, so
   the rule baseline is strong by construction. Fusion's advantage is that it
   weighs noisy evidence rather than following one branch — which is exactly where
   it wins (emergency F1 0.818 vs 0.565, conflicts, degraded cues).
3. **T04 is not yet properly measured.** The taxonomy's context-dependent branch
   has no test clips. The Hospital/Museum AI scenarios are the right material for
   it, but they must first be processed through the same cue pipeline as the
   training data, or the result confounds a new environment with a different
   pipeline.
4. **Known data gaps.** Scenarios #24 and #56 have missing context annotations,
   and #57 (neutral + no gesture) scores 0.32 for both methods because there is
   genuinely almost nothing to key on.
