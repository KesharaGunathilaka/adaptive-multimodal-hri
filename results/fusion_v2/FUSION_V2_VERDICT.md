# Fusion engine v2 — findings and verdict

Built from the four **frozen** cue models, targeting ~0.80 intent accuracy with a
self-attention fusion over cue embeddings. Recorded test set n=773 (v4-labellable
clips); val n=283 used for all design decisions.

## Phase 1 — what the frozen cue models can actually discriminate

Two findings, both of which changed the design.

**(a) Train-split cue accuracy is inflated by fine-tuning.** 57% of the train split
(888 of 1547 clips) is `curated_clip` footage the cue models were fine-tuned on:

| cue | train (fine-tuned) | train (other) | val | test |
|---|---|---|---|---|
| emotion | 0.973 | 0.939 | **0.752** | **0.762** |
| gesture | 1.000 | 1.000 | 0.997 | **0.864** |
| motion | 0.845 | **0.399** | 0.671 | 0.671 |
| context | 0.988 | 0.976 | 0.972 | 0.995 |

Designing against the train numbers would have assumed 0.94 emotion where
deployment delivers 0.76.

**(b) Motion is unusable and the costly emotion confusions were not the assumed
ones.** Motion scores 0.435 on honest footage — `standing→walking` 44%,
`stepping_back` recall 0.149 — and only 0.507 even when merged to
static/approach/retreat. The dominant emotion confusions are **Neutral→Sad
(0.146)** and **Disgust→Anger (0.125)**, not the Fear↔Surprise pair assumed from
the earlier test-set matrix.

The v3 rubric placed intent boundaries on **all three** of these unreliable
distinctions, and on motion alone in two cells.

## Phase 2 — the perception-aware taxonomy (the substantive win)

v4 keys intents on **gesture first** (most reliable), **emotion superclass** second
(`alarm={Fear,Surprise}`, `hostile={Anger,Disgust}`, `positive`, `low`, `neutral`),
**context** as a modifier, and **excludes motion entirely**. Critically, F07/F08
split on *gesture polarity* (thumbs-up vs thumbs-down) rather than the unreliable
anger-vs-disgust distinction.

| split | v4 ceiling | v4 rules | v3 rules |
|---|---|---|---|
| **val (gate)** | **0.9505** | **0.8269** | 0.6996 |
| test | 0.9237 | 0.7646 | 0.7439 |

Gate passed (ceiling ≥0.90, rules ≥0.80 on val). The taxonomy alone lifts the
rule baseline by **+0.127 on val** and raises the achievable ceiling.

## Phases 3–4 — fusion experiments (all on the v4 label space)

Self-attention over cue tokens, exhaustive v4 cue-combination augmentation (66 of
80 combinations, every test combination excluded), 3 seeds per arm.

| lever tested | outcome |
|---|---|
| representation: softmax-super / embeddings / two-stage | **tied at ~0.65** — embeddings are not disadvantaged |
| motion token | no effect (0.6520 → 0.6516) — confirms its exclusion |
| augmentation size 300 / 50 / **none** | **essential**: none collapses to 0.51 |
| clean (non-fine-tuned) augmentation pools | mixed: two_stage +0.026, softmax_super −0.066 |
| cue-dropout 0.4→0.0 + plain vs masked checkpoint selection | +0.01–0.04; masked selection over 3 tokens was over-optimising degraded accuracy |

**Best configuration reached: 0.6598 ± 0.0059** (softmax-super, no motion token,
augmentation 300/combo, dropout 0.0, plain val selection).

## Verdict

**Learned fusion does not beat the v4 rule cascade at this cue quality:
0.660 vs 0.719–0.765 on test.** This reproduces the v3 pattern (fusion ≈ or below
rules) and has a structural cause: the intent label is a deterministic function of
the ground-truth cues, so the rule cascade is Bayes-optimal given correct cues and
can only lose to perception error. A learner must both infer the cue class from
noisy features *and* apply the mapping, and it does the first step worse than the
deployed cue models do.

**0.80 is not attainable with the cue models frozen.** The measured arithmetic:
test cue accuracy is gesture 0.864 / emotion 0.762, which caps the rule baseline at
~0.72–0.76 and the ceiling at 0.92. Closing that needs better cues, and the per-cue
oracle analysis prices the options: **emotion +0.119**, gesture +0.059, motion
−0.013.

## Recommendation

1. **Adopt the v4 taxonomy regardless of the model choice.** It is the real
   contribution here: a rubric designed against measured perception reliability,
   lifting the achievable ceiling and the baseline substantially, with every
   boundary justified by a confusion measurement rather than intuition.
2. **Report fusion honestly against rules** and place the fusion claim on the axes
   where it is defensible — degraded-cue robustness, cue-conflict handling, and
   unseen-actor generalization — not on raw accuracy.
3. **To reach 0.80, unfreeze the emotion model.** It is the only lever with the
   measured value to get there (+0.119 to 0.803 at oracle), and its dominant real
   errors (Neutral→Sad, Disgust→Anger, Fear→Surprise) are exactly the ones the v4
   superclasses already partially absorb — so cue-model gains and the taxonomy
   compound rather than overlap.
