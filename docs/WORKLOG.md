# WORKLOG — cross-machine progress log

## 2026-08-08 (later) — [WIN-3060] — Embedding-level fusion: built end-to-end, tested four ways, all negative-to-neutral

**Trigger:** following the statistical-audit session (below), user asked about extracting
penultimate-layer embeddings from all four unimodal models (motivated by `PERCEPTION_BAND.md`
identifying perception, not fusion generalisation, as the dominant remaining error band) and
requested the full investigation through to a properly-tested conclusion.

**0. Extraction pipeline built and run** (`fusion/extraction/perframe.py::extract_embeddings_only`,
`windows.py`'s new embedding hooks, `scripts/54_extract_embeddings.py`,
`scripts/55_pool_embeddings.py`). All three learned models' penultimate embeddings are FREE — a
classifier's final `Linear` layer's *input* is the embedding, captured via a forward hook with zero
extra compute; CLIP's 512-d embedding was already computed and discarded before the text-similarity
step. Skipping MediaPipe Holistic entirely for the embedding-only pass (emotion's face crop uses a
separate cheap detector) cut a naive ~6h re-extraction estimate to ~1h actual. **2869/2869 clips
extracted, 1.8GB total** (vs the corrupted transfer attempt's 7.7GB of per-frame JSON — this stays
local, pooled at window/clip grain like everything else in this pipeline).

**Two real bugs caught before trusting the output** (both documented in the
`embeddings-corrupt-on-win3060`/`perframe-4k-face-detection-gotcha` memories): (1) a 4K clip with no
visible face cost 54 minutes in the face-detection fallback (228 more 4K clips were queued —
fixed with a resolution-scoped workaround, verified zero-effect below 3000px); (2) 2/2869 clips had
fully-NaN pooled emotion embeddings from a cross-pass face-detection mask mismatch (fixed, verified
zero NaN across all four modalities after the fix).

**1. Pipeline sanity + dimensionality risk** (`scripts/56_embedding_probe.py` →
`EMBEDDING_PROBE.md`). A linear probe on pooled embeddings recovers known argmax accuracy closely
for emotion/gesture (single-linear-layer classifier heads; motion's gap is expected, its head has an
extra ReLU+Linear a probe can't reproduce) — confirms the pipeline is correct. But **motion's
probe train/test gap (+0.286) is LARGER than gesture's (+0.151) or emotion's (+0.245) despite motion
never being fine-tuned on train** — ruling out fine-tune contamination as the sole explanation and
pointing at raw dimensionality: 2176 combined raw dims against ~1547 train clips.

**2. Plain (no augmentation) embeddings lose significantly**
(`scripts/57_embedding_fusion.py` → `EMBEDDING_FUSION.md`, `AttentionFusion` generalised to
arbitrary per-modality width via a new `modality_dims` constructor arg, fully backward-compatible).
PCA (32 components/modality, fit on train only) is required, not optional, given finding 1. 10
seeds: `probs_only` 0.5552±0.0245 > `embed_only` 0.5292±0.0291 (p=0.0014) > `embed_probs`
0.5104±0.0379 (p<0.0001) — concatenating hurts more than embeddings alone. A 2-seed smoke test had
shown embeddings AHEAD; reversed completely at 10 seeds, same lesson as the significance-testing
session below.

**3. Full recipe (recombination + dropout + jitter) ties** (`fusion/model/recombine_embed.py`,
`fusion/model/generic_train.py` — a generalised training loop since `WindowDataset`/`train_fusion`
are hardcoded to the 24-dim probability layout; `scripts/58_embedding_fusion_full.py` →
`EMBEDDING_FUSION_FULL.md`). Same 448-combo rubric as `recombine_merged.py`, in-fold pools (known
contamination, not yet fixed at this step). 10 seeds, paired via the identical training loop:
`embed_full` 0.6904±0.0070 vs `probs_full` 0.6993±0.0135 — **McNemar p=1.0000, an exact tie**
(n10=44, n01=45). Recombination rescues embeddings from "significantly worse" to "tied", but no
further.

**4. Out-of-fold pools (contamination fix) make it WORSE, not better** — the most informative
result. Built 2-fold held-out retraining for gesture+emotion only (`scripts/59_outfold_finetune.py`
— motion/context skipped, never contaminated per `POOL_PURITY.md`; folds split TRAIN v3_rows,
stratified by intent, persisted to `outfold_split.json`; reused scripts 34/36's exact recipes,
warm-started from deployed, output to new `*_fold{0,1}.pth` files, never touching deployed/promoted
checkpoints). `scripts/60_outfold_pools.py` scores each fold's held-out clips with the checkpoint
that never trained on them (gesture: free via `WindowFeaturizer(gesture_ckpt=...)`'s existing
override + hook; emotion: reuses cached crops, no video decode) — 99.9%+ coverage.
`scripts/61_embedding_fusion_outfold.py` re-ran the full-recipe comparison with these pools (PCA/
scaler kept fit on the DEPLOYED model's embeddings, unchanged, so real train/val/test scoring stays
consistent — only the recombination pool SOURCE changed). Result: `embed_full_outfold`
**0.6742±0.0112**, WORSE than both the in-fold version (0.6904) and `probs_full` (0.6993,
p=0.0544 borderline) and now significantly worse than rules (p=0.0480). Interpretation: each fold
checkpoint trained on only half the data is a weaker classifier, so "clean" out-of-fold pool vectors
are also noisier — contamination was propping the number up, not suppressing a hidden win.

**Verdict, all four tests:** embedding-level fusion never significantly beat the 24-dim probability
baseline in any configuration (plain: loses; full in-fold: ties; full out-of-fold: loses). Combined
with finding 1's dimensionality evidence, the defensible conclusion is that **raw penultimate
embeddings do not help this fusion architecture at this dataset's scale** — the bottleneck is
sample count relative to representation width, not pool contamination or lack of recombination.
`PERCEPTION_BAND.md`'s ~0.22 perception band remains open; this was one well-tested candidate lever
for it, ruled out rather than confirmed.

**Next (not done):** if revisited, the more promising directions are (a) a much lower-dimensional
embedding (8-16 components) or a supervised projection (e.g. LDA) instead of unsupervised PCA, since
variance-maximising PCA is not optimised for class separation the way the classifier's own softmax
already is; (b) more training data before richer representations pay off; (c) directly improving the
underlying cue classifiers (the actual perception fix) rather than exposing more of their internals
to fusion.


## 2026-08-08 (earlier) — [WIN-3060] — Statistical audit: the headline claim fails, and the gap decomposition reverses

**Trigger:** user restated the publication objective (fusion must beat rules; results >70%) and
pointed at newly extracted per-cue embeddings. Four investigations run in order; three of the four
findings are corrections to things the project currently asserts.

**0. The embeddings are unusable as copied.** `fusion/fusion-engine-embeddings/` (7.7 GB, four
JSONL dumps: emotion 1280-d, gesture 128-d, motion 256-d, context 512-d) is a valid JSON prefix
followed by raw binary. Salvageable: emotion 4.8% (~178 of 2,904 clips), gesture 30.4%, motion
8.9%, context 2.5%. Run logs on the HPC end `batch done: 2904 clips processed`, so this is a
transfer failure, not an extraction bug. Re-copy from
`/home/hri_multimodal/Downloads/fusion-engine/data/features/embeddings/` with checksums, ideally
re-dumped as float16 `.npy` (~40x smaller than this JSON). **User elected to park embedding work**
— but finding 2 below is a direct argument to revisit that.

**1. The headline fusion-vs-rules claim is not significant** (`scripts/49_significance.py` →
`SIGNIFICANCE.md`; stats helpers factored into `scripts/realworld_eval/stats.py`).
`SCENARIO_TEST_REPORT.md` asserted "Fusion beats rules by 0.0071 acc — the G1/T05 claim holds" on
a 3-seed mean of 0.7191 ± 0.0164 — a margin under half its own std. At **10 seeds**:

| | Clip acc | Macro-F1 |
|---|---|---|
| Fusion, 10 seeds | 0.7133 ± 0.0160 | 0.6146 |
| Fusion, 10-seed majority vote | 0.7130 | 0.6140 |
| Rules | 0.7120 | **0.6414** |

McNemar exact (ensemble vs rules) **p=1.0000** (n10=73, n01=72); bootstrap over clips +0.0009,
95% CI [−0.0235, +0.0255] containing zero; **4 of 10 seeds favour rules**, 1 of 10 beats them
significantly. The 3-seed figure was a small-sample artifact — seeds (0,1,2) happen to include
seed 0 (0.7416), the best of ten and the only significant one. **Rules also beat fusion on
macro-F1**, the metric that matters for rare/safety intents. `SCENARIO_TEST_REPORT.md` corrected
in place with the superseded number left visible.

**1b. No comparison this project has made is resolvable at 3 seeds.** With the measured σ=0.0160,
3 seeds give a 95% CI half-width of ±0.0397, so the smallest resolvable difference is ~0.079.
Every claimed gap is far below it: self-attn vs GMU 0.0126, vs cross-attn 0.0201, vs concat-MLP
0.0354, window x0.5 vs x1.0 0.0061, R1 vs R2 0.0082, fusion vs rules 0.0071. The architecture
roster's whole spread (0.7191→0.6581 = 0.061) is under the threshold. Written up as
`docs/methodology/06_fusion_model.md` §6.10. **Fix: compare configs paired by seed** (common seed
effects cancel) and persist per-seed values — Studies 1–3 saved only aggregated mean/std, so they
cannot be re-analysed without re-running.

**2. The gap decomposition REVERSES on the deployed recipe** (`scripts/50_perception_band.py` →
`PERCEPTION_BAND.md`). `GAP_DECOMPOSITION_MERGED.md`'s "generalisation 0.381 dominates perception
0.072" was measured on the *plain* model and has been stale since recombination landed — the
2026-08-03 entry flagged this re-run as a to-do and it was never done. On the `full` recipe,
10 seeds:

| Cues | Fusion | Rules |
|---|---|---|
| real | 0.7133 ± 0.0151 | 0.7120 |
| oracle one-hot | **0.9373 ± 0.0216** | 1.0000 |
| oracle realistic (true class, real soft vector) | 0.8176 ± 0.0156 | 0.8233 |

fusion+oracle moved **0.619 → 0.9373**: generalisation cost **0.381 → 0.063**, perception cost
**0.072 → 0.224**. Cross-checked — under the realistic-oracle condition rules cannot generalise
wrongly by construction, so their 0.1767 shortfall is pure residual noise and fusion is only
+0.0057 worse. **Recombination did its job; perception is now the shared bottleneck of both
systems.** Implication: fusion-architecture work has ~0.06 of headroom left and is a dead end;
the ~0.22 perception band holds everything remaining.

**3. T02's `point` collapse diagnosed and partly fixed** (`fusion/model/recombine_merged.py`
gained `allocate()`; `scripts/51_recombination_balance.py` → `RECOMBINATION_BALANCE.md`). Row #27
scored 0.038 vs rules' 0.642 with **clean cues** (anger 0.80, point 0.65, walking 0.81), ruling
out perception. Cause: constant `n_per_combo` weights cue COMBOS equally, so each intent's share
follows its combo count — skewed globally (F01 104/448, F10 8/448) and, decisively, *within a cue
family*: `Anger+point` is F07 6/8 vs **F06 2/8**, `Happy+point` is F05 6/8 vs **F03 2/8**. Both
collapsing rows are the 25% minority branch. Four allocation modes, 5 seeds, equal synthetic
budget, **mode selected on VAL macro-F1**:

| Mode | Val macro-F1 | Test acc | Test macro-F1 | row #27 | row #55 |
|---|---|---|---|---|---|
| `uniform` | 0.7541 | 0.7134 ± 0.0175 | 0.6137 | 0.038 | 0.357 |
| `sqrt` | 0.7678 | 0.7216 ± 0.0072 | 0.6220 | 0.019 | 0.476 |
| **`intent`** ←selected | **0.7838** | 0.7299 ± 0.0139 | 0.6346 | 0.057 | 0.571 |
| `family` | 0.7589 | 0.7354 ± 0.0113 | **0.6425** | 0.076 | 0.643 |
| rules | — | 0.7120 | 0.6414 | 0.642 | 0.571 |

Monotonic improvement in the predicted direction confirms the mechanism (`family` equalises the
*conditional* prior to 1:1.00 where `intent` only reaches 1:2.80). **`family` scores best on test
and is the only mode to exceed rules' macro-F1 — but `intent` won on val, and swapping to
`family` because it looks better on test would be tuning on test.** Reported as selected=`intent`;
`family` needs a pre-registered re-test (more seeds, or a fresh val split) before adoption.
Selected mode vs rules: McNemar p=0.1807, favours fusion, bootstrap CI still contains 0. **Row #27
is NOT fixed** (0.076 vs rules' 0.642) — rebalancing moves it monotonically but recovers only a
fraction; F06 has a residual cause, see finding 5.

**4. T04 is structurally near-vacuous on this rubric** (`scripts/52_context_counterfactual.py` →
`CONTEXT_COUNTERFACTUAL.md`). The row-pairing T04 table has **20 of 30 rows at n=0** test clips.
Replaced with a counterfactual every clip can take: keep real emotion/gesture/motion, swap only
the context cue for a real vector from the other room, ask whether the prediction moves as the
rubric says. Structural result: **context changes the intent for 24 of 224 (emo × ges × mot)
tuples (10.7%), every one of them `raise_hand`** — so 89% of any T04 measurement is an invariance
test passable by ignoring context. In the test split the flip subset is n=52, all from row #25,
all classroom. Independently agrees with two earlier findings (Phase-1 context swap changed 0
clips; cue attribution ranks context last on every intent). Fusion invariance 0.9946 vs rules
1.0000. **Caught a misleading metric before it became a claim**: `flip_followed` reads fusion
0.827 vs rules 0.404, but row #25's kitchen answer is F01 — the majority class — so an F01-biased
model scores well without tracking context at all (the tell: `flip_changed_at_all` exactly equals
`flip_correct_before`). Added `flip_correct_both` (right in BOTH rooms), the metric that cannot be
won by class bias: **fusion 0.3077, rules 0.1923**. Fusion is genuinely better but both are poor.

**5. Recombination pools are not representative of deployment** (`scripts/53_pool_purity.py` →
`POOL_PURITY.md`) — the session's most consequential finding, surfaced while chasing F06. Pools
are built from the TRAIN split, and emotion+gesture were **fine-tuned on that split** (promoted
2026-08-04). Purity (fraction of a GT bucket the cue model argmaxes correctly):

| Cue | Fine-tuned? | Train | Val | Test | Train−Test |
|---|---|---|---|---|---|
| gesture | yes | **1.000** | 0.997 | 0.864 | +0.136 |
| emotion | yes | 0.959 | 0.752 | 0.762 | +0.197 |
| motion | **no** | 0.655 | 0.671 | 0.671 | −0.016 |

Motion — the one model never promoted — has the only honest pool. **Fusion is therefore trained on
gesture cues that are never wrong and tested on cues wrong 13.6% of the time.** This independently
predicts the cue-attribution ordering (gesture 31.8% > emotion 29.5% > motion 17.6% > context
10.1%): the model weights cues by how clean their training pools were, not by how reliable they
are. Worst case `raise_hand`: train 1.000, **test 0.327** — fusion has never seen a mistaken
raise_hand vector. Note **val pools are not the fix** (gesture val purity is still 0.997 — val is
actor-disjoint but shares scenarios); the sound fix is out-of-fold prediction (refit each cue model
K times holding out scenario groups, pool the held-out predictions), or calibrated per-class noise
injection.

**Net position for the thesis:** the ">70%" bar is met (0.7133, and human-level by field
standards — humans score 71% on MIntRec2.0). "Fusion beats rules on accuracy" is **not**
supportable and should be retired in favour of the axes where they diverge (T03 missing-cue, F02
recall) plus the now-quantified fact that both are perception-limited. The defensible new
contributions from today are the recombination **prior** and **pool-realism** analyses — both are
general findings about rubric-driven augmentation, not artifacts of this dataset.

**Next (not done):** re-test `family` allocation pre-registered at 10 seeds; out-of-fold pools;
re-run Studies 1–3 paired-by-seed; re-transfer the embeddings given finding 2.


## 2026-08-07 — [WIN-3060] — Conflict-holdout: the honest generalization number

**Trigger:** user correctly identified that T02 (`SCENARIO_TEST_REPORT.md`) never actually tested
generalization — recombination covers all 448 combos, so fusion had a synthetic labelled example
of every "conflicting" combo it was then tested on. Proposed excluding conflicting combos from
training entirely and testing zero-shot. Confirmed correct, built it.

**Method** (`fusion/model/recombine_merged.py::classify_combos`, discussed and refined with the
user): a combo is CONFLICTING if its rubric intent differs from the same (context,gesture,motion)'s
intent under emotion='Neutral' — i.e. emotion measurably overrides the gesture's default reading.
164 conflicting combos identified, reviewed against the user's own judgement via an interactive
artifact-based table (all 164 rows, grouped by gesture) — confirmed matching their intuition.

**v1 (failed, informatively):** excluded ALL 164 conflicting combos from BOTH synthetic
recombination AND the 753 real training clips whose true tuple happened to be conflicting.
Result: exactly 0.0% accuracy on every conflicting test — clean, but a red flag (a model that's
merely *bad* at generalizing still gets some right by chance). Diagnosed: F02, F07, F08, F10 (4 of
9 intent classes) are produced EXCLUSIVELY by an emotion overriding a gesture's default — they have
no "calm/neutral" pathway anywhere in the rubric. So v1's `aligned_only` model had literally zero
training exposure to 4 of 9 possible answers; 0% measured "can you name a class you were never
shown," not compositional generalization. A genuine finding about the rubric's structure, but not
the intended experiment.

**v2 (the real result), per the user's explicit correction:** recombination is allowed to
reinforce any conflicting combo already present in real TRAINING data (not new information, just
more synthetic repeats of a pattern already there); real training runs unfiltered. Only combos
with ZERO presence anywhere in training — 143 of 164, verified to cover all 9 classes and give
healthy test coverage (372 vs 401 real test clips) — are truly held out.

| Test set (real, headline) | n | `restricted` (never saw combo) | `full_reference` (saw it) | rules |
|---|---|---|---|---|
| Novel combo | 372 | **0.2849** | 0.6828 | 0.7634 |
| Seen combo | 401 | 0.818 | 0.7955 | 0.7257 |

**The generalization gap is large: +0.3979 accuracy (0.2849 → 0.6828) between never-having-seen a
combo and having seen it.** This is the honest number the user's original critique was pointing
at: fusion's conflict-resolution ability on `T02`/`SCENARIO_TEST_REPORT.md` comes overwhelmingly
from recombination teaching it the specific answer, not from an independently-learned compositional
rule. 28.5% is better than the ~10% floor for a 10-class problem (some transfer is happening — not
zero), but nowhere near what direct supervision achieves. This CONFIRMS the user's hypothesis from
the "is this pure research" discussion, now with real (non-circular) evidence rather than
conceptual argument alone.

**Secondary findings**: rules score 0.7634 on the same novel-combo clips — NOT 1.0, because this
uses REAL (perception-noisy) cues, not oracle ones; the gap from 1.0 reflects perception error on a
harder subset (fear/anger/disgust-heavy scenarios), consistent with everything already established
about rules+real vs rules+oracle. On SEEN-combo clips, `restricted` (81.8%) edges out
`full_reference` (79.6%) — plausibly because its smaller combo set gets more synthetic repetitions
per combo (30,500 samples / 305 combos vs 44,800 / 448) — a specialization effect, not a
contradiction.

**Bottom line for the thesis**: the "fusion beats rules on conflict resolution" claim from T02
needs to be qualified sharply — it holds for combos fusion was taught (a real, if narrower, result:
`restricted` still beats rules 81.8% vs 72.6% here), but fusion's ability to handle a genuinely
novel conflicting combination it was never shown is weak, well below both the taught-case number
and rules' perception-limited-but-structurally-complete performance. Recombination teaches;
it does not by itself confer independent reasoning.


## 2026-08-07 (later) — [WIN-3060] — Phase 2 completes: real video degradation, resolved

**Did:** re-ran Phase 2 (`scripts/44_video_degradation.py`) after the user restarted the machine to
free memory — the two prior attempts (2026-08-06) both failed on this exact step (memory
exhaustion loading the CLIP context model, `OSError: paging file too small`). Confirmed free RAM
went from ~2GB to ~7.3GB before relaunching. **This time it worked**: 8/8 conditions completed
(7 cleanly, 1 — `downsample_0.35` — hit a one-off shell-level exit 127 unrelated to the earlier
memory bug, re-ran individually without issue).

**Result: real pixel degradation reinforces Phase 1's finding, does not overturn it.**

| Condition | Fusion | Rules |
|---|---|---|
| Clean | 0.6984 | 0.7778 |
| Light/heavy blur | 0.6349 / 0.4603 | 0.6667 / 0.5397 |
| Light/heavy darkness | 0.6667 / 0.5556 | 0.7619 / 0.6349 |
| Light/heavy downsample | 0.6508 / 0.4921 | 0.6349 / 0.5079 |
| JPEG compression | 0.5714 | 0.5714 |

Rules ahead in 6/8 conditions, tied in 1, fusion ahead in 1 (downsample light, +0.016, n=63 — too
small a sample/margin to claim). **Second negative result (after Phase 1 part B) for the "fusion
is more robust to degraded input" hypothesis** — genuine pixel corruption tells the same story as
simulated probability noise did.

**One honest, useful side-finding**: observation rates stayed at 1.0 across nearly every
condition — even heavy blur/downsampling — meaning the perception stack still found *something*
to read, just less accurately; only the heaviest darkness condition actually dropped observation
below 100% (emo 0.984, ges/mot 0.937). This is the first REAL (not scenario-designed, not
simulated) missing-cue event produced anywhere in this investigation — contrast with the
`SCENARIO_TEST_REPORT.md` finding that V3's "designed-missing" rows stayed 100% observed despite
being scripted as absent.

**Updated overall verdict, all phases now complete**: across Phase 1 (context/degradation/safety),
Phase 2 (real degradation), and Phase 3 (sequence recombination), fusion did not demonstrate a
general robustness or accuracy advantage over rules — 4 of 5 tests came back negative or tied.
The two genuine, well-evidenced fusion advantages remain unchanged: **missing-cue robustness**
(T03a, `SCENARIO_TEST_REPORT.md`: 46.4% vs 28.2% with gesture masked) and the **F02 safety fix**
(2026-08-07 earlier entry: class-weighted loss now beats rules' emergency recall). That is the
complete, honest, defensible set of claims from this investigation.


## 2026-08-07 — [WIN-3060] — F02 recall fix (concrete win) + cue attribution (CAM-analogue)

**Did:** two follow-ups to yesterday's mostly-negative Phase 1-3 investigation, targeting the one
identified, actionable weakness and the interpretability gap.

**F02 recall fix** (`fusion/model/train.py` gained a `class_weights` param on `CrossEntropyLoss`;
`scripts/46_f02_recall_fix.py`). Phase 1 found fusion's F02 (emergency) recall CEILING across its
entire threshold sweep (0.645) never reached rules' fixed-point recall (0.727) — a training-time
deficiency, not a thresholding problem. Isolated intervention: only F02's loss weight changes,
every other class stays at 1.0. Swept {1.0 (control), 2.0, 3.0, 5.0}, 3 seeds each:

| F02 weight | Headline acc | F02 recall ceiling | Beats rules' 0.727? |
|---|---|---|---|
| 1.0 (baseline) | 0.7191 ± 0.0164 | 0.645 | no |
| 2.0 | 0.7102 ± 0.0067 | 0.719 | no (close) |
| **3.0** | 0.7045 ± 0.0093 | **0.736** | **yes** |
| 5.0 | 0.7031 ± 0.0175 | **0.818** | **yes** (bigger margin, but recall @ rules' own precision level is actually LOWER than weight=3.0's — 0.678 vs 0.736 — so the higher ceiling comes with more false alarms, not a strictly better operating point) |

**Concrete, well-evidenced win**: weight=3.0 or 5.0 both close the Phase 1 safety gap for a modest
1.5-2 point headline-accuracy cost. **Recommend weight=3.0 as the better default** — smallest
accuracy cost that still crosses rules' recall, and better recall-at-matched-precision than
weight=5.0. weight=5.0 is the choice if you'd rather maximise the ceiling and accept more false
alarms. Not yet promoted to the deployed checkpoint — this was a diagnostic sweep, not a
final-model decision; see `docs/DECISIONS.md`.

**Cue attribution** (`fusion/model/attribution.py`, `scripts/47_cue_attribution.py` ->
`CUE_ATTRIBUTION.md`) — the CAM-analogue requested for the fusion architecture zoo. Literal Class
Activation Mapping doesn't apply (needs spatial feature maps; the input is a 24-dim vector with no
spatial extent) — built attention-weight extraction instead, which is the faithful transplant of
the same idea (the CLS token's attention onto each modality token IS the model's own mixing
coefficient, no approximation needed). Required manually replicating `TransformerEncoderLayer`'s
`norm_first` forward, since `need_weights=False` is hardcoded on its internal fast path and isn't
retrievable via a hook. Smoke-tested: masked (obs=0) modalities correctly get exactly 0 attention.

Results: overall attention gesture(31.8%) > emotion(29.5%) > motion(17.6%) > context(10.1%).
**Notable cross-validation**: context gets the lowest attention on EVERY intent (7.7-12%) —
independently rediscovering, via a completely different method, what code-reading `rule_intent()`
already showed in Phase 1 part A (context only matters in one branch). Per-intent breakdown is
plausible throughout (F02/emergency elevates motion to 22.4%, above its 17.6% average — consistent
with fleeing/step-back motion being diagnostic; F07/anger has the highest emotion weight of any
class at 37.5%). Qualitative same-gesture-different-emotion spotlight table came out thin (only
2/10 target rows had headline test clips — the same test-split coverage limit T04 hit in
`SCENARIO_TEST_REPORT.md`), reported honestly rather than padded.

**Overall verdict on the full 2026-08-06/07 investigation**: the sweeping "fusion beats rules on
accuracy" story is not supported by today's evidence (Phase 1 A/B negative, Phase 3 a tie). What
IS now solidly evidenced: fusion wins clearly on missing-cue robustness (T03a,
`SCENARIO_TEST_REPORT.md`: 46.4% vs 28.2% with gesture masked), and — after today's fix — fusion
has a genuine, tunable safety capability rules structurally cannot have (adjustable recall/
precision tradeoff for emergency detection, now demonstrably beating rules' fixed point). That is
the defensible thesis claim going forward, not blanket superiority.


## 2026-08-06 — [WIN-3060] — "Why does fusion only tie rules" investigation (Phases 1-3)

**Trigger:** user flagged that the project's core claim — learned fusion beats rule-based
reasoning — wasn't landing: headline accuracy is fusion 0.7191 vs rules 0.712, a near-tie, not
the decisive win the thesis narrative assumed.

**Root-cause diagnosis (before any new code):** every clip's intent label comes from its V3
scenario row, and that row's intent is `rule_intent(true_cues)` — i.e. the rubric wrote the
labels. A rule system given true cues is therefore Bayes-optimal by construction (confirmed
independently: rules+oracle = 1.000, exactly the ceiling, `GAP_DECOMPOSITION_MERGED.md`). Rules
can only lose points to *perception* error, never reasoning error. Recombination then trains
fusion on `rule_intent()` labels — textbook knowledge distillation with rules as teacher. Clean-
data accuracy was never a fair arena for "fusion beats rules"; the real question is where a
hand-written argmax rubric structurally cannot compete.

**Phase 1 — robustness battery** (`scripts/43_robustness_battery.py` → `ROBUSTNESS_BATTERY.md`),
three tests:
- **A. Fair context**: `rule_predict` hands rules the clip's TRUE context; fusion must infer it.
  Built a predicted-context variant. Result: **0 clips changed** — `rule_intent` branches on
  context in exactly one place (`raise_hand`), so the "advantage" is real but structurally
  worthless. Hypothesis closed, cleanly negative.
- **B. Degradation sweep**: Gaussian log-prob noise (sigma 0→2.0) applied identically to fusion
  and rules. Expected rules' `argmax` to break down faster than fusion's soft weighting. Result:
  **mixed, not decisive** — roughly tied through moderate noise, fusion only ~1-2 points ahead at
  the highest noise levels. Does not support a clean "fusion is more robust to noise" claim.
- **C. F02 safety operating point**: rules are a single fixed point (P=0.481, R=0.727, not
  adjustable); fusion emits a probability, so its threshold is tunable. Result: **fusion's recall
  ceiling across the ENTIRE threshold sweep (0.645) never reaches rules' recall (0.727)** — the
  "tunable safety dial" argument does not hold empirically. This is a genuine, specific fusion
  weakness (the model itself under-predicts F02), consistent with F02 rows failing even under
  ORACLE cues in the gap decomposition and fusion losing to rules on the F02-missing-gesture rows
  in `SCENARIO_TEST_REPORT.md`. Caught and corrected an overclaiming first-draft conclusion in the
  report before it became a wrong thesis claim.

**Phase 2 — real video degradation** (`scripts/44_video_degradation.py`) — **PAUSED, unresolved.**
Corrupts actual video pixels (blur/dark/downsample/jpeg) before Holistic/face-detection/CLIP run,
re-extracting through the real perception stack rather than simulating cue noise. Two attempts,
both failed:
1. Original single-process design leaked memory across ~430 sequential clip decodes (no release
   point for MediaPipe/OpenCV/CUDA state), crashed via OOM→segfault, and lost all 3 completed
   conditions' results because they were only written to disk at the end (a real bug — fixed for
   future use, but too late for that run).
2. Rewrite ran each of 8 conditions as an isolated subprocess with incremental result-writing
   (fixes the data-loss bug) — but ALL 8 failed immediately on `OSError: paging file too small`.
   Diagnosed: this machine had ~2GB free RAM of 16GB at the time, with 5 orphaned Python processes
   left over from attempt 1 still holding memory. This is a genuine environment/resource
   constraint, not a code defect. User chose to deprioritize rather than debug further right now.
   Script is fixed and ready to resume on a machine/moment with more headroom.

**Phase 3 — trajectory recombination for R3** (`fusion/model/recombine_sequences.py`,
`scripts/45_sequence_recombination.py` → `SEQUENCE_RECOMBINATION.md`). The one experiment that can
genuinely exceed rules on accuracy: rules see one pooled snapshot and cannot use temporal
dynamics at all. Study 2 (2026-08-05) found R3 scoring far below R1 (0.533 vs 0.719) but flagged
it as confounded — R3 never got a recombination analogue. Built one: for each of 448 combos,
stitch a synthetic per-window TRAJECTORY from four real per-clip window sequences (one per
modality, ground-truth indexed, independently resampled to a shared target length via
`uniform_indices`) rather than the pooled version's single mean vector. 44,800 synthetic
trajectories, 21/21 class bins covered.

**Result: R3 causal jumps from 0.5329±0.027 to 0.7136±0.017 (+0.181)** — closes nearly the ENTIRE
augmentation gap. But the delta vs R1 (0.7191) is -0.0055, **within seed noise — a tie, not a
win**. Confirms the Study 2 confound diagnosis was correct (coverage, not order-awareness, was
the dominant factor) but does NOT establish that this architecture exploits temporal order beyond
what pooling already captures. Bidirectional (offline ceiling) scored 0.7031±0.005, slightly
BELOW causal — a mildly counterintuitive result attributed to the different readout mechanisms
(causal reads one strong final-position token; bidi masked-mean-pools), not investigated further.

**Overall verdict on the day's investigation:** three of four executed tests (A, B, Phase 3) came
back negative-to-neutral on the "fusion decisively beats rules" question; one (C, F02 safety)
surfaced a genuine, specific, and actionable fusion weakness. This is reported as a real result,
not spun positively — the project's honest finding is that on rubric-generated labels, learned
fusion's advantage over rules is narrow and situational (strongest on gesture-missing robustness,
per the earlier `SCENARIO_TEST_REPORT.md` T03a sweep: 46.4% vs 28.2% with gesture masked), not the
sweeping "fusion is better" story originally hypothesized.

**Concrete next step identified, not yet built:** F02 recall looks specifically fixable via
class-weighted loss during fusion training (the same intervention that worked for emotion/gesture
unimodal fine-tuning, `unimodal-finetune-promotion` memory) — a scoped, plausible fix, not
evaluated here.


## 2026-08-05 — [WIN-3060] — Fusion architecture zoo + temporal representation study

**Did:** built and ran three studies answering "what improves the fusion result further,
and clip vs window for Jetson deployment" (user request, following the promotion cycle below).

**New code:** `fusion/model/fusion_zoo.py` (GMU, LMF, cross-attention, channel-attention/CAM —
all share `AttentionFusion`'s `forward(x[B,24], obs[B,4])` contract), `fusion/model/gbt.py`
(LightGBM head), `fusion/model/sequences.py` + `fusion/model/temporal.py` (window-SEQUENCE
representation + causal/bidirectional temporal transformer, built straight from
`unimodal_windows.parquet`'s per-window rows — no re-extraction needed), `fusion/model/train.py`
gained a `model_factory` param so every zoo head reuses the identical augmentation/selection loop.
`lightgbm` installed into `.venv` (no protobuf dependency, safe with the pinned mediapipe/onnx).
Scripts: `38_fusion_architectures.py`, `39_temporal_representation.py`, `40_window_size_sweep.py`.

**Two real bugs found and fixed during Study 2** (documented since they'd have produced a
confidently-wrong deployment recommendation if missed):
1. **Causal-attention collapse.** `TemporalFusion`'s causal model always converged to a
   single-class prediction (val_acc identical to 4 decimals across all seeds). Root cause:
   sequences are right-aligned (padding at the start, real data at the end, so the last position
   is always the most recent real window — needed for zero-retrain trailing-K eval). Combining the
   causal triangular mask with the padding key-mask meant every PADDING query position had zero
   visible keys (causal blocks future, padding-mask blocks the only keys still in range) →
   NaN-adjacent degenerate attention at those positions → corrupted the real readout position via
   later self-attention layers. Fixed by dropping the key-padding mask entirely for the causal
   path — the causal mask alone guarantees every query sees at least itself, and only the
   guaranteed-real last position is ever read.
2. **Headline-eval mismatch.** R3's test split was built from the raw 1003-row test split, not
   the 979-row `headline_eval` subset R1/R2 use — an apples-to-oranges comparison. Fixed by
   filtering `real` to headline-only test rows before building R3's sequences.

**Study 1 — architecture ablation** (`FUSION_ARCHITECTURES.md`, clip-pooled input, `full` recipe,
3 seeds): incumbent self-attention stays best (0.7191, beats rules 0.712). **GMU is the standout
secondary finding** — 0.7065 (statistically close to the incumbent) with **8x fewer parameters**
(8,970 vs 70,090) and far tighter seed variance (±0.0024 vs ±0.0164) — a strong Jetson candidate
if the last ~0.01 acc isn't worth 8x the params. LMF 0.6994, cross-attention 0.699, GBT 0.684 (no
architecture engineering at all, notably strong), concat-MLP floor 0.6837, channel-attention/CAM
weakest at 0.6581. **Confirms the gap decomposition's prediction: fusion capacity was never the
bottleneck** — architectures cluster within ~0.06 of each other once recombination is applied.

**Study 2 — temporal representation** (`TEMPORAL_REPRESENTATION.md`): R1 clip-pool (mean over ALL
of a clip's windows) wins at 0.7191. R2 (train+predict per-window, majority-vote to a clip
decision) close behind at 0.7109 — re-confirms `07_evaluation.md` §7.2's 2026-07-28 rule
(clip-mean beats per-window-vote), but the margin shrank from 0.032 to 0.008 now that recombination
augments BOTH. R3 (order-aware sequence transformer, bidi and causal) scored much lower (~0.533)
— **but this is confounded, not a clean verdict against temporal modeling**: R3 has no
recombination analogue (out of scope for v1), so it trains on ~30x fewer augmented samples than
R1/R2; its high val-acc (~0.86-0.90) next to a much lower headline-test score is the same
real-vs-synthetic coverage gap recombination was built to close for R1. The causal trailing-K
buffer sweep (K=4/8/16/40 windows = 1.07/2.13/4.27/10.67s, same model, isolates buffer length only)
showed accuracy flat from K=8 through the full buffer — **short buffers suffice**, informative
regardless of the R3 confound.

**Study 3 — window-size (lookback span) sweep** (`WINDOW_SIZE_SWEEP.md`, re-run of §7.9's old
sweep on `final_merged` + recombination, 3 seeds vs the original's 1): confirms **shorter spans
are still better** — x0.5 (1.07s/1.0s gesture/motion span) scores 0.7252, beating deployed x1.0's
0.7191; x1.5/x2.0 progressively worse (0.7004/0.6782). Same direction as the old table's dramatic
0.976→0.756 collapse, but a much gentler slope — recombination stabilised the whole span range.

**Deployment recommendation:** keep self-attention + clip-pool (R1) as the shipped architecture —
best-verified, best-augmented; GMU is the one worth a second look if Jetson params/latency get
tight. Shorten the gesture/motion lookback span toward x0.5 (validated twice now, old table and
this one). R3 (temporal sequence modeling) is NOT ruled out — revisit only after a recombination
analogue exists for sequences.

**Next (not done, flagged as follow-up):** build a recombination-for-sequences approach (synthetic
window trajectories, not just synthetic pooled vectors) before drawing any final conclusion on
whether order-aware temporal modeling helps.


## 2026-08-04 (latest) — [WIN-3060] — Promotion complete: re-extraction + fusion retrain closes the loop

**Did:** completed the 3-step plan from the previous entry.
1. **Promoted** `finetuned_MobileNetV2_merged.pth`→`finetuned_MobileNetV2.pth` (emotion) and
   `best_TCN_finetuned_merged.pth`→`best_TCN.pth` (gesture) as the deployed checkpoints (backed up
   originals as `*_pre_merged_backup.pth`); synced `jetson_deploy/` copies; regenerated
   `docs/checkpoint_manifest.sha256` (7/7 `OK`). Motion left untouched (see prior entry — 4/4
   fine-tune attempts regressed).
2. **Refreshed emotion's per-frame cache** (`scripts/37_refresh_emotion_cache.py`) — emotion's
   Pass-1 cache stores model *output* (`emotion_probs`), not raw features, so pointing
   `WindowFeaturizer` at the new checkpoint (sufficient for gesture, which caches raw keypoints)
   would NOT pick up the fine-tune; had to fully re-decode all 2869 clips and re-run face
   detection + the new checkpoint per-frame. 2869/2869 refreshed, 0 failures (~50 min).
3. **Rebuilt** `unimodal_windows.parquet` (`scripts/28_merged_unimodal_eval.py --rebuild-windows`)
   with both promoted checkpoints — confirms the fine-tunes took effect (emotion/gesture numbers
   moved, motion/context unchanged as expected).
4. **Re-ran gap decomposition** (`scripts/29_merged_gap_decomposition.py`) and the **recombination
   fusion ablation** (`scripts/30_merged_recombination.py`, unchanged 4-way config,
   `n_per_combo=100`) on the refreshed features.

**Result — the unimodal gains reach the intent-prediction number.** Rules+real (the perception
proxy) headline acc rose 0.608→**0.712** (macro-F1 →0.6414) purely from better emotion/gesture
cues, no fusion retraining needed for that number. Fusion+real (plain, no recombination) rose
0.475±0.034→**0.5475±0.025** in lockstep. Recombination's advantage is fully preserved on the
refreshed features and the **`full` config now nearly matches/exceeds the new rules+real ceiling**
(0.7191±0.016 acc vs 0.712 ceiling — the G1 claim holds even more cleanly than before, where `full`
was still ~0.007 short of ceiling).

| Config | Clip acc (headline, 3 seeds) | Clip macro-F1 |
|---|---|---|
| plain | 0.5475 ± 0.025 | 0.438 ± 0.023 |
| augmented | 0.5264 ± 0.027 | 0.4267 ± 0.037 |
| recomb_only | 0.7085 ± 0.008 | 0.6147 ± 0.009 |
| **full** | **0.7191 ± 0.016** | **0.6241 ± 0.023** |

Gap decomposition breakdown (headline): ceiling 1.0, rules+oracle 1.0 (still exact), fusion+oracle
0.619±0.048 (essentially unchanged — oracle-cue generalisation gap is a fusion-model property, not
a perception one), rules+real 0.712, fusion+real (plain) 0.5475. Generalisation cost
(ceiling−oracle) stays 0.381; perception cost (oracle−real) shrank from the old baseline as
expected since emotion/gesture perception improved.

F02 (emergency) rows #23/#53/#54 remain the hardest — recombination lifts them to 0.36–0.66 mean
hit rate (was 0%), consistent with the pre-promotion run; still the weakest intent class and worth
flagging as a known limitation, not something this promotion cycle was expected to fix (gap
decomposition shows oracle cues ALSO fail on these rows — it's a fusion-generalisation problem,
not a perception one).

**All artifacts regenerated:** `results/realworld_eval_merged/UNIMODAL_final_merged.md`,
`GAP_DECOMPOSITION_MERGED.md`, `RECOMBINATION.md`; MLflow runs logged under `03_diagnostics`
(gap decomposition + recombination configs) and `02_fusion` (already had the emotion/gesture
fine-tune runs from the previous entries).

**This closes the "fine-tune uni models on complete dataset → promote → propagate to fusion"
pipeline the user requested.** Final combined numbers reported to user in-chat.


## 2026-08-04 (later still) — [WIN-3060] — Motion retry (loss-only) also fails: the sampler hypothesis is wrong

**Did:** retried motion fine-tuning with the `WeightedRandomSampler` REMOVED (class-weighted
loss only, matching emotion/gesture exactly), warm-started from deployed, selecting on val
macro-F1. Discovered while editing: the ORIGINAL script had the sampler AND class-weighted loss
**simultaneously** — double-correcting the imbalance, more aggressive than either alone, and
worse than I'd realised when proposing the retry.

**Result: still a regression on test.**
| Metric | Deployed | Sampler+loss (first attempt) | Loss-only (this attempt) |
|---|---|---|---|
| Headline test acc | 0.665 | 0.560 | **0.561** |
| Headline test macro-F1 | 0.589 | 0.487 | **0.491** |
| `raw_take` acc | 0.635 | 0.536 | **0.538** |

Essentially identical to the sampler-based result. **This overturns the loss-weighting
hypothesis for motion specifically** — across 4 combinations now (2 warm-starts × 2 balancing
strategies), motion fine-tuning regresses on genuine test performance every time, while val
accuracy is consistently ~80%+ in all 4. The common factor is not the sampler; it is that
motion's actor-disjoint val set (P03+P04, same scenarios as train) does not predict test
performance (new scenarios, mostly different actors) — a scenario-generalisation gap specific
to this modality, not a training-recipe bug.

**Decision: motion is NOT promoted.** Deployed `best_model_finetuned.pt` stays authoritative.
Documented as a genuine negative result with strong replication (4/4 attempts failed
consistently) — worth keeping in the thesis as evidence that "more real data, same recipe"
does not universally help, unlike emotion and gesture where it clearly did.

**Final verdict, all three unimodal fine-tunes:**
| Model | Promoted? | Headline test delta |
|---|---|---|
| Emotion | ✅ yes | +0.093 acc / +0.081 F1 |
| Gesture | ✅ yes | +0.101 acc / +0.094 F1 |
| Motion | ❌ no (kept deployed) | −0.10 to −0.14 acc across 4 attempts |

**Next:** promote emotion+gesture checkpoints (update `checkpoint_manifest.sha256`), re-run
Pass-1/2 extraction with the promoted checkpoints, retrain fusion (incl. the validated
recombination approach) on the refreshed features, and report the final combined numbers.


## 2026-08-04 (later) — [WIN-3060] — Gesture fine-tune on complete dataset: also a clean win

**Did:** `scripts/36_finetune_gesture.py` — warm-started from deployed `best_TCN.pth`,
class-weighted loss (gesture's own original recipe already does this, no change needed), lr
1e-4, 46 epochs to early-stop, selecting on val macro-F1. Evaluated via
`scripts/32_refresh_and_compare.py --gesture-ckpt ...` (full windowed pipeline, test-split-only
comparison).

**Result — real improvement, third data point confirming the pattern:**
| Metric | Deployed | Fine-tuned | Delta |
|---|---|---|---|
| Headline test acc | 0.763 | **0.864** | +0.101 |
| Headline test macro-F1 | 0.741 | **0.836** | +0.094 |
| `raw_take` (genuinely unseen) acc | 0.738 | **0.856** | +0.118 |
| `curated_clip` acc | 0.96 | 0.93 | −0.03 |

**Per-class:** `wave` (the worst class before, 0.47) is now **perfect (1.00)**. `beckoning`
0.61→0.95, `point` 0.68→0.78, `thumbs_down` 0.90→0.97. Two mild regressions: `idle` 0.99→0.85,
`thumbs_up` 0.89→0.85. **`raise_hand` remains the persistent weak spot, still worst-in-class**
(0.37→0.33 — essentially unchanged) — likely still confused with `wave` per the earlier
confusion matrix; a genuine open item, not fixed by this fine-tune. Both contexts improved
(classroom 0.74→0.85, kitchen 0.79→0.88).

**Confirms the loss-weighting hypothesis a third time.** All three fine-tunes now on record:
emotion (class-weighted loss) +9.3pts headline, gesture (class-weighted loss) +10.1pts headline,
motion (`WeightedRandomSampler`) −13.7pts headline (regression, both warm-starts). The pattern
is consistent enough to treat as a real finding, not noise: **class-weighted loss generalises;
resampling/duplicating minority-class examples overfits to this dataset's narrow
(2-actor, same-scenario) validation set.**

**Minor script bug fixed:** `classification_report` crashed on a val batch missing one gesture
class (needs explicit `labels=range(8)`) — crashed AFTER the checkpoint saved, so no data lost;
fixed for future runs, this run's MLflow entry logged post-hoc.

**Status of all three fine-tunes (none promoted to deployed yet):**
| Model | Verdict | Checkpoint |
|---|---|---|
| Emotion | ✅ promote | `finetuned_MobileNetV2_merged.pth` |
| Gesture | ✅ promote | `best_TCN_finetuned_merged.pth` |
| Motion | ❌ do not promote (regressed); deployed checkpoint stays | `best_model_finetuned_merged_{ntu,deployed}.pt` kept as evidence only |

**Next:** decide on promoting emotion+gesture to deployed (updates `checkpoint_manifest.sha256`,
requires re-running Pass-1/2 extraction + re-training fusion on the refreshed features);
optionally retry motion with class-weighted loss instead of the sampler, now that the pattern
is well-evidenced, before writing it off entirely.


## 2026-08-04 — [WIN-3060] — Emotion fine-tune on complete dataset: a genuine win (unlike motion)

**Did:** `scripts/33_extract_emotion_crops.py` — new pass caching up to 15 evenly-spaced
face-crop IMAGES per clip (all 2,869 clips, 0 failures, ~0.37 clips/s once warmed) — needed
because Pass-1's per-frame cache only stored emotion_probs (the OLD model's output), not the
input crops. `scripts/34_finetune_emotion.py` — warm-started from the DEPLOYED checkpoint,
**class-WEIGHTED loss** (not resampling — deliberately avoiding the mechanism suspected of
causing motion's regression), lr 3e-5, label smoothing 0.1, selecting on val macro-F1.
`scripts/35_compare_emotion.py` — evaluates old vs new checkpoint on the IDENTICAL cached test
crops (isolates the checkpoint as the only variable — the methodology fix from the motion
comparison).

**Result — real improvement, confirmed on genuinely unseen test clips:**
| Metric | Deployed | Fine-tuned | Delta |
|---|---|---|---|
| Headline test acc | 0.681 | **0.775** | +0.093 |
| Headline test macro-F1 | 0.587 | **0.668** | +0.081 |
| `raw_take` (genuinely unseen) acc | 0.644 | **0.762** | **+0.118** |
| `curated_clip` (data/old overlap) acc | 0.96 | 0.87 | −0.09 (expected — less memorisation, more generalisation) |

**Per-class — fixes exactly the two weakest classes identified in the 2026-08-03 unimodal
audit:** Fear 0.42→0.81 (+0.39), Disgust 0.39→0.78 (+0.39). Both contexts improved (classroom
0.63→0.75, kitchen 0.72→0.80). One regression: Neutral 0.79→0.60 (−0.19) — expected trade-off,
Neutral got the lowest loss weight (0.31, vs Surprise's 2.31) to force attention onto the
minority classes.

**Why this worked where motion didn't (the key methodological lesson):** emotion used
class-WEIGHTED LOSS; motion used a `WeightedRandomSampler` (exact-duplicate oversampling of the
rarest class). Two data points now point the same direction — loss-weighting is the safer
intervention for this dataset's actor/scenario-narrow val set. Also notable: emotion's val
macro-F1 peaked at epoch 1 and declined every epoch after (early-stop correctly kept epoch 1),
whereas the FINAL/kept checkpoint still generalised to test — unlike motion where a good val
score at epoch 9+ did not transfer at all.

**Checkpoint:** `modalities/emotion/checkpoints/finetuned_MobileNetV2_merged.pth` — NOT YET
promoted to deployed (`finetuned_MobileNetV2.pth`); promote after gesture is also checked, per
user's plan to review all three together before finalising.

**Next:** gesture fine-tune (`scripts/36_finetune_gesture.py`, same class-weighted-loss approach,
already gesture's own original recipe).


## 2026-08-03 (latest) — [WIN-3060] — Rubric-driven cue recombination: recovers 95% of the fusion-generalisation gap

**Did:** built and ran the recombination experiment the gap decomposition pointed at.
- Refactored shared gap-decomposition logic (REAL/ORACLE table construction, the rule
  baseline incl. its F09→F01 remap and missing-cue defaults, ceiling check) out of script 29
  into `scripts/realworld_eval/merged_gap.py`, so script 30 reuses it instead of duplicating.
- `fusion/model/recombine_merged.py`: generates synthetic training samples spanning **all
  448** context(2)×emotion(7)×gesture(8)×motion(4) combinations — not a curated row list like
  `data/old`'s `recombine.py`. Each combo is labelled via the shared `rule_intent()` (with the
  F09→F01 remap). Two deliberate departures from the `data/old` version, both documented in
  the module docstring: (1) pools are indexed by **ground-truth class**, not the model's own
  argmax (the old approach self-selects vectors the model already gets "right" and
  under-represents confusable cases); (2) full combinatorial span, which explicitly includes
  every F02 (emergency) combination — the specific gap the safety finding demanded.
- `scripts/30_merged_recombination.py`: 4-way ablation isolating recombination's contribution
  from dropout/jitter's — `plain` / `augmented` (dropout+jitter only) / `recomb_only`
  (recombination only) / `full` (both) — 3 seeds each, on the same headline test protocol as
  the gap decomposition.

**Results (headline test, 3 seeds, n_per_combo=100 → 44,800 synthetic samples):**
| Config | Clip acc | macro-F1 |
|---|---|---|
| plain (reproduces gap decomposition's fusion+real) | 0.475 ± 0.034 | 0.395 |
| augmented (dropout+jitter, no recombination) | 0.489 ± 0.036 | 0.390 |
| recomb_only (recombination, no dropout/jitter) | 0.590 ± 0.013 | 0.502 |
| **full (recombination + dropout+jitter)** | **0.601 ± 0.003** | **0.508** |

Reference: rules+real = 0.608 (the number learned fusion needs to beat for G1 to hold on
unseen combinations); rules+oracle / ceiling = 1.000.

**Recombination recovers 95% of the gap** ((0.601−0.475)/(0.608−0.475) = 0.947) and gets
**tighter, not just better** — seed std drops from ±0.034 (plain) to ±0.003 (full), the most
stable fusion config measured yet. It does **not** quite cross rules+real (0.601 vs 0.608, a
0.007 gap — within measurement noise of a single seed but the deterministic rule number has no
variance to compare against). Recombination alone (no dropout/jitter) already gets most of the
way (0.590); dropout/jitter's marginal contribution on top of recombination is smaller than its
contribution alone (0.475→0.489) — the two are not simply additive, recombination subsumes
most of what dropout/jitter buys standalone.

**Sensitivity check**: tried n_per_combo=250 (112,000 synthetic samples) — mean headline acc
similar (~0.602) but **seed variance nearly doubled** (±0.025 vs ±0.003). n_per_combo=100 is
the better operating point: more synthetic data did not help and cost stability. Reverted to
100 as the canonical/saved config.

**F02 (emergency) rows #23/#53/#54 — the safety check the whole experiment was aimed at:**
| Row | Before (fusion+oracle, gap decomposition) | After (full config, real cues) |
|---|---|---|
| #23 | 0% (fails even with PERFECT cues) | 47.7% (44 clips) |
| #53 | 0% | 66.7% (39 clips) |
| #54 | 0% | 71.1% (38 clips) |

Large, real recovery on the exact rows the gap decomposition flagged as unreachable by
perception fixes alone — direct evidence the fusion-generalisation diagnosis was correct and
that recombination is the right lever. Not yet reliable enough for a safety-critical claim
(#23 still <50%), but this is now a tuning/coverage problem, not an architectural dead end.

**Logged:** 12 MLflow runs across 3 script invocations (n=100 canonical run before and after
the n=250 sensitivity check) to `04_recombination`, exported to `results/EXPERIMENTS.csv`
(51 runs total across all 4 experiments). Artifacts: `results/realworld_eval_merged/
{RECOMBINATION.md, recombination_report.json, recombination_results.json}`.

**Next:** the 0.007 shortfall vs rules+real is close enough that a few candidate next steps
could close it — per-intent-class n_per_combo weighting (F10/F08 are thin: 800/1,600 samples
vs F01's 10,400), longer patience, or ensembling seeds. Re-run the exact gap-decomposition
oracle/real comparison with the `full`-config model to get an updated, apples-to-apples
generalisation-cost number (this run only measured against the OLD gap-decomposition baseline
numbers, not a fresh oracle re-run on this model).


## 2026-08-03 (latest) — [WIN-3060] — Full-dataset gap decomposition: confirms classroom finding, exposes F02 as a fusion problem

**Did:** `scripts/29_merged_gap_decomposition.py` — extended
`results/realworld_eval_final/GAP_DECOMPOSITION.md` (classroom-only) to the complete
`data/final_merged` (both contexts, all 62 V3 rows, 2,869 clips). Same protocol: clip-level,
4s mean-pooled, actor-disjoint val, plain fusion (no augmentation), 3 seeds.

**Ceiling is 1.0, not 0.900** — confirmed computationally (0 colliding cue tuples at the
intent level; F09's removal deleted the classroom direction collision that capped the old
table). This is a materially easier ceiling than the classroom-only study used.

**Bug found and fixed before trusting any number:** `rules+oracle` (perfect cues) scored
0.902, not the expected 1.0. Root-caused to a SINGLE dead branch: `rule_based.py`'s
`wave+walking+non-happy→F09` logic is correct for `data/old` (still has F09) but wrong here
(F09 was folded into F01). Diagnosed precisely — rows #22 and #61 only, 96/979 headline
clips, both 0% before the fix — confirmed by adding a table-specific F09→F01 remap (in the
script, not the shared baseline) which took rules+oracle to **exactly 1.000**. Also fixed a
second, non-impactful bug in oracle construction: masked cues were defaulting to class index 0
via `argmax` of an all-zero vector (e.g. emotion silently “Surprise”) instead of a reasoned
safe default (Neutral/idle/standing, per the table's own stated fallback) — verified this
dataset's masked-emotion rows never hit the one branch where it would have mattered, but it is
now the correct contract.

**Final headline numbers (test split, excl. #58's 24 derived clips):**
| Configuration | Clip acc | macro-F1 |
|---|---|---|
| Rules + oracle cues | **1.000** | 0.900 |
| Fusion + oracle cues | 0.619 ± 0.048 | 0.514 |
| Rules + real cues | 0.608 | 0.542 |
| Fusion + real cues | 0.475 ± 0.034 | 0.395 |

**Decomposition: fusion generalisation cost 0.381 (dominant) vs perception cost 0.144.**
Directly confirms the classroom-only finding (there: 0.40 vs 0.175) on the complete,
two-context dataset — the bottleneck is the learned model's inability to generalise the
rubric to unseen cue combinations, not the perception models. **Rules still beat fusion on
real cues** (0.608 vs 0.475), an even clearer margin in relative terms than classroom-only.

**New this run — real seed variance on fusion+oracle** (0.556–0.673, ±0.048), unlike the
classroom study's exact ±0.000. With kitchen's extra rows and tuple diversity, different
random inits now partially recover different parts of the compositional structure instead of
collapsing identically. Worth a closer look — possibly informative about what the successful
seed's decision boundary looks like.

**Safety-relevant finding — F02 moved from “perception problem” to “mostly fusion
problem”.** Per-row diagnosis (`results/realworld_eval_merged/gap_per_row_merged.csv`): rows
**#23, #53, #54 (all F02, emergency) fail EVEN WITH PERFECT ORACLE CUES.** This changes the
prioritisation from the earlier unimodal-eval session, which found F02 rows scoring 0% on
emotion/motion and read that as a perception gap — the oracle test shows perfect perception on
those rows *still* wouldn't fix F02 detection with the current fusion training. Fine-tuning
emotion/motion alone will not close this; it needs the same fix as the rest of the
generalisation gap (rubric-driven recombination), with F02 rows a priority inclusion given the
safety stakes.

Logged: 4 new runs to MLflow `03_diagnostics` (`final_merged__rules_oracle/real`,
`final_merged__fusion_oracle/real`), exported to `results/EXPERIMENTS.csv` (39 runs total).
Artifacts: `results/realworld_eval_merged/{GAP_DECOMPOSITION_MERGED.md,
gap_decomposition_merged.json, gap_per_row_merged.csv}`.

**Next:** rubric-driven cue recombination — the lever both this and the classroom study point
at. Design must explicitly include F02 combinations given the finding above. Re-run this exact
decomposition afterward on the same table to confirm the 0.381 gap actually closes.


## 2026-08-03 (later still) — [WIN-3060] — `data/final_merged` extracted & scored: complete dataset

**Did:** full Pass 1 + Pass 2 over the complete, corrected dataset (2,869 usable clips, all
62 V3 rows, both contexts, all views).
- `scripts/27_merged_extract.py` — Pass 1, with sha256-verified cache reuse from
  `data/final` (safety check: `n_frames` + `fps` must match `clips.csv`, or re-extract —
  guards against the R4 re-encoding). **1,500 reused / 1,369 fresh, 0 failures.** Smoke-tested
  both paths (reuse and fresh `.MOV` extraction) before the full run.
- `scripts/realworld_eval/merged_unimodal.py` + `scripts/28_merged_unimodal_eval.py` — Pass 2
  + evaluation, mirroring `final_unimodal.py`/`16_final_unimodal_eval.py` but reading the new
  `split`/`headline_eval`/`resolution_class` columns from `23_build_splits.py` instead of raw
  `view`/`split_design`. **41,144 windows** written to `unimodal_windows.parquet`.
- **Bug caught before the numbers were trusted:** first run's "headline" metric filtered on
  `headline_eval` alone, which is True by default for train/val rows too (it only ever turns
  False for row #58's 24 clips) — so it silently included training data and read
  emotion=0.77/motion=0.66 acc, an inflated number. Fixed to require `split=='test' AND
  headline_eval` — corrected headline: **emotion 0.674, gesture 0.763, motion 0.665,
  context 0.971** (n=856–979). Logged to MLflow (`03_diagnostics`) and exported to
  `results/EXPERIMENTS.csv`.

**Headline held-out numbers, complete dataset (test split, excl. #58's derived clips):**
| Modality | clip acc | macro-F1 |
|---|---|---|
| emotion | 0.674 | 0.581 |
| gesture | 0.763 | 0.741 |
| motion | 0.665 | 0.589 |
| context | 0.971 | 0.492 (near-perfect acc; low F1 = occasional hallucinated hospital/museum) |

Matches the classroom-only numbers in `ASSESSMENT.md` closely (emotion 0.643, gesture 0.766,
motion 0.637) — kitchen does not change the picture much on its own.

**Two findings that matter more than the headline table:**
1. **The T03 honesty problem is now confirmed dataset-wide, not just classroom.** Every
   designed-missing row (`docs/methodology/04_missing_cues.md` §4) shows **observation_rate =
   1.0** — emotion, gesture AND context all fire 100% of the time on rows the table says
   should be unobservable. T03 remains simulated (flag-driven) masking only; still true nothing
   about the underlying pipeline has changed on this front.
2. **F02 (emergency) rows are a systematic weak spot — safety-relevant.** Of 7 F02 rows
   (#3,4,23,34,35,53,54), at least **4 score 0% on emotion and/or motion**
   (#4 stepping_back→0, #23 Fear→0.068 recall region, #34/#35 stepping_back→0). Motion's
   `stepping_back` recall is 0.243 overall (confusion: 150/404 → standing, 150/404 → walking)
   — confirms and extends the 2026-07-16 kitchen-only finding to classroom F02 rows too.
   Needs an oracle-cue check on F02 specifically before deciding whether this is a perception
   problem or (as the classroom gap decomposition suggested) a fusion generalisation problem.

**Held-out vs fine-tune-overlap (source column), full dataset:**
| Modality | curated_clip (fine-tune-adjacent) acc | raw_take (unseen) acc |
|---|---|---|
| emotion | 0.961 | 0.642 |
| gesture | 0.858 | 0.771 |
| motion | 0.862 | 0.537 |
| context | 0.973 | 0.952 |

Confirms emotion and motion's `data/old`-derived numbers are inflated by fine-tune overlap;
gesture and context show little gap (gesture's custom clips and context's zero-shot backend
were never fit to this specific footage).

**Next:** oracle-vs-real gap decomposition on the complete dataset (both contexts, all V3 rows)
— the classroom-only version (`GAP_DECOMPOSITION.md`) found the bottleneck was fusion
generalisation (0.40) not perception (0.175); worth re-confirming now kitchen is complete and
before deciding on any unimodal fine-tuning. Then the rubric-driven recombination experiment.


## 2026-08-03 (later) — [WIN-3060] — MLflow experiment tracking

**Did:** stood up MLflow with a storage model that mirrors the checkpoint manifest, so it
does **not** become a second cross-machine sync problem.
- `fusion/tracking.py` — thin wrapper; callers never import mlflow. `start_run()`
  **requires** `dataset`, `split_kind` and `cues`, enforcing methodology 07 §7.1 (no number
  in this project is interpretable without all three). Auto-tags git commit + dirty flag,
  machine name, and the checkpoint-manifest hashes.
- Store: **SQLite** `mlruns.db` + `mlruns/` in the repo root, both **gitignored**
  (MLflow 3.x put the plain-file store in maintenance mode and refuses it by default).
- `scripts/25_export_runs.py` → **`results/EXPERIMENTS.csv` + `.md`, which ARE committed**
  — that export is the cross-machine record; there is no tracking store to merge.
- `scripts/26_backfill_mlflow.py` — idempotent backfill from saved outputs; every run
  tagged `backfilled=true` with its original measurement date so it is never mistaken for
  a fresh one. **23 runs logged**: 6 baselines, 5 fusion ablations, 12 diagnostics
  (masking sweep, window sweep, gap decomposition, clip-vs-window).

**⚠ Environment trap found and fixed (would have broken the next extraction run):**
installing mlflow upgraded **protobuf 3.20.3 → 6.33.6**, which breaks MediaPipe
(`SymbolDatabase.GetPrototype` removed). **A blank-frame smoke test does NOT catch this**
— it only fails once a frame actually yields landmark protos, which is why it passed my
first check and failed on a real clip. Fix: `protobuf==3.20.3` restored and re-verified on
a real frame (pose found ✅); mlflow 3.15 works fine under it. Both pins now carry
explanatory comments in `requirements.txt`.

**Next:** feature extraction for `data/final_merged` (2,869 usable clips; ~1,440 per-frame
caches from `data/final` are re-mappable by sha256 instead of re-decoding), then the
rubric-driven recombination experiment — logged through the tracker from the start.


## 2026-08-03 — [WIN-3060] — `data/final_merged` audit + 3 fixes

**Did:** independent audit of the merged dataset after the Aug 1–2 CSV rebuilds
(later than `DATASET_FIXLIST.md`'s body). Findings appended to that file as
"ADDENDUM — independent audit 2026-08-03" (N1–N9).

**Confirmed healthy:** `person_id` now **100 %** (2,869/2,869) in clips.csv *and*
splits.csv with **0 disagreements** (was 1,097/2,870); disk↔CSV exact (2,904 files,
0 orphans, 0 missing — note the tree mixes `.mp4` + `.MOV`, a `*.mp4`-only glob
under-counts by 516); 35 duplicates correctly excluded; 62 rows ↔ 62 folders.

**Fixed this session:**
- **N1 — no validation split.** `23_build_splits.py` now emits `split`
  (train/val/test) beside `split_design`. val = actors **P04+P03** carved from
  train: 319 clips (17.1 %), 30/40 rows covered, none emptied, actor-disjoint and
  take-grouped — all self-checked by the script. **train 1,547 / val 319 / test
  1,003.** Training code must read `split`, not `split_design`.
- **N5 — stale takes.csv.** New `24_regen_takes.py` (script 20's aggregation,
  surgical). Orphan `S28_F07` take 19 dropped; person_id **956 → 1,736/1,736**.
  `22_verify_csvs.py` now reports zero inconsistencies.
- **N6 — `21_validate_table.py` crashed** on stock Windows consoles
  (UnicodeEncodeError on `→`) *while printing its own errors*. Now reconfigures
  stdout/stderr to UTF-8; reports 1 error + 13 warnings and exits cleanly.

**Still open, needs user decisions:** N4 (row #58 is 56 % train footage —
`crosses_split=True`, and the mask changes intent F01→F06; exclude from headline
test metrics or re-record), N9 (RealSense is only 275/1,003 test clips; 2 rows have
zero, 4 test rows have 5–6), N7 (127 clips <4 s, min 1.67 s — need a policy), N8
(only 289/1,015 `phone_1080p` clips are actually 1080p), N3 (rows 40/41/57 and
38/63 duplicate a tuple across train/test → say "19 of 22 test rows are unseen"),
N2 (6 actors span train/test **by design** — test measures compositional, not
subject, generalisation), plus the pre-existing E1 action collision (#1 vs #18).

## 2026-08-01 — [WIN-3060] — V3 table corrected at source; CSVs verified against it

**Did:** fixed `docs/final_dataset_merged.docx` itself (backup `.docx.bak`) rather than
patching around it in code, then rebuilt and verified the annotation tables.

**Document edits — 45 cells, 1 row deleted, 1 row moved, 1 legend cell.** 11 label cells
were Word run-splitting typos (`F0 1`, `T est`, `s t ep back`); 10 were the agreed changes
(#18→`T01`, #32→`T01, T04`, #52 test tags cleared, #57 and #63 unmasked, #63 motion
`sitting`→`sit`); 24 were prose. Row #30 (already empty) deleted; #63 moved to the end;
`Motion (6)`→`Motion (4)`. **No cue, intent or action value changed in meaning** beyond the
two authorised mask removals.

**12 broken cross-references found and corrected** in the justification column: #47 cited
itself (→#46); #39 named #57 as its context-masked twin (→#56); #9 named #54 as a
gesture-masked test (→#53); #53 named #44 (→#43); #55 named #42 (→#41); #21/#62 listed #52
among gesture-none rows (→#50); #26 listed #45 among both-hands-up rows (→#44); #2/#13/#28/#32
named #60 as a conflict row when the kitchen conflict is #59. The last group matters: those
sentences define the **T02** subset, and #60 (disgust + thumbs down) is an *aligned* row, not
a conflict.

**`TABLE_OVERRIDES` is now empty** — the parser reads the document literally again.

**New code:** `scripts/21_validate_table.py` (validates the .docx against its own legend
tables — vocabulary, code coverage, cue/action collisions, split-vs-T-tag, prose citations,
scenario text vs cue columns) and `scripts/22_verify_csvs.py` (re-reads the .docx with an
independent parser and compares **41,848 field values** across 62 rows and 2,905 clips).

**Verification: ALL 62 rows match exactly** between the .docx and the CSVs. Table validator:
**1 error, 13 warnings**.

**The one error is the finding that matters:** rows **#1 (train, A01)** and **#18 (test, A09)**
are the identical observable tuple `classroom|happy|wave|walk` with the same intent F01 but
**different actions**. Deleting F09 removed the direction ambiguity from the intent level; it
reappeared at the action level. F01 maps to A01/A09/A10 and no cue distinguishes them, so an
`intent → action` policy leaves A09/A10 unreachable. `pose_img` is already cached for a
direction feature (DECISIONS 2026-07-27) if that is the route chosen.

**Also resolved:** row #63 *is* the old `S28_F10` — the 2026-07-16 S21/S28 collision. It is
correctly relabelled F04 here, and #51 was re-recorded gesture-free on 2026-07-28 so F10 keeps
its own footage. `data/old/labels.csv`'s `recombination_pool` status is obsolete. Row #50's
"VERIFY against recordings" note is closed: the actors perform no thumbs down, `idle` is right.

**Still open (see `docs/DATASET_FIXLIST.md`):** #31's motion says `step back` while its own
text and justification say standing (unchanged, needs a call); rows #10/#40/#41 label a
described point as `idle`, which the gesture model will not reproduce; `run` is used in three
scenarios but is not in the motion vocabulary; #40/#56 have no RealSense clips; `person_id`
for 781 takes.

## 2026-07-31 — [WIN-3060] — `data/final_merged` audited + annotated

**Did:** audited the new collection root (`data/final dataset merged`, renamed to
`data/final_merged`) and the new table (`docs/final_dataset_merged.docx`) against
`data/final` / `docs/Final_Dataset.docx`, then built its annotation tables.

**The root is a reorganisation, not new footage.** SHA-256 of all 2,905 files: every one
already existed under `data/final` — 0 new bytes. Deltas: `+S06_F03`, `+S09_F04` (physical
copies of S05/S08, with `.txt` notes saying so), `S18/S19_F09 → _F01`, `−S30_F09` (48 clips),
`−classroom/dilanka` (36 of 60 folded 4-at-a-time into the S22–S31 test folders, 24 dropped).
Everything was renamed to `S{row}_F{intent}_c{NNN}`, which destroyed the only record of each
clip's session and `data/old` identity; both are recovered here by hash, not filename.

**The table change is the important one: F09 (Farewell) is gone.** Rows #18/#19/#48/#49/#61
relabelled F09→F01, row #30 blanked, row #63 added. Effect, measured on observable cue tuples:
**0 collisions across the 62 live rows** (was 4 pairs separable only by walking direction, which
no model outputs — the thing that capped classroom test accuracy at 0.900 per the 2026-07-27
audit). The direction cue is no longer required by the label design. 9 intents; 62 live rows ↔
62 folders, 1:1; every folder's F-tag and context match its row; folder-number == V3-row
re-verified on all 23 migrated scenarios (0 mismatches).

**Integrity problems found (all in `annotations/INTEGRITY.md`):**
- **24 clips are shared byte-identically between rows #49 and #58.** User decision: #58 is a
  **derived row** — #49's footage re-used with emotion+gesture masked, same trick as #6←#5 and
  #9←#8. Recorded with the three objections raised at the time: (i) #49 is *train* and #58 is
  *test*, so those frames sit on both sides of the split; (ii) the mask **changes** the intent
  (F01→F06) where #6/#9 preserve it; (iii) #58's rationale makes direction the deciding cue
  ("toward robot") while #49's footage walks toward the exit. `derived_rows.csv` carries a
  `crosses_split` flag; #58's 24 derived clips must be scored separately from its own 19.
- **35 exact-duplicate files** (23 within a folder, 12 claimed by two folders whose rows share
  a cue tuple and intent, so label-neutral). Excluded with a `dup_of` pointer.
- **Rows #40 and #56 have no RealSense clips at all** — unscoreable in the deployment view.
- Rows **#57 and #63 were internally contradictory** (cue columns observed, Missing/Goal/Test
  still masked; #63's motion read `sitting`). User: the cue columns win, read only cues +
  intent. **Cost:** unmasked, both now repeat a training tuple, so test rows presenting an
  unseen cue combination drop from 21 to **19 of 22**, and T03 loses both. `INTEGRITY.md`
  tracks this in a "what the test rows actually test" section.
- Cosmetic, unfixed in the doc: #18/#32 are test rows with no T-tag, #52 is train but tagged
  T04/T05, #31's justification still cites F09, #63's justification is copy-pasted from #56.

**New code:** `scripts/20_merged_annotations.py`, `scripts/realworld_eval/merged_common.py`.
**Artifacts:** `data/final_merged/annotations/{scenarios_v3,clips,derived_rows,takes,retired_rows}.csv`
+ `INTEGRITY.md` (gitignored). 2,905 clips / **2,870 usable** = 1,866 train + 1,004 test;
**2,726 need feature extraction, 144 reuse a source row's features**; person_id known for
1,121 clips (all migrated ones) via `data/old` hashes.

**Next:** extract per-frame features for the 2,726 non-derived clips against `data/final_merged`,
then re-run the unimodal + fusion evaluation on the collision-free table.
**Blocked on user:** person_id for the 781 non-migrated takes in `takes.csv`; re-record decision
for #40/#56 RealSense.

## 2026-07-28 — [WIN-3060] — Gap decomposition: the bottleneck is NOT perception

**Did:** two studies on `data/final` (classroom, RealSense, 868 clips, real V3
train/test split, 3 seeds, take-grouped val).

**1. Clip-level vs window-level fusion** (user's proposal). Aggregating cues over a clip
reads the intended cue better than a single window (emotion 0.771→0.802, gesture 0.788→0.805,
motion 0.719→0.733) and clip-level training beats window+majority-vote (**0.374 vs 0.342**,
macro-F1 0.320 vs 0.284). **Never mix** train-on-window with infer-on-clip (0.331/0.309 — worse
than either consistent choice). mean ≈ peak > max. Clip-level wins with 784 samples vs 10,946 —
the extra windows were largely redundant. **Decision: adopt 4 s mean-pooled aggregation windows
(stride 1 s) as the primary path, KEEP the window pipeline for comparison** (user's request).

**2. Gap decomposition — the important one.** Oracle test: same fusion architecture fed the
table's TRUE cue labels instead of model predictions.
| Configuration | Test clip acc |
|---|---|
| ceiling (direction ambiguity) | 0.900 |
| rule-based + oracle cues | **0.900** (hits ceiling exactly) |
| learned fusion + oracle cues | **0.500** ±0.000 |
| learned fusion + real cues | 0.325 ±0.018 |

→ **fusion generalisation cost 0.40 (dominant), perception cost 0.175, label ambiguity 0.10.**
Even with perfect cues the learned head reaches only 0.50 while a hand-written rubric reaches
0.90. Zero seed variance on the oracle run = systematic, not noise. Cause: the head trains on
~19 distinct cue tuples and must classify 10 unseen-by-design tuples — a compositional
generalisation failure. Rows #26 and #23 fail *even with perfect cues*; row #24 is the opposite
(oracle 1.00, real 0.02) i.e. a genuine perception failure.

**This overturns `ASSESSMENT.md` §5.** Its prediction "retrain fusion on data/final → high-0.8s"
was tested and is wrong (0.325–0.374): the 10 test rows are unseen *by design*, so retraining
cannot make them seen without leakage. Revised priority: (1) rubric-driven cue recombination
spanning the combinatorial cue space — the only lever on the 0.40 band; (2) emotion/motion
fine-tune — the 0.175 band; (3) direction cue — the 0.10 band.

**Honesty note recorded in the doc:** training on rubric-generated samples means fusion is
*taught* the rubric, not discovering it. Defensible framing: rules+perfect cues = 0.90 but
rules+real cues are brittle (0.695 vs fusion 0.951 on data/old), so the learned model's
contribution is *noise/missing-cue robustness*, and augmentation supplies the semantics it
cannot induce from 19 tuples.

**Artifacts:** `results/realworld_eval_final/GAP_DECOMPOSITION.md`.
**Next:** implement rubric-driven recombination for `data/final` and re-measure the same table.

**3. Baseline reversal (added same day).** Rule-based on REAL 4 s-pooled cues, classroom:
train 0.758 / **test 0.494** (macro-F1 0.473). Learned fusion on the same test rows: **0.325**.
**On unseen cue combinations the rule baseline BEATS learned fusion** — the G1 claim as measured
on `data/final` currently does not hold. On `data/old` (unseen *people*, familiar tuples) it was
the reverse: fusion 0.951 vs rules 0.695. Each method has the strength the other lacks — rules
generalise compositionally by construction, fusion absorbs perception noise. This is the
argument for rubric-driven recombination (semantics from rules + robustness from real noisy
cue vectors). Written up in `docs/methodology/05_baselines.md` §5.3–5.4.
Methodology folder renumbered: 04 = missing cues (user-authored), 05 = baselines, 06 = fusion
model, 07 = evaluation, 08 = deployment.

**4. Methodology folder complete (2026-07-28).** All eight stages written:
`docs/methodology/{README, 01_data_and_labels, 02_unimodal_models, 03_feature_extraction,
04_missing_cues (user), 05_baselines, 06_fusion_model, 07_evaluation, 08_deployment}.md`.
Each stage carries its own "open points you might want to change" list. Corrected two stale
facts while writing: AttentionFusion is **70,090 params** (not ~110K as WORKLOG previously
said), and the `data/final` "fusion+real, train rows ≈0.89" cell was removed from
`05_baselines.md` because 0.887 belongs to the frozen fusion v1 model trained on `data/old`
and is not comparable to the retrained model.

## 2026-07-24 — [WIN-3060] — Methodology docs + regenerated gesture report

**Did:**
- Started `docs/methodology/` (narrative, thesis-facing): README (overview+pipeline),
  `01_data_and_labels.md`, `02_unimodal_models.md`. Stage-by-stage; more to come.
- **Regenerated the stale gesture real-world report.** The old
  `modalities/gesture/reports/evaluation/TCN/REALWORLD_REPORT.md` was from a pre-both-hands-up
  checkpoint (showed both_hands_up AND point as untrained → 0%). New script
  `scripts/12_regen_gesture_realworld.py` reruns the DEPLOYED `best_TCN.pth` over the curated
  clips via the per-frame caches (offline whole-clip resample). Old report preserved as
  `REALWORLD_REPORT_pre_bhu.md`.
- Verified all deployed checkpoints against `checkpoint_manifest.sha256` — all 7 OK.

**Regenerated gesture numbers (deployed best_TCN.pth, curated data, offline):**
overall **76.8% acc / 77.7% macro-F1**. both_hands_up now recognized (S05 100%, S09 98%,
S19 83%, S24 85%, S26 84%). Still weak: `point` in classroom (S03 0%, S22 5%) though S29
kitchen recovered to 78%; `wave` weak (S02 62%, S25 25% → raise_hand). Confirms the earlier
finding that the pre_bhu report was stale.

**Next:** per-model deep-dive discussion with user, then methodology Stage 3 (feature extraction).

## 2026-07-20 (later) — [WIN-3060] — TensorRT path for Jetson

**Did:** added the TensorRT acceleration path and documented it end-to-end.
- `jetson_deploy/fusion/pipeline.py`: new **`--backend tensorrt`** = ONNX Runtime TensorRT
  Execution Provider (Route A). Reuses the existing `onnx/*.onnx`; JIT-builds + caches FP16
  engines in `onnx/trt_cache/` on first run. Refactored the 3 backends behind `self.use_ort`.
  Regression-checked `--backend onnx` still gives S01→F04→A05 after the refactor.
- `jetson_deploy/fusion/build_engines.sh`: Route B — native `trtexec` → `onnx/*.engine` (FP16,
  batch=1). `jetson_deploy/fusion/trt_check.py`: verifies each engine's argmax vs ONNX (pycuda+TRT).
- `jetson_deploy/TENSORRT_GUIDE.md`: full guide — corrects the mental model (see below), Route A
  vs B, JetPack prereqs, workflow, FP16 accuracy note, troubleshooting table.
- `.gitignore`: `*.engine` + `trt_cache/` (device-specific build artifacts, never commit/copy).

**Key correction to the plan (user asked "PyTorch→ONNX→TensorRT, correct?"):** correct for the
4 NETS, but (1) TensorRT does NOT touch the real bottleneck — MediaPipe Holistic (~55 ms/frame)
is a TFLite graph, stays CPU; the nets it accelerates are already cheap. (2) Engines MUST be
built ON the Jetson (hardware/version-specific — cannot build on the 3060 and copy). (3) CLIP
stays PyTorch. Net effect: TensorRT is worth it for headroom + a clean latency table, but the
pipeline already meets the 300 ms budget without it. Biggest Jetson latency win is Holistic
`model_complexity=0`, not TensorRT.

**Artifact-sync note:** `jetson_deploy/onnx/*.onnx` are gitignored (same policy as `.pt`/`.pth`).
Regenerate on the Jetson with `scripts/08_export_onnx.py`, or copy `onnx/` manually. `.engine`
files are Jetson-only — build there with `build_engines.sh`.

**Next (on the Jetson):** verify `onnxruntime-gpu` has `TensorrtExecutionProvider`; run the
Route A / Route B workflow in TENSORRT_GUIDE.md §3; profile onnx vs tensorrt and log numbers here.

## 2026-07-20 (later) — [WIN-3060] — Jetson streaming pipeline + inference guide

**Did:**
- `jetson_deploy/fusion/pipeline.py` — streaming pipeline: one Holistic pass per frame feeding
  gesture+motion, rolling time-based buffers, fusion every stride (8/30 s), majority-vote
  smoothing (N=3) + hysteresis (2 consecutive), F02 emergency bypass (skips both), policy layer.
  Backends: `--backend torch|onnx`. Sources: `realsense` / camera index / video file.
  **Windowing mirrors `fusion/extraction/windows.py` exactly** — any change must be made on both
  sides or train/serve skew silently costs accuracy.
- `jetson_deploy/fusion/{model.py,policy.py}` — self-contained copies (noted as copies in their
  headers; re-copy + re-export ONNX if the originals change).
- `jetson_deploy/fusion/profile_latency.py` — per-component profiler with the 300 ms budget check.
- `jetson_deploy/JETSON_INFERENCE_GUIDE.md` — setup, run commands, expected output, profiling,
  tuning ladder, robot-integration snippet.

**Verified on real clips (both backends):** S01_F04 (raise hand/neutral/sit) → steady F04→A05;
S19_F02 (fear/both hands up/step back/kitchen) → F02→A02 with EMERGENCY on every step.

**Problems / fixes:**
- My first profiler derived "per-stride = 8 × holistic + step" → falsely reported 550 ms OVER
  BUDGET. Wrong: frames are processed as they arrive (pipelined), so capture→intent latency is
  ONE holistic pass + step work. Corrected → **125 ms mean / 267 ms p95 on the 3060, within the
  300 ms budget**. Throughput is the separate constraint (Holistic caps input ~18 fps → ~5 fresh
  frames per stride, which is fine because windowing is time-based and training clips were 15 fps).

**Reference latency (RTX 3060, ONNX):** holistic 55 ms/frame; per step emotion 46, context 19
(only every 1 s), gesture 2.4, motion 3.0, fusion <1 ms.

**Next:** run `fusion/profile_latency.py` ON THE JETSON and record numbers here; tuning ladder is
in the guide §4 (onnx → Holistic model_complexity=0 → fewer emotion frames → larger stride;
never shrink lookback spans).

## 2026-07-20 — [WIN-3060] — Phase 3: deployed model, ONNX, G3/window-sweep/policy

**Did:**
- **Final deployed fusion model** (`scripts/07_finalize_fusion.py`) →
  `jetson_deploy/fusion/fusion_attn.pt` + `fusion_config.json`. Config: attn exclude-mode,
  dropout 0.3, jitter 0.15, recombination, masked-val selection. Test clip acc **0.939 on all
  3 seeds** (chose seed 1 by masked-val score 0.8094).
- **Trial ONNX export PASSED for all 4 nets** (`scripts/08_export_onnx.py` →
  `jetson_deploy/onnx/`): emotion MobileNetV2, gesture TCN, motion LSTM, fusion attention
  (incl. the exclude-mode padding mask) — max |native−ORT| ≈ 1e-6, opset 17. The schedule's
  biggest risk is retired. Context/CLIP deliberately NOT exported (runs PyTorch from the bundled
  HF cache on Jetson; revisit only if latency profiling demands TensorRT).
- **G3 spotlight** (`scripts/09_g3_spotlight.py` → `results/fusion_v1/G3_SPOTLIGHT.md`):
  13/14 same-gesture cases resolve to the correct intent across emotion/context flips
  (miss: S24 kitchen angry both-hands-up, 2 test clips split F06/F07).
- **Window sweep** (`scripts/10_window_sweep.py` → `window_sweep.json`), spans ×0.5/×1/×2:
  test clip acc **0.976 / 0.951 / 0.756**. Shorter (1.07 s) lookback is BETTER and cheaper —
  ×2 (4 s) exceeds clip length and collapses. Deployed model stays on ×1 (matches the unimodal
  engines' validated buffers); try ×0.5 during Jetson profiling.
- **Intent→action policy** (`fusion/actions/policy.py`, demo `scripts/11_policy_demo.py` →
  `POLICY_DEMO.md`): V3 legend mapping, τ=0.5 fallback→F05/A06, F02 bypass τ=0.30 with
  context-routed hazard action (classroom→A14, kitchen→A02/A03). **F02 safety: 22/22 test
  emergency clips fire, window recall 0.968, false-emergency 1.3%.**

**Problems / fixes:**
- Installing onnx pulled **protobuf 7.x which BREAKS mediapipe** (FieldDescriptor error).
  Fixed: pin `protobuf==3.20.3` + `onnx==1.14.1` (now in requirements.txt). HPC: respect these
  pins or extraction dies.
- User committed the Phase-2 work themselves (`f57f27d fusion`); push deferred by user — deploy
  branch is local-ahead, sync to HPC via manual pull/copy for now.

**State:** Phases 0–2 complete + Phase 3 export/policy done on WIN-3060. Remaining: T02/T05
proper numbers need the V3 test scenarios recorded; Jetson-side `pipeline.py` (streaming buffers,
temporal smoothing/hysteresis, F02 bypass wiring) + on-device latency profile; attention-weight
interpretability figure; thesis tables from results/fusion_v1/.

## 2026-07-17 (later) — [WIN-3060] — Phase 2: attention fusion, augmentation, T03 sweep

**Did:**
- Built `fusion/model/`: `model.py` (attention fusion, 4 cue tokens + CLS, d=64, 2 layers, 70,090
  params; `missing_mode='token'` learned [MISSING] embedding OR `'exclude'` key-padding mask),
  `datasets.py` (modality dropout ≤2 cues + confidence jitter, train-only), `recombine.py`
  (cue recombination: 7,600 synthetic windows for the 19 unrecorded V3 train rows incl. all F10;
  V3 #18 skipped — cue-identical to #1 without direction; sources = train-subject windows only,
  S28 pool used as sources).
- `scripts/05_train_fusion.py`: ablation grid × 3 seeds + T03 masking sweep →
  `results/fusion_v1/{results.json, RESULTS.md, attn_robust_best.pt}`.
- `scripts/06_robustness_experiments.py`: follow-up after a negative result (below) →
  `results/fusion_v1/robustness.json`.

**Results (actor-disjoint test, clip-level):**
- Full cues: attn_do_jit **0.951±0.017** > concat-MLP 0.931±0.011 > best unimodal 0.732 > rules 0.695.
- attn_full (with recombination) 0.943±0.006, best macro-F1 0.649±0.005 (recombination gives F10
  supervision; lowest seed variance).
- **Negative result worth keeping:** token-substitution attention degraded WORSE under cue
  masking than the plain concat-MLP (gesture masked 0.63 vs 0.88). Fixes tried:
  (1) masked-val model selection (`select_masked`) + dropout 0.3 — helped;
  (2) `missing_mode='exclude'` (architectural marginalization) — attn_exclude ≈ dropout-trained
  MLP within noise on every mask. Conclusion: at 24-dim input, missing-cue robustness comes from
  dropout training + marginalization, NOT from a learned [MISSING] token — thesis ablation point.
- Emotion is the load-bearing cue (masking it costs ~28 points); emotion+gesture masked → ~0.30
  (genuinely ambiguous under the table's own rubric → F05 default territory).

**Next:** pick deployed config (recommend attn_exclude w/ dropout+jitter+recombination —
attention-weight interpretability for thesis, equal robustness); G3 spotlight table; window-size
sweep (W=8/16/32/64 from perframe caches — no re-extraction needed); trial ONNX export (4 models
+ fusion head); intent→action policy module.


> **Purpose.** Two machines work on this branch (`deploy`): the Windows laptop and the Ubuntu HPC.
> Code and docs sync via git; **checkpoints and datasets are gitignored and copied manually** — that
> mismatch is what caused past "model conflict" losses. This file + `checkpoint_manifest.sha256`
> exist so any human or Claude agent on either machine can see, in one place: what was done, on
> which machine, what broke, how it was fixed, and what state the artifacts should be in.

## Machines

| Tag | Machine | OS | GPU | Notes |
|---|---|---|---|---|
| `WIN-3060` | Personal laptop | Windows 11 | RTX 3060 Laptop, 6 GB | `.venv` (Python venv) is the GPU env — torch 2.5.1+cu121. Do NOT use system python. |
| `HPC-5090` | HPC workstation | Ubuntu | RTX 5090 (handover doc says 5060 — verify) | Where heavy training runs (gesture v2 was trained here). |

## Sync protocol (follow every session, both machines)

1. **Start of session:** `git pull`, then read the newest WORKLOG entry to see where the other machine left off.
2. **Verify artifacts:** `sha256sum -c docs/checkpoint_manifest.sha256` (Git Bash on Windows).
   If any hash fails, the local checkpoint is stale/different — copy the canonical one from the
   other machine **before** running anything. Never "just retrain" to make it match.
3. **After producing a new canonical artifact** (retrained model, new features parquet):
   - add/update its line in `docs/checkpoint_manifest.sha256`
   - add a WORKLOG entry saying which machine produced it and why
   - commit both, push, and copy the binary to the other machine when you next switch.
4. **End of session:** append a WORKLOG entry (template below), commit, push. Entries are
   **newest first**. Never rewrite old entries — corrections get a new entry.
5. Fusion feature extraction reads `data/features/manifest.json` (per handover §7.1); that manifest
   must name the exact checkpoint hashes used, so a features file built on one machine is
   reproducible/verifiable on the other.

### Entry template

```
## YYYY-MM-DD — [MACHINE] — Stage
**Did:** …
**Problems:** …
**Fixes:** …
**State / artifacts:** …
**Next:** …
```

---

## 2026-07-17 — [WIN-3060] — Fusion Phase 1: feature-extraction pipeline built & running

**Decision:** Phase 1 (extraction → baselines → fusion training) stays on WIN-3060 — the curated
dataset and verified checkpoints live here; the 5090 is only needed if a unimodal retrain becomes
necessary later.

**Did:**
- Built the two-pass extraction pipeline (`fusion/extraction/`):
  - **Pass 1** `scripts/02_extract_perframe.py` → `data/features/perframe/<clip_id>.npz` —
    one decode per clip: MediaPipe **Holistic** (one pass serves gesture image-landmarks AND
    motion world-landmarks), emotion's robust face detector + MobileNetV2 per frame, CLIP scene
    probs at ~3 Hz. Resumable (skips existing npz).
  - **Pass 2** `scripts/03_build_features.py` → `features_v1.parquet` + `manifest.json`
    (checkpoint sha256s, class orders, window params). Re-runnable without touching videos.
- **Windowing is TIME-based** (mixed 15/24/30 fps clips): stride 8/30 s; per-cue lookbacks match
  each model's training statistics — gesture 2.133 s→32 frames (mirrors ENGINE_BUFFER_FRAMES),
  motion 2.0 s→30 frames (dt≈1/15 like training), emotion mean over last 0.267 s (widen to 2.13 s
  before declaring missing), context mean over last 1 s. Per-cue runtime-missing = NaN probs +
  `*_obs=False` + coverage ratio — distinct from the scenario-designed `missing_designed` flag.
- Modality code reused via `fusion/extraction/modloader.py` (isolates the four packages'
  colliding `config`/`src`/`model` module names).
- Smoke test (3 clips): S01_F04 windows → Neutral 0.73 / raise_hand 0.995 / sitting 1.0 /
  classroom 0.996 — all four cues correct, all softmax sums = 1.0.

**Problems / fixes:**
- `.venv` had no pyarrow (and no pip — uv-managed): `uv pip install pyarrow --python .venv/Scripts/python.exe`.
- Context CLIP weights resolve from `jetson_deploy/hf_cache` via `HF_HOME` (set in perframe.py) —
  no download needed on either machine.

**State / artifacts (end of session):**
- Pass 1 complete: **1,061/1,061 clips extracted, 0 failures** (~2.2 h) → `data/features/perframe/`.
- Pass 2 complete: **`features_v1.parquet` = 15,892 windows**, obs rates emo 1.0 / ges 0.998 /
  mot 0.998 / ctx 1.0; `manifest.json` records checkpoint sha256s + class orders + window params.
- **Per-cue agreement vs intended labels** (windows, macro over scenarios): emotion 0.94,
  context 0.97, gesture 0.75, motion 0.81. Systematic failures confirmed: static `point` ~0
  (S03/S22), weak `wave` (S25 0.24), kitchen `stepping_back` ~0 (S19/S23/S26). S28 actors DID
  perform thumbs_down (0.76 detected) — validates pooling it instead of training F10 on it.
- **Baselines run** (`scripts/04_run_baselines.py` → `fusion/baselines/RESULTS.md`), actor-disjoint
  test subjects (82 clips), clip-level majority vote:
  | model | clip acc | clip macro-F1 |
  |---|---|---|
  | rule-based (V3 rubric, no direction) | 0.695 | 0.452 |
  | unimodal emotion | 0.732 | 0.436 |
  | unimodal gesture | 0.427 | 0.227 |
  | unimodal motion | 0.512 | 0.234 |
  | unimodal context | 0.146 | 0.150 |
  | **concat-MLP (3 seeds)** | **0.931±0.011** | **0.621±0.011** |
  G1 evidence already visible: naive fusion 93% vs best unimodal 73% vs rules 70%. macro-F1 is
  capped at 0.9 because F10 has zero supervised rows (S28 pooled) — recombination augmentation
  (Phase 2) is what fills F10.
- NOTE: this val/test = subject-splits of recorded TRAIN scenarios; the V3 test scenarios are
  unrecorded, so T01–T05 numbers are not yet measurable.

**Next:** attention fusion model (`fusion/model/`) + modality dropout, beat concat-MLP; then
recombination/jitter augmentation (fills F10 + unrecorded V3 rows); trial ONNX export of all
4 models + fusion head.

---

## 2026-07-16 — [WIN-3060] — Fusion Phase 0: audit & documentation bootstrap

**Did:**
- Pulled `deploy` from HPC; copied checkpoints + dataset to this machine.
- Full Phase 0 audit of the 4 unimodal models → `docs/MODEL_AUDIT.md` (frameworks, checkpoints,
  input/output specs, accuracies, contract discrepancies).
- Audited recorded data vs the Final_Dataset V3 table → `docs/DATASET_STATUS.md` (recorded-scenario
  → V3-row mapping, coverage gaps, label issues).
- Created this WORKLOG + `docs/checkpoint_manifest.sha256` (canonical checkpoint hashes).
- Verified `.venv` CUDA works on this machine (torch 2.5.1+cu121, RTX 3060 detected).
- Verified `jetson_deploy/` checkpoints are byte-identical to `modalities/` deployed ones.

**Problems found:**
1. `data/raw/clips` (1,061) vs `videos/struct` (1,270) mismatch — **resolved: intentional.** User
   manually curated out 209 bad clips (incl. all of S27_F06); `data/` is canonical, struct = archive.
2. **Recorded label collision** S21_F04 vs S28_F10 (identical cue tuple kitchen/sad/thumbs-down/sit).
   **User decision:** S21 trains as F04; S28 excluded → `recombination_pool` for synthetic generation.
3. **S05_F02 relabel** (V2 #5 → V3 #14, F02→F07). **User decision:** labels.csv carries F07.
4. **No model outputs motion direction** though V3 leans on it. **User decision:** skip for v1,
   revisit after error analysis.
5. Motion model head is **4 classes** (sitting, standing, walking, stepping_back) — handover §5.2
   said 5 (with `run`); V3 table §2.5 says "(6)" but lists 4. Actual head wins: 4, no `run`.
6. `clips.csv` metadata was wrong for **104 kitchen clips**: annotated 15 fps, actually 24/30 fps
   phone video in mixed resolutions (some portrait 1080×1920). Discovered via OpenCV probe.

**Fixes / work done on data:**
- `scripts/00_inventory.py` — probes every curated clip (first+last frame): **all 1,061 readable**,
  0 corrupt → `data/inventory.csv`.
- `scripts/01_update_annotations.py` — filtered `data/annotations/clips.csv` to the curated 1,061,
  merged `split_subject` from struct `splits.csv`, then fps/resolution/duration rewritten from probe.
- `data/labels.csv` created — per-scenario V3-corrected labels (22 scenarios: 21 train / S28 pool,
  1,008 training clips). Subject split survives curation: P01/P02/P06 train (886), P04 val (93),
  P03/P05/P07-09 test (82) — actor-disjoint.

**State / artifacts:** `data/labels.csv`, `data/inventory.csv`, updated `data/annotations/clips.csv`
(all gitignored — copy `data/` to HPC or rerun scripts 00/01 there against the same curated videos).
No fusion code yet (`fusion/rule-based/` empty). Canonical checkpoint hashes in
`docs/checkpoint_manifest.sha256`.

**Next:** per-modality window-level feature extractors on the (W, S) grid — **time-based windows
using per-clip probed fps** (clips are mixed 15/24/30 fps, deployment is 30 fps) → 
`data/features/features_v1.parquet` + `manifest.json` (checkpoint hashes) → baselines
(rule-based, unimodal, concat-MLP) → attention fusion. Trial ONNX export of all 4 models is also
due this week (handover §9).

---

## ≤2026-07-16 — [HPC-5090] — Prior unimodal work (reconstructed from reports; HPC sessions predate this log)

- **Emotion:** RAF-DB pipeline (restructured); EfficientNet_B0 / MobileNetV2 / MobileNetV2-LSTM
  trained + tuned; fine-tuned on real clips. Deployed: `finetuned_MobileNetV2.pth` — 92.5% acc /
  90.1% macro-F1 on held-out real video. Reports: `modalities/emotion/reports/`.
- **Gesture:** v1 replaced by v2 keypoint-sequence design (`modalities/gesture/reports/GESTURE_V2_DESIGN_AND_HPC_GUIDE.md`);
  TCN (683K params) trained on HPC 2026-07-15 — val 93.2% acc / 92.8% macro-F1; real-world 84.1% / 82.9%.
  Deployed: `best_TCN.pth` + `model_config.json`.
- **Motion:** skeleton-based classifier (14 joints, 84-dim features, window 30), fine-tuned;
  76.8% / 73.2% real-world. Deployed: `best_model_finetuned.pt`. Known weak: kitchen `stepping_back`.
- **Context:** CNN scene classifier (legacy, 3-class) superseded by **CLIP ViT-B-32 zero-shot**
  (5 scenes) — 98.8% / 99.3%; plus SmolVLM2-500M caption path (`modalities/context/src/`).
- **Dataset:** 23 old-table train scenarios recorded (~1,270 clips, 15 fps, 640×480), structured
  into `videos/struct/` with annotations incl. `splits.csv` (scenario/subject/leaky-random splits).
  Known incident 2026-07-13: kitchen clips were 0-byte after a copy; re-copied (0 zero-byte files now).
- **Jetson:** `jetson_deploy/` inference-only package built 2026-07-16, all 4 models load-verified.

---

## 2026-07-27 — [WIN-3060] — Unimodal + fusion audit on the complete classroom set (`data/final`)

**Ran:** `15_final_extract.py` (866 RealSense classroom clips, 90.5 min, 0 failures) →
`16_final_unimodal_eval.py` (12,203 windows) → `17_cue_ambiguity.py` → `19_final_fusion_eval.py`.
Phone views (572 clips) extracting after.

**New code:** `scripts/15_final_extract.py`, `16_final_unimodal_eval.py`, `17_cue_ambiguity.py`,
`18_final_subjects.py`, `19_final_fusion_eval.py`, `scripts/realworld_eval/final_unimodal.py`.
`fusion/extraction/perframe.py` now caches `pose_img`. `14_final_annotations.py` path fixed to
`data/old/labels.csv` after the data reorg.

**Headline (held-out `raw_take_20260725` clips, clip-level):** context 1.000 · gesture 0.766 ·
emotion 0.643 · motion 0.637. Training-overlap clips read 1.000 / 0.882 / 0.969 / 0.920 — the
emotion and motion figures were inflated by fine-tune overlap.

**Deployed fusion head, frozen, on this data:** train-design rows 0.887 (ceiling 0.977),
**test-design rows 0.388 (ceiling 0.900)**. Every failing row is a 2026-07-25 row; every
`curated_clip` row is ≥0.778. Cause is that fusion v1 never saw 18 of the 29 recorded classroom
rows, not a defect in the fusion design — under masking it sits at the table ceiling.

**Two findings that change the plan:**
1. **The V3 table itself caps classroom test accuracy at 0.900** — rows #22 (F01) and #30 (F09)
   are the identical cue tuple and differ only by walking direction. Confirmed empirically: row #1
   scores 1.000 and row #18 (its train-side twin) scores 0.000.
2. **The designed-missing rows are not missing at the sensor** — emotion observed on 99.7% of
   windows for rows #22/#25/#30, gesture on 100% for #12/#23. T03 currently measures simulated
   masking, and the deployed runtime would not mask these rows at all (detector succeeds).

**State / artifacts:** `results/realworld_eval_final/{ASSESSMENT.md,unimodal/,ambiguity/,fusion_transfer/}`,
`data/final/features/{perframe/,unimodal_windows.parquet}` (gitignored),
`data/final/annotations/{derived_rows.csv,subjects_pending.csv}`, `docs/methodology/04_missing_cues.md`.

**Next:** retrain fusion on `data/final` (biggest single gain), then emotion fine-tune targeting
Fear/Disgust/Sad, then motion fine-tune for `stepping_back`/`standing`. Blocked on user:
`subjects_pending.csv` person_ids (316 takes), `dilanka/` folder split, direction-cue go/no-go.

**Update (same day, +3.5 h):** phone-view extraction finished — 572 clips in 211.8 min, 0 failures;
all 1,440 classroom clips cached (217 MB). Per-view scoring on the **same synchronised takes**
shows a 36× pixel increase buys nothing: emotion 0.643/0.683/0.674 and motion 0.637/0.526/0.617
for RealSense-480p / phone-1080p / phone-4K. Motion is *worse* on the higher-resolution phones,
which a skeleton classifier cannot blame on resolution — supports the camera-framing hypothesis.
Conclusion: the weakness is not the sensor, so fine-tuning (not a camera upgrade) is the lever.
See `ASSESSMENT.md` §4b.
