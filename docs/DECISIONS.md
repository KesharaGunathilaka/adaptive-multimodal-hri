# DECISIONS — one-line rationale log (append-only, newest first)

- **2026-08-08 [WIN-3060]** Embedding-level fusion is **not adopted**. Tested four ways (plain,
  full-recipe in-fold, full-recipe out-of-fold, all vs the 24-dim probability baseline, 10 seeds
  each): plain loses significantly (p=0.0014), full-recipe ties (p=1.0000), out-of-fold "fixed"
  pools make it WORSE (p=0.0544, and significantly worse than rules at p=0.0480). Root cause is
  likely dimensionality/sample-size (2176 raw dims vs ~1547 train clips), evidenced independently
  by `EMBEDDING_PROBE.md` (motion's linear-probe train/test gap exceeds the fine-tuned cues'
  despite motion never being fine-tuned). `PERCEPTION_BAND.md`'s perception band remains the open
  problem; this was a tested, ruled-out lever for it, not a confirmed one.
- **2026-08-08 [WIN-3060]** Out-of-fold retraining for pool-purity contamination targeted
  **gesture and emotion only** — motion and context were never fine-tuned on train and are not
  contaminated (`POOL_PURITY.md`: motion's train/test gap is already negative). New checkpoints
  (`*_fold{0,1}.pth`) are 2-fold, scenario-stratified, warm-started from deployed, and are
  diagnostic artifacts only — never promoted, never touching the deployed or `_merged` checkpoints.

- **2026-08-08 [WIN-3060]** **Headline numbers require ≥10 seeds and a paired significance test.**
  The measured per-seed σ is 0.0160, so 3 seeds resolve nothing below ~0.079 — larger than every
  gap this project has claimed. `scripts/49_significance.py` (McNemar exact + clip bootstrap +
  t-CI over seeds) is the standard; helpers live in `scripts/realworld_eval/stats.py`. Per-seed
  values must be persisted (Studies 1–3 saved only mean/std and so cannot be re-analysed). See
  `docs/methodology/06_fusion_model.md` §6.10.
- **2026-08-08 [WIN-3060]** The claim "fusion beats rules on accuracy" is **withdrawn**. 10-seed
  fusion 0.7133 ± 0.0160 vs rules 0.7120, McNemar p=1.0000, bootstrap CI containing zero, 4/10
  seeds favouring rules, and rules ahead on macro-F1 (0.6414 vs 0.6146). The prior 0.7191 was a
  3-seed artifact driven by seed 0. T05 must be argued on missing-cue robustness and F02 recall.
- **2026-08-08 [WIN-3060]** `GAP_DECOMPOSITION_MERGED.md` is **superseded** by `PERCEPTION_BAND.md`
  for anything about where the error lives: on the deployed `full` recipe fusion+oracle is 0.9373
  (not 0.619), so generalisation cost is 0.063 and perception cost 0.224 — the reverse of the
  standing conclusion. Fusion-architecture work has ~0.06 of headroom; the perception band holds
  the rest.
- **2026-08-08 [WIN-3060]** Recombination allocation stays **`uniform` (unchanged default)** for
  now. `allocate()` adds `sqrt`/`intent`/`family` modes; `intent` won on val macro-F1 and beats
  uniform on test (0.7299 vs 0.7134), and `family` scored best of all on test (0.7354 / macro-F1
  0.6425, the only mode above rules' 0.6414) — but `family` LOST on val, so adopting it would be
  tuning on test. Promote only after a pre-registered re-test at ≥10 seeds. Default left unchanged
  so all 14 existing callers reproduce byte-identically.
- **2026-08-08 [WIN-3060]** T04 is reported with its structural limit stated up front: context
  changes the intent for only 24/224 cue tuples and **exclusively for `raise_hand`**, whose test
  purity is 0.327. The flip subset is n=52 from one row in one context. Report
  `flip_correct_both` (right in BOTH rooms), never `flip_followed` alone — the latter is inflated
  by F01 majority-class bias, since row #25's kitchen answer *is* F01.
- **2026-08-08 [WIN-3060]** Known, unfixed: **recombination pools are built from the split that
  emotion and gesture were fine-tuned on**, so gesture pool purity is 1.000 in training vs 0.864
  at test (`POOL_PURITY.md`). Fusion is trained to over-trust exactly the cues whose pools were
  artificially clean — which matches the cue-attribution ordering. Val pools are NOT the fix
  (gesture val purity 0.997); out-of-fold pooling or calibrated per-class noise is.

- **2026-08-07 [WIN-3060]** F02 class-weighted loss (`scripts/46_f02_recall_fix.py`) is a
  **validated fix, NOT YET promoted to the deployed checkpoint**. weight=3.0 crosses rules' F02
  recall (0.736 vs 0.727) for a -0.0146 headline-accuracy cost; weight=5.0 reaches 0.818 recall for
  -0.016 accuracy but with worse recall-at-matched-precision than weight=3.0 (0.678 vs 0.736 at
  rules' own precision level — a wider ceiling bought with more false alarms, not a strictly better
  operating point). **Recommend weight=3.0 if/when this is promoted** — smallest accuracy cost that
  still beats rules, best recall at a comparable false-alarm rate. Held back from promotion pending
  a decision on whether the accuracy trade-off is acceptable for the thesis's headline number, and
  whether it should apply project-wide or only to a safety-specific deployment variant.
- **2026-08-06 [WIN-3060]** The "fusion beats rules on accuracy" framing is **retired** as the
  project's primary claim, replaced with a narrower, better-evidenced one: fusion matches rules on
  clean data, wins clearly on missing-cue robustness, and (after the F02 fix above) has a tunable
  safety capability rules cannot have. Root cause: V3 intent labels ARE `rule_intent()`'s output, so
  rules-given-true-cues are Bayes-optimal by construction (rules+oracle=1.000, exact ceiling) — a
  3-part robustness battery (context fairness, degradation sweep, F02 operating point) and a
  sequence-recombination retry of R3 were run to find a fair arena for fusion; 3 of 4 came back
  negative-or-tied, only the F02 gap was real, specific, and (per the entry above) fixable. See
  `docs/WORKLOG.md` 2026-08-06/07 and the `fusion-vs-rules-investigation` memory for full detail.
- **2026-08-05 [WIN-3060]** Fusion architecture stays **self-attention + clip-pool (R1)** —
  Study 1 (`FUSION_ARCHITECTURES.md`) tried GMU/LMF/cross-attention/channel-attention/GBT against
  it and none beat it, confirming the gap decomposition's implication that fusion capacity was
  never the bottleneck. GMU noted as the fallback if Jetson params/latency get tight (0.7065 vs
  0.7191, 8x fewer params, far lower seed variance). Gesture/motion lookback span narrowed toward
  the **x0.5 scale** (1.07s/1.0s) — Study 3 re-confirmed §7.9's old "shorter is better" finding on
  `final_merged` + recombination (0.7252 vs deployed x1.0's 0.7191). Order-aware temporal sequence
  modeling (R3) is explicitly **not** adopted or ruled out — its Study 2 numbers are confounded by
  the lack of a recombination analogue for sequences, not a fair test of the idea.
- **2026-08-03 [WIN-3060]** `scripts/29_merged_gap_decomposition.py`'s `rule_predict` remaps a
  predicted F09 → F01 for `data/final_merged` only. Diagnosed: the shared
  `fusion/baselines/rule_based.py::rule_intent()` still has a `wave+walking+non-happy→F09`
  branch, correct for `data/old` (which has F09) but dead/wrong for the merged table (F09
  folded into F01). This single branch was the ENTIRE 0.098 shortfall between rules+oracle and
  the 1.0 ceiling (rows #22, #61, 96/979 headline clips) — confirmed by remapping alone taking
  rules+oracle from 0.902 to exactly 1.000. Not applied to the shared `rule_intent()` since
  `data/old`'s baseline still needs the F09 branch.
- **2026-08-03 [WIN-3060]** Oracle-cue construction fixed to give masked cues a REASONED
  default (emotion→Neutral, gesture→idle, motion→standing — the V3 table's own stated safe
  fallback) rather than `argmax` of an all-zero one-hot vector (which silently picks class
  index 0, e.g. "Surprise" for emotion — arbitrary, not a design choice). Verified this dataset's
  masked-emotion rows never combine with `gesture=both_hands_up` (the only branch where the two
  approaches would diverge), so it did not change this run's numbers, but is the correct
  contract going forward.

- **2026-08-03 [WIN-3060] (user)** Headline test metrics exclude row #58's 24 derived
  clips (`headline_eval=False` in splits.csv): they are row #49's TRAIN footage re-used
  with emotion+gesture masked, and the mask flips the intent F01→F06. #58's own 19
  clips stay. Quote the full-test number alongside the headline.
- **2026-08-03 [WIN-3060] (user)** Report **all views**, not RealSense-only — 3.6× more
  test data, and the resolution spread is evidenced not to matter (ASSESSMENT §4b: a 36×
  pixel increase changed accuracy by <0.04). Truthful `resolution_class` and `orientation`
  columns added because `view` mislabels 726/1,015 phone clips; per-view comparisons must
  also note that view is confounded with context.
- **2026-08-03 [WIN-3060] (user)** Clips shorter than the 4 s aggregation window are
  aggregated over their **full length**, never padded — padding invents frames. The span
  actually used is recorded per clip in `agg_span_s` (127 clips, min 1.67 s).
- **2026-08-03 [WIN-3060]** `splits.csv` gains `split` (train/val/test) beside
  `split_design`; val = actors P04+P03 carved from train, actor-disjoint and take-grouped.
  **Training code reads `split`.**

- **2026-08-01 [WIN-3060] (user)** The V3 table is fixed **at source** in
  `docs/final_dataset_merged.docx`, not worked around in code — so
  `merged_common.TABLE_OVERRIDES` is empty and the parser reads the document literally.
  Pre-fix copy kept at `final_dataset_merged.docx.bak`. Two scripts now enforce this:
  `21_validate_table.py` (the .docx against its own legend) and `22_verify_csvs.py`
  (41,848 field comparisons, .docx vs CSVs, using an independent parser so a bug in the
  production one is caught rather than mirrored).
- **2026-08-01 [WIN-3060] (user)** `#18` → `T01` and `#32` → `T01, T04`; neither gets `T05`,
  because that tag means "naive per-cue rules fail" and post-F09-merge the `wave→greet` rule
  *succeeds* on #18 while #32's aligned thumbs-up was never a rule failure.
- **2026-08-01 [WIN-3060]** OPEN ISSUE, not yet decided: **F01 maps to three actions**
  (A01/A09/A10) and rows #1 (train, A01) and #18 (test, A09) are the identical observable cue
  tuple. Deleting F09 moved the direction ambiguity from the intent level to the **action**
  level rather than removing it; an `intent → action` policy makes A09/A10 unreachable.
- **2026-08-01 [WIN-3060]** The 2026-07-16 **S21/S28 collision is resolved** by this dataset:
  row #63 *is* old `S28_F10`, correctly relabelled F04, and #51 was re-recorded gesture-free
  on 2026-07-28 so F10 keeps its own footage. `data/old/labels.csv`'s `recombination_pool`
  status is obsolete. Row #50's "VERIFY against recordings" is closed — the actors perform no
  thumbs down, so `gesture = idle` is correct.

- **2026-07-31 [WIN-3060] (user)** `data/final dataset merged` renamed to **`data/final_merged`**
  and adopted as the canonical root; `data/final` kept as the archive (it still holds the row-30
  and dropped-`dilanka` clips). Spaces in the old name broke shell paths.
- **2026-07-31 [WIN-3060] (user)** Where the V3 table contradicts itself, **the cue columns win
  and only cues + intent are read** — applied to **#57** (Missing/G2/T03 dropped) and **#63**
  (Missing `context` dropped, motion `sitting`→`sit`). Cost, accepted: unmasked, both rows now
  repeat a training tuple, so test rows with an unseen cue combination fall 21→19 of 22 and T03
  loses both. Tracked in `annotations/INTEGRITY.md`.
- **2026-07-31 [WIN-3060] (user)** V3 row **#30 stays retired**; its 48 clips are not carried into
  `data/final_merged` (they remain under `data/final`). Post-F09-merge the row was cue-identical
  to #22, so keeping it would add a duplicate, not a distinction. Nothing downstream needs them.
- **2026-07-31 [WIN-3060] (user)** The 24 clips shared by `S49_F01` and `S58_F06` make **#58 a
  derived row**: #49's footage with emotion+gesture masked. Taken over three recorded objections —
  #49 is *train* and #58 is *test* (same frames both sides), the mask *changes* the intent
  (F01→F06) where #6/#9 preserve it, and #49 walks toward the exit while #58's rationale requires
  walking toward the robot. `derived_rows.csv.crosses_split` flags it; score #58's 24 derived
  clips separately from its own 19.
- **2026-07-31 [WIN-3060]** V3 **#6/#9 are derived, not independent**: the recording team's own
  `S06.txt`/`S09.txt` confirm they are S05/S08's videos with a channel masked. Extract features
  once from the source row and mask at load — treating them as separate clips would duplicate
  every window and leak identical frames across rows.
- **2026-07-31 [WIN-3060]** Clip provenance (`source`, `old_clip_id`, `recorded_at`, `person_id`)
  is recovered by **SHA-256**, not filename: the merge renamed every file to
  `S{row}_F{intent}_c{NNN}`, which also made `14_final_annotations.py`'s folder<->row check pass
  vacuously. `20_merged_annotations.py` re-derives the old scenario IDs by hash and re-runs the
  real check (23 migrated scenarios, 0 mismatches).

- **2026-07-20 [WIN-3060]** `protobuf==3.20.3` + `onnx==1.14.1` pinned — newer protobuf breaks
  mediapipe 0.10.x; pins recorded in requirements.txt for the HPC/Jetson.
- **2026-07-20 [WIN-3060]** Context/CLIP not exported to ONNX — stays PyTorch on Jetson from the
  bundled HF cache; only revisit if the latency budget fails.
- **2026-07-20 [WIN-3060]** Policy thresholds: τ=0.5 (fallback F05/A06), F02 bypass τ=0.30 —
  gives 22/22 clip-level emergency detection at 1.3% false-emergency on test subjects.
- **2026-07-20 [WIN-3060]** Deployed fusion keeps ×1 lookbacks (matches unimodal engines'
  validated buffers) even though ×0.5 scored higher offline — decide after Jetson profiling.

- **2026-07-17 [WIN-3060]** Fusion model selection uses **masked-val early stopping** (mean of
  unmasked + 4 single-masked val accuracies) — plain val selection picked robustness-poor epochs.
- **2026-07-17 [WIN-3060]** Recombination skips V3 #18 (classroom happy+wave+walk F09): without
  direction it is cue-identical to recorded #1 (F01) — synthesizing it would inject label noise.
- **2026-07-17 [WIN-3060]** Missing cues handled by **attention exclusion** (key-padding mask)
  rather than a learned [MISSING] token — token substitution measurably hurt masked accuracy.

- **2026-07-16 [WIN-3060] (user)** S21/S28 label collision: S21 → F04 training; S28 (53 clips) →
  `recombination_pool` (synthetic-generation inputs), promotable to F10 if gesture model says `idle`.
- **2026-07-16 [WIN-3060] (user)** S05 relabeled F02 → F07 per V3 row #14 (folder name unchanged).
- **2026-07-16 [WIN-3060] (user)** Direction cue skipped for fusion v1; revisit via trajectory
  feature only if error analysis shows F01/F09 or F03/F06 confusion dominates.
- **2026-07-16 [WIN-3060] (user)** `data/` is the canonical **curated** dataset (bad clips manually
  removed, all of S27_F06 dropped); `videos/struct/` kept only as the uncurated archive.
- **2026-07-16 [WIN-3060]** `clips.csv` fps/resolution/duration rewritten from OpenCV probe —
  104 kitchen clips had wrong fps (24/30 fps phone video, not 15); windowing must be time-based
  using per-clip probed fps.
- **2026-07-16 [WIN-3060]** Fusion feature table stores each cue in its model's **native class
  order** (see MODEL_AUDIT.md) — remapping to table names happens once at extraction, avoiding
  silent order mismatches between machines.
- **2026-07-16 [WIN-3060]** Cross-machine artifact integrity enforced via
  `docs/checkpoint_manifest.sha256` + WORKLOG entries; git carries hashes, binaries copied manually.
- **2026-07-16 [WIN-3060]** Fusion consumes the CLIP zero-shot scene classifier only (5-dim); the
  SmolVLM2 caption path is out of scope for the fusion cue vector.
- **2026-07-27 [WIN-3060]** V3 #6 derived from #5 (mask context) and #9 from #8 (mask gesture)
  rather than recorded: the rows differ only by a *sensor state*, which no video can depict.
  Spec in `data/final/annotations/derived_rows.csv`; train-design rows only, splits follow the
  source clip, never scored. Rationale + the four rules in `docs/methodology/04_missing_cues.md`.
- **2026-07-27 [WIN-3060]** Designed-missingness is read from **both** the V3 cue column
  (`[missing]`) and the V3 `Missing` column. Context can only express it the second way (#6/#24/#56
  name a real room while Missing says `context`), so cue-column-only parsing silently left the
  context token observed on those rows. Fixed in `scripts/realworld_eval/final_unimodal.py`.
- **2026-07-27 [WIN-3060]** `fusion/extraction/perframe.py` now also caches `pose_img` [T,33,4],
  the raw image-space MediaPipe pose. `gesture_feats` divides out the mid-shoulder centre and the
  shoulder-width scale, so approach/recede is unrecoverable from it; caching the raw landmarks
  keeps a direction cue possible without re-decoding 1,440 videos. Backward compatible (old npz
  simply lack the key).
- **2026-07-27 [WIN-3060]** Unimodal accuracy on `data/final` is reported split by `source`:
  `curated_clip` clips were migrated from `data/old` and the emotion/motion models were fine-tuned
  on them, so only `raw_take_20260725` is a generalisation estimate (emotion 0.969 → 0.643,
  motion 0.920 → 0.637). Pooling the two overstates both models.
- **2026-07-27 [WIN-3060] (user)** Direction cue: decision deferred until unimodal results were in
  (they now are — see `results/realworld_eval_final/ASSESSMENT.md` §3.5, still open).
- **2026-07-27 [WIN-3060] (user)** All three camera views to be extracted for classroom; RealSense
  480p remains the deployment view and carries the headline numbers.
- **2026-07-27 [WIN-3060]** Camera resolution ruled out as a cause of the unimodal weakness: the
  three synchronised views of the same takes agree within ±0.06 accuracy (emotion, motion), and
  motion is *worse* at 1080p than at 480p. Deployment stays on the RealSense 640×480 view.
