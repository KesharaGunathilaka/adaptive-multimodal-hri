# DECISIONS — one-line rationale log (append-only, newest first)

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
