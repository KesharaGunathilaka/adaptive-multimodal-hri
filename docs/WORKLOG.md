# WORKLOG — cross-machine progress log

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
