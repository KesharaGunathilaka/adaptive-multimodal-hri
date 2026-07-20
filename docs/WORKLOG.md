# WORKLOG — cross-machine progress log

## 2026-07-17 (later) — [WIN-3060] — Phase 2: attention fusion, augmentation, T03 sweep

**Did:**
- Built `fusion/model/`: `model.py` (attention fusion, 4 cue tokens + CLS, d=64, 2 layers, ~110K
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
