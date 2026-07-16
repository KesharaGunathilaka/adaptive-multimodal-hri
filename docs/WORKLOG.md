# WORKLOG — cross-machine progress log

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
