# CLAUDE.md — Project Handover: Multimodal Fusion for Adaptive HRI

> Place this file at the root of the project repo on the Ubuntu HPC as `CLAUDE.md`.
> Claude Code reads it automatically. It is the single source of truth for continuing this project.

---

## 1. Project overview

**Title:** Adaptive Human–Robot Interaction through Multimodal Emotion, Gesture, and Context-Aware Policy Learning (final year project, 4 members).

**Core idea:** Rule-based HRI is rigid. This system perceives 4 cues — **emotion, gesture, motion, context** — from video, fuses them to predict a human **intent** (F01–F10), and maps intent to a **robot action** (A01–A15). Must run on a **Jetson Orin Nano**.

**Research goals:**
- **G1** — Fusion of all 4 cues beats any single cue and beats rule-based logic.
- **G2** — Graceful degradation: correct intent even when 1–2 modalities are missing (sensor occlusion, motion blur, sensor offline).
- **G3** — Same gesture, different intent: emotion/context flips the meaning (e.g., raise hand + neutral + classroom = help request; raise hand + happy + kitchen = greeting).

**Test cases:** T01–T05 (fusion vs unimodal, cue conflict, missing modality, cross-context, robustness). Defined in the dataset table.

**Hard deadline:** ~3 weeks from 2026-07-13.

## 2. Current status (what is DONE)

- 4 unimodal models trained and working: **emotion, gesture, motion, context**. Some need improvement — but only improve one if fusion error analysis proves it is the bottleneck.
- Real-world dataset **fully collected**: ~39 scenarios × 50 videos ≈ 1,950 clips, organized as **one folder per scenario** on this machine.
- Dataset design table exists (`docs/HRI_Dataset_Table.docx`): per scenario → context, emotion, gesture, motion, missing-cue flags, intent label (F01–F10), robot action (Axx), test cases.

## 3. What is NOT done (your job)

1. Audit the 4 models and standardize their outputs. Take those all standarize outputs and files under the seperate root folder to maintain good folder structure (Phase 0). Cause keep 4 unimodal training files under the modalities seperately.
2. Fix known label inconsistencies in the dataset table and inform me (Phase 0).
3. Extract a feature table: every video → 4 probability vectors + label (Phase 1).
4. Build baselines: rule-based, CAM fusion model(channel attention fusion model), gradient boost tree in separete folder under the fusion to keep good folder structure (Phase 1).
5. Train the attention fusion model with modality dropout + augmentation (Phase 2).
6. Evaluate against T01–T05 (Phase 2).
7. Export everything to ONNX for Jetson Orin Nano deployment (Phase 3).
8. (Stretch) Rule-based intent→action lookup module (Phase 3).

## 4. Hardware / environment

- Ubuntu HPC, **RTX 5060**, 32 GB RAM. Plenty for this: feature extraction is the only GPU-heavy step; fusion training is tiny (CPU-capable). And also my windows pc with **RTX 3060**, 6 GB VGA. 
- Deployment target: **Jetson Orin Nano** (a different machine). Everything trained here must export to ONNX.
- Use a single venv environment using UV. Pin versions in `requirements.txt` from day one — the Jetson port depends on knowing exact versions.

## 5. Phase 0 — Audit & fixes

### 5.1 Model audit (do this FIRST, report findings before writing pipeline code)

The user does not know exactly which frameworks the 4 models use, and they process video at different granularities (per-frame / per-window / per-clip). For **each** of the 4 models, discover and document in `docs/MODEL_AUDIT.md`:

- Framework and version (PyTorch / TensorFlow / Keras / MediaPipe / sklearn / other) — find checkpoints (`.pt`, `.pth`, `.h5`, `.keras`, `.onnx`, `.pkl`) and training scripts on disk.
- Exact input spec: resolution, frames per input, preprocessing (normalization, cropping, landmark extraction).
- Exact output spec: number of classes, class names/order, logits vs softmax, per-frame vs per-clip.
- How to load and run inference (write a minimal working snippet per model and verify on 2–3 real videos).
- Rough accuracy claims from the team's prior training, if logs exist.

### 5.2 The output contract (standardize everything to this)

All downstream code assumes **one probability vector per WINDOW per modality** (sliding-window, not per-clip — deployment is streaming on Jetson + RealSense at 30 fps). But our real sense video clips are 15 fps.

**Windowing defaults (confirm via the sweep in §8.3):**
- **Window W = 32 frames (~1.07 s @ 30 fps)** for gesture and motion — long enough to capture a wave/beckon/walking direction.
- Emotion: aggregate the last 8 frames within each window step (near-instant cue).
- Context: recompute every 30 frames (near-static cue); hold the last value between updates.
- **Stride S = 8 frames (~0.27 s)** → a new fused intent every 8 frames ≈ **3.75 Hz** update rate.
- If a model's native input length differs (found in the audit), adapt its internal processing but still emit one vector per (W=32, S=8) grid step so all 4 modalities align on the same timeline.

| Modality | Classes (from dataset table — VERIFY against actual model heads) | Dim |
|---|---|---|
| Emotion  | neutral, happy, sad, angry, disgust, fear, surprise | 7 |
| Gesture  | raise_hand, wave, point, thumbs_up, thumbs_down, both_hands_up, beckoning, neutral(none) | 8 |
| Motion   | sitting, stand, walk, run, stepping_back | 5 |
| Context  | classroom, kitchen, Hospital, Museum, Shop | 5 | (model training dataset include only classroom, kitchen)

- If a model is per-frame or per-window: aggregate to per-clip by **mean of softmax vectors** (simple, robust). Store the aggregated vector.
- Always store **softmax probabilities, never argmax labels**.
- If the actual class lists differ from the table above, the actual model heads win — update this contract and the table mapping, and record it in `docs/MODEL_AUDIT.md`.

### 5.3 Dataset table fixes (needs user/team confirmation — ASK, don't guess)

I update the data table and it store under the docs/Final_Dataset.docx
Check the data table and if there is any issue let me know.

Convert the fixed table to `data/labels.csv` (machine-readable, one row per scenario).

### 5.4 Verify data inventory

Recorded and structured data (according to old data table) store under the `data/` folder. Read csvs under the annotations.
Script `scripts/00_inventory.py`: walk the scenario folders, count videos per scenario, check readability (open with OpenCV, read first/last frame), flag corrupt files and scenarios with <50 clips. Output `data/inventory.csv`. **If actor identity is recoverable** (filename, folder, or ask user), record `actor_id` per video — required for the actor-disjoint split (§7.3).

## 6. Project structure (create this)

```
hri-fusion/
├── HANDOVER_CLAUDE.md                  # this file
├── requirements.txt
├── docs/
│   ├── HRI_Dataset_Table.docx # original design table
│   ├── MODEL_AUDIT.md         # Phase 0 findings
│   └── DECISIONS.md           # log every design decision + date
├── data/
│   ├── raw/                   # symlink to scenario video folders (never modify)
│   ├── labels.csv             # scenario_id → context, cue labels, intent, action, split, test_cases
│   ├── inventory.csv
│   └── features/
│       ├── features_v1.parquet  # THE feature table (see §7.1)
│       └── manifest.json        # which model checkpoints/versions produced it
├── models/
│   ├── unimodal/              # 4 existing checkpoints + loading wrappers
│   │   ├── emotion/  ├── gesture/  ├── motion/  └── context/
│   └── fusion/                # trained fusion checkpoints + ONNX exports 
├── src/
│   ├── extraction/            # one extractor per modality + orchestrator
│   ├── fusion/                # datasets.py, models.py, augment.py, train.py
│   ├── baselines/             # rule_based.py, unimodal_eval.py, concat_mlp.py
│   ├── eval/                  # metrics.py, testcases.py (T01–T05), report.py
│   └── actions/               # intent→action lookup (stretch goal)
├── scripts/                   # numbered entry points: 00_inventory, 01_extract, 02_train_fusion, 03_eval, 04_export_onnx
├── experiments/               # one folder per run: config.yaml, metrics.json, checkpoint
└── results/                   # final tables/figures for the thesis
```

## 7. Phase 1 — Feature extraction & baselines

### 7.1 Feature table (the key efficiency mechanism)

Run all 4 models over all ~1,950 videos **once** with sliding windows (W=32, S=8 per §5.2), save `data/features/features_v1.parquet`, **one row per window**:

```
video_id, scenario_id, actor_id, split, context_gt,
window_idx, start_frame, end_frame,
emotion_probs (7), gesture_probs (8), motion_probs (6), context_probs (2),
missing_flags (4 bool, from table's Missing column),
intent_label (F01–F10), action_label, test_cases
```

A ~5 s clip yields ~15 windows → ~30K training rows instead of 1,450 (a free data multiplier). Per-clip vectors, when needed, are just the mean over a clip's windows — so nothing from the per-clip plan is lost. Windows within one video are correlated: the actor-disjoint split rule (§7.3) matters even more, and clip-level accuracy (majority vote over a clip's window predictions) is the headline metric; window-level accuracy is secondary.

Rules:
- Keep **per-model extraction scripts separate**, so when one unimodal model is retrained, only its columns are regenerated → `features_v2.parquet`. Never overwrite old versions; record checkpoint hashes in `manifest.json`.
- All fusion development after this point reads ONLY the parquet — never touch videos again during fusion iteration.
- For scenarios whose table row marks a cue as `[MISSING]`, still run the model if the video allows, but set the missing flag; at training/eval time the flag controls masking.

### 7.2 Baselines (must exist BEFORE the fusion model — they are the evidence for G1)

1. **Rule-based**: hand-coded if/else over argmax cue labels, mirroring the table logic. Expected to fail on missing/conflict rows — that failure is a thesis result, document it per-row.
2. **Unimodal**: predict intent from each single modality's probabilities (small logistic regression / MLP per modality).
3. **Concat + MLP**: concatenate 23-dim input → 2 hidden layers (128, 64) → 10-class softmax. This will already be decent; it is the floor the attention model must beat.

### 7.3 Split protocol (critical for validity)

- Train/test scenario split is **fixed by the table** (Train rows / Test rows). Never move scenarios across.
- Within train scenarios, make the validation split **actor-disjoint**: all clips of a given person are entirely in train or entirely in val. If actor IDs are unrecoverable, split by recording session/date as proxy, and note the limitation in `DECISIONS.md`.
- Test set: **100% real, never augmented, never used for any tuning decision.**

## 8. Phase 2 — Fusion model, augmentation, evaluation

### 8.1 Fusion architecture (attention over modality tokens)

- Each modality's probability vector → linear projection to d=64 → add learned modality embedding → 4 tokens (+1 CLS token).
- 1–2 transformer encoder layers (4 heads, d=64, FFN 128, dropout 0.1–0.3).
- CLS token → linear → 10-class intent softmax.
- ~100–300K params. Trains in seconds–minutes on the 3060; also fine on CPU.
- **Missing modality handling:** a learned `[MISSING]` token replaces any masked modality. At inference, low unimodal confidence or the missing flag triggers masking.

### 8.2 Training augmentation (applied on-the-fly to the feature table; TRAIN ONLY)

1. **Modality dropout:** each modality independently masked with p≈0.2 per sample → trains G2 robustness directly.
2. **Cue recombination:** build synthetic samples by combining real cue vectors from different videos into new combinations; label via the table's cue→intent logic. My new data table that store under the docs; that all rows are not recorded. For that I think we can use cue recombination to fill those rows. If I am wrong fix me and give me guide.
3. **Confidence jitter:** Gaussian noise on logits / temperature scaling on softmax → simulates sensor noise.
4. Ablate: report fusion with vs without each augmentation — this is a thesis contribution ("data-efficient / few-shot" claim).

### 8.3 Evaluation protocol (implement in `src/eval/testcases.py`)

Report per model (rule-based, 4× unimodal, concat-MLP, attention fusion):
- Overall intent accuracy + macro-F1 (macro-F1 matters for rare) + confusion matrix, on the real test set.
- **T01/T04** fusion vs unimodal comparison; **T02** conflict rows subset accuracy; **T03** missing-modality rows subset accuracy, plus systematic masking sweep (drop each modality, then pairs, on the full test set); **T05** as defined by the team — confirm its exact definition with the user before implementing.
- G3 spotlight: rows 1/11/33-style same-gesture triads — show the per-row predictions in a small table for the thesis.
- **Window-size sweep:** re-extract (or sub-sample) features at W = 8, 16, 32, 64 (stride 8) for a subset of modalities/videos; plot validation accuracy vs window length per modality; confirm or revise the W=32 default. Include the plot in the thesis (justifies the deployment latency choice).
- Every experiment: config + seed + metrics saved under `experiments/`. Fix seeds; report mean±std over ≥3 seeds for headline numbers.

### 8.4 Unimodal improvement loop (only if justified)

After first fusion results, compute per-modality reliability (accuracy of each unimodal model on test). Only retrain a unimodal model if fusion error analysis attributes failures to it. After retraining → regenerate that modality's columns → `features_v2.parquet` → rerun fusion (cheap).

## 9. Phase 3 — Jetson export & actions

- Export all 4 unimodal models + fusion head to **ONNX** (opset ≥ 13). Verify ONNX outputs match native outputs (tolerance 1e-4) on 20 sample clips. If any model's framework resists ONNX export, flag immediately — that model may need re-implementation, and this is the schedule's biggest risk. **Do a trial ONNX export of all 4 models in Week 1** (right after the audit), not in Week 3.
- Fusion head: also export a version taking raw probability vectors as input — on the Jetson it runs after the 4 models regardless of their runtimes.
- Provide a `deploy/` folder: ONNX files, preprocessing code per modality, and a streaming `pipeline.py` for Jetson + Intel RealSense (RGB @ 30 fps; ignore depth — models weren't trained on it):
  - Rolling frame buffer per modality; every **stride S=8 frames**, run the 4 models on their windows (W per §5.2) → fusion head → intent, i.e. a fresh intent at ~3.75 Hz. If Orin Nano profiling can't sustain that, increase S (to 12–16), never shrink W.
  - **Temporal smoothing:** majority vote over the last 3 fused outputs, plus hysteresis — switch the active intent only after 2 consecutive agreeing predictions. Prevents intent flicker in the demo.
  - **Emergency bypass:** F02 skips smoothing/hysteresis — act on the first prediction where P(F02) exceeds its (deliberately lower) threshold. Latency there is a safety issue.
  - Target: end-to-end (capture → intent) latency ≤ 300 ms per stride step on Orin Nano (FP16); document HPC profiling numbers as reference.
- **Stretch — actions:** `src/actions/policy.py`: dict lookup intent (+context) → action A01–A15 from `labels.csv`, with a confidence threshold: if max fusion softmax < τ (tune on val, ~0.5), fall back to F05 Neutral pass / safe action. Safety rule: **F02 Emergency must have the lowest miss rate** — check its recall explicitly; if needed, lower its decision threshold.

## 10. Schedule

After each work document each work seperately to understand what do in that phase or task in understandable way. 
Make a complete pipeline of complete project what pipeline follow our work to understand and futher documentation and methodology explanations and if anything changes update it.

| 1 | Phase 0: model audit, table fixes confirmed with team, `labels.csv`, inventory, repo scaffold |
| 2 | Feature extraction complete (`features_v1.parquet`); trial ONNX export of all 4 models |
| 3 | Baselines done: rule-based, unimodal, concat-MLP + first results table |
| 4 | Attention fusion + modality dropout; beats baselines on val; T03 masking sweep |
| 5 | Augmentation (recombination, jitter) + ablations; T02 conflict results |
| 6 | Full T01–T05 evaluation on real test set; unimodal retraining only if error analysis demands; freeze `features_vN` |
| 7 | ONNX pipeline + `deploy/` package; HPC latency profile |
| 8 | Intent→action module + end-to-end demo script |
| 9 | Buffer: final results tables/figures for thesis, demo video, reproducibility check (fresh-clone run of scripts 00→04) |

## 11. Rules & guardrails

- First of all read the Final_Dataset doc under the `doc/`
- Document each of every step with timestamps to identify by anyone and any AI agent to what are the latest approches and what are mistakes or improvements previously did and we can have a clear pipeline to identify which we follow. 
- **Never** train end-to-end through all 4 unimodal models jointly — no time, will overfit ~1,450 clips.
- **Never** put synthetic/augmented samples in the test set or tune on test.
- **Never** split train/val with the same actor on both sides.
- Dataset is frozen — no new recording. Test set will record and include. For new datarows fill those with exist data rows synthetilcally in suitable way.
- Every non-obvious choice goes in `docs/DECISIONS.md` with one-line rationale (the thesis writer will need it).
- Ask the user before: changing any dataset label, dropping a scenario, redefining the output contract, or spending >1 day retraining a unimodal model.

## 12. Open questions for the user (ask early)

1. Where exactly on this machine are the 4 model checkpoints and the scenario video folders? (paths)
all are include in the project folder.

2. Exact definitions of test cases T01–T05 (the table references them; the definitions live with the team).
all are include in the dataset document. Read it carefully and understand the full content.

3. Are actor IDs recoverable per video (filenames / folder / memory)?
yes. in data folder annotation document has each video person id.

4. Confirm the class lists per modality (§5.2) against the actual trained model heads.
I dont have idea about this question.