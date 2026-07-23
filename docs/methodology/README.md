# Methodology — Adaptive Multimodal HRI

This folder is the **narrative explanation** of the project: what we did at each
stage, *why*, and how the pieces connect. It is written to be read top-to-bottom
by a person (thesis writer, examiner, or a new teammate) with no prior context.

It is deliberately separate from the operational records:
- `docs/WORKLOG.md` — dated, machine-tagged log of *what happened when* (two PCs).
- `docs/DECISIONS.md` — one-line rationale for each design choice.
- `docs/MODEL_AUDIT.md`, `docs/DATASET_STATUS.md` — Phase-0 reference tables.
- `results/fusion_v1/` — the actual numbers and figures.

The methodology docs cite those for evidence but stand on their own as an
explanation. If a number here disagrees with `results/`, `results/` wins (it is
regenerated from code); tell me and I'll fix the prose.

---

## 1. The problem in one paragraph

A service robot must react to a person, but rigid *if-this-gesture-then-that*
rules break the moment a cue is ambiguous, missing, or contradicted by another
cue. A raised hand means "I need help" in a classroom with a neutral face, but
"I'm participating" with a happy face, and "hello" in a kitchen. This project
perceives **four cues** — emotion, gesture, motion, context — from ordinary RGB
video, and **fuses** them into one of ten **intents** (F01–F10), which a policy
layer maps to a robot **action** (A01–A15). It must run in real time on a Jetson
Orin Nano.

## 2. Three research goals (everything traces back to these)

| Goal | Claim | How we test it |
|---|---|---|
| **G1** Fusion necessity | Weighing all four cues beats any single cue and beats rule-based logic. | Baselines vs fusion on the same test set. |
| **G2** Robustness | Correct intent even when 1–2 cues are missing/degraded. | Modality-masking sweep (T03). |
| **G3** Generalization | Meaning flips with context / emotion–gesture conflict / unseen combos. | Same-gesture spotlight, conflict rows. |

## 3. The full pipeline at a glance

```
                          ┌─────────────── per RGB frame ───────────────┐
   RGB video / RealSense  │  MediaPipe (face + Holistic keypoints)      │
        30 fps            │        │            │                       │
                          │   face crop     pose+hands / world joints   │
                          ▼        ▼            ▼             ▼         │
                     ┌─────────┐┌────────┐┌────────┐┌──────────────┐    │
      4 UNIMODAL     │ Emotion ││Gesture ││ Motion ││ Context/CLIP │    │
        MODELS       │MobileNet││  TCN   ││  LSTM  ││  zero-shot   │    │
                     └────┬────┘└───┬────┘└───┬────┘└──────┬───────┘    │
                          │ 7-dim   │ 8-dim   │ 4-dim      │ 5-dim      │
                          ▼         ▼         ▼            ▼            │
                    ┌──────────────────────────────────────────┐        │
   FUSION (every    │  attention over 4 cue tokens + CLS token │        │
   8/30 s stride)   │  → 10-class intent softmax (F01–F10)     │        │
                    └───────────────────┬──────────────────────┘        │
                                        ▼                               │
                    smoothing + hysteresis + F02 emergency bypass       │
                                        ▼                               │
                    POLICY: intent (+context) → action A01–A15 ─────────┘
```

## 4. Stages (read in order)

| # | Doc | What it covers | Key artifacts |
|---|---|---|---|
| 1 | [01_data_and_labels.md](01_data_and_labels.md) | Dataset design (V3 table), recording, curation, `labels.csv`, actor-disjoint splits | `data/`, `scripts/00`,`01` |
| 2 | [02_unimodal_models.md](02_unimodal_models.md) | The 4 perception models: architecture, I/O, accuracy, failure modes | `modalities/`, `MODEL_AUDIT.md` |
| 3 | [03_feature_extraction.md](03_feature_extraction.md) | Two-pass extraction, time-based windowing, the feature table | `scripts/02`,`03`, `features_v1.parquet` |
| 4 | [04_baselines.md](04_baselines.md) | Rule-based, unimodal, concat-MLP — the evidence floor for G1 | `fusion/baselines/`, `scripts/04` |
| 5 | [05_fusion_model.md](05_fusion_model.md) | Attention architecture, augmentation, the [MISSING]-token result | `fusion/model/`, `scripts/05`,`06`,`07` |
| 6 | [06_evaluation.md](06_evaluation.md) | Metrics, masking sweep, G3 spotlight, window sweep | `results/fusion_v1/` |
| 7 | [07_deployment.md](07_deployment.md) | ONNX, TensorRT, streaming pipeline, policy, latency | `jetson_deploy/`, `scripts/08` |

## 5. A note on honesty (important for the thesis)

Two facts constrain every result below, and we state them everywhere rather than
hide them:
1. **The V3 test scenarios are not recorded yet.** All numbers so far use an
   *actor-disjoint* split of the 22 recorded (V2-train) scenarios: the model is
   tested on people it never trained on, but on the same situations. The harder
   cross-scenario tests (T02 conflict, T04 context-flip) become fully measurable
   only when the test scenarios are recorded.
2. **Synthetic data fills gaps.** Unrecorded intents (notably F10) and
   missing-cue rows are covered by *cue recombination* — real cue vectors
   recombined into new intents per the table's rubric. This is a stated
   contribution ("data-efficient fusion"), not a hidden shortcut, and synthetic
   samples are **never** in the test set.
