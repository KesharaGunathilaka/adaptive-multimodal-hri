# Stage 2 — The four unimodal perception models

**Goal of this stage:** four independent models, each turning raw RGB frames into
a **probability vector** over its own cue vocabulary. They are the "senses" of
the system. The fusion model (Stage 5) never sees pixels — it only sees these
four vectors, so their quality and their *failure patterns* set the ceiling for
everything after.

These four models were built by the team **before** the fusion work, each in its
own `modalities/<name>/` folder with its own training pipeline. Stage 2 of the
fusion project was to **audit** them (not retrain) — discover exactly what each
one eats and emits, and record it. The audit is in `docs/MODEL_AUDIT.md`; this
doc is the readable explanation.

A note you must internalize first: **a model's "own" accuracy and its accuracy
*as used in fusion* can differ**, and several published numbers come from
different splits. Wherever they disagree, I give all of them and say which to
trust. The single most consistent, honest signal is the **cue-agreement measured
in Stage 3** (how often each model's argmax matches the scenario's intended cue,
on the deployed checkpoint over the curated data) — that is what actually feeds
fusion, so it appears in every section as "as used in fusion."

---

## 2.0 What all four have in common

- Each emits a **softmax probability vector**, never a hard label. Fusion needs
  the full distribution (an uncertain "60% wave / 40% raise_hand" carries
  information a hard "wave" throws away).
- Each is **small** (Jetson target): MobileNetV2, a 683K-param TCN, a 3-layer
  LSTM, and a CLIP backbone that already runs for other reasons.
- Two of them (gesture, motion) consume **MediaPipe keypoints**, not pixels — so
  in deployment one MediaPipe pass serves both (Stage 3/7).
- Class orders are **fixed by each model's head** and are NOT the same order as
  the design table. Fusion stores each cue in its *native* order and remaps once
  (Decision, `DECISIONS.md`). This is the kind of silent mismatch that caused
  cross-machine bugs, so it's pinned down.

| Model | Framework | Input | Output dim | Deployed checkpoint |
|---|---|---|---|---|
| Emotion | PyTorch (MobileNetV2) | 224×224 face crop, per frame | 7 | `finetuned_MobileNetV2.pth` |
| Gesture | PyTorch (TCN) + MediaPipe | 32-frame × 185-dim keypoints | 8 | `best_TCN.pth` |
| Motion | PyTorch (LSTM) + MediaPipe | 30-frame × 84-dim skeleton | 4 | `best_model_finetuned.pt` |
| Context | open_clip (CLIP ViT-B/32) | RGB frame (sampled) | 5 | none — zero-shot from HF cache |

---

## 2.1 Emotion — MobileNetV2 face CNN

### Role
Read the person's facial expression → 7-way emotion. Emotion is the cue that
most often *flips* an intent (thumbs-down means help if sad, frustration if
angry), so it is the highest-value cue — and, as it turned out, the most
reliable one.

### Architecture & why
- **MobileNetV2** backbone (ImageNet-pretrained), classifier head replaced with a
  7-way linear layer. Chosen for being tiny and Jetson-friendly.
- **Per-frame**, not temporal. A temporal LSTM variant was also trained
  (`MobileNetV2_LSTM`) but scored *worse* on honest held-out subjects (87.5% vs
  92.5%) and its training script was lost — so it is explicitly **not deployed**.
  Lesson recorded in the modality README: expression is near-instantaneous, so a
  per-frame CNN + softmax-averaging over the window beats a learned temporal head
  here.

### Input pipeline
Frame → **MediaPipe face detection** (a robust multi-pass detector: plain →
CLAHE contrast boost → tiled quadrants) → tight face crop → 224×224 → ImageNet
normalization → CNN. The multi-pass detector exists because the project's
footage has faces at 2–5 m (40–90 px) where a single pass misses ~30%.

### Output
7-dim softmax in **RAF-DB order**: `Surprise, Fear, Disgust, Happy, Sad, Anger,
Neutral`. (Note this order — it is not the table's order.)

### Training data & procedure
1. **Pretrain on RAF-DB** (a public facial-emotion dataset of close-up portraits).
2. **Fine-tune on this project's real crops** (`data/`), because RAF-DB-only
   collapses on far-field faces. Recipe: weighted cross-entropy (classes are
   imbalanced), label smoothing 0.1, mixup, cosine LR with warmup, AMP.

### Results (and reconciling the numbers)
| Checkpoint | RAF-DB | Real **test-subject** acc / macro-F1 |
|---|---|---|
| `best_MobileNetV2` (RAF-DB only) | 84.3% / 76.7% | 58.8% / 38.9% — collapses far-field |
| **`finetuned_MobileNetV2` (deployed)** | 81.9% / 74.3% | **92.5% / 90.1%** |

The **92.5% / 90.1%** is the number to trust: it is measured on held-out
subjects (P03/P05/P07-09) that were in neither fine-tuning nor checkpoint
selection. Some other reports in the repo show ~74% "val" — those are an
*older/leakier* evaluation on subject P04 (the very split used for early
stopping), which the modality README explicitly flags as
selection-inflated/stale. **As used in fusion (Stage 3 cue-agreement): 0.94
macro** — consistent with 92.5%, and the strongest of the four cues.

### Known failures
- **Fear & Disgust are thin** in training data → lower per-class F1 there.
- Depends on a face being visible; if the person turns away, the cue is *missing*
  (handled downstream, not faked).

### Footgun worth knowing (relates to your cross-PC pain)
`config.py` and the two inference scripts each hardcode a **separate** default
checkpoint name. A folder copy once silently reset them all to the weak
`best_MobileNetV2.pth`, producing a fake "the model got worse" for days. **Always
print the resolved checkpoint path** before trusting a bad result. This is one
reason the fusion pipeline hashes checkpoints (`checkpoint_manifest.sha256`).

---

## 2.2 Gesture — Temporal Convolutional Network over keypoints

### Role
Recognize the hand/arm gesture → 8-way. Gestures are the most *explicitly
communicative* cue (wave, beckon, thumbs), but also the most ambiguous alone
(the same gesture spans several intents — that's the whole G3 story).

### Architecture & why
- **TCN** (temporal convolutional network), 683K params, 8-way head.
- Input is **MediaPipe keypoint sequences, not pixels**. The previous gesture
  model (hands-only MLP + hand-written rules for wave/beckon) was scrapped:
  dynamic gestures were un-measurable heuristics, and a hands-only view can't see
  arm/body gestures. The rewrite ("v2") uses **pose + both hands** landmarks so
  static and dynamic gestures are learned uniformly, no rules.

### Input pipeline
Per frame, MediaPipe **Holistic** → pose (33) + left/right hand (21 each)
landmarks → a **185-dim feature vector** (pose normalized to mid-shoulder origin
& shoulder-width scale; hands wrist-relative & size-normalized; presence flags).
A **32-frame window** (~2 s) of these is the model input. Storing *raw* landmarks
means features can be re-engineered without re-running MediaPipe (the expensive
step) — the same idea the fusion extraction reuses.

### Output
8-dim softmax: `idle, wave, point, thumbs_up, thumbs_down, beckoning, raise_hand,
both_hands_up`. **`idle` is a real trained class** (the table's "none"), NOT a
threshold fallback — distinct from a *missing* cue.

### Training data & procedure
- Public datasets (**Jester, NTU RGB+D 120**, optionally HaGRIDv2) for most
  classes; **~100 custom clips/class for `beckoning` and `raise_hand`** where
  public coverage is missing.
- Live RealSense test clips held out entirely.
- Deployed engine adds EMA smoothing + a 0.6 confidence gate + 300 ms debounce.
- Reported validation: **93.2% acc / 92.8% macro-F1** (`model_config.json`).

### Results — read this carefully (checkpoint provenance)
The gesture real-world numbers are the messiest, for a reason directly relevant
to your two-PC concern:
- An older report (`reports/evaluation/TCN/REALWORLD_REPORT.md`) shows `point`
  **and** `both_hands_up` as *untrained* — both always predicted `idle` (0%
  accuracy on those scenarios). That report was generated from a checkpoint
  **before** `both_hands_up` was added (there is literally a
  `best_TCN_pre_bhu.pth` = "pre-both-hands-up" checkpoint on disk).
- The **currently deployed** `best_TCN.pth` was retrained to include
  `both_hands_up`. Confirmed by Stage-3 cue-agreement: both-hands-up scenarios
  (S05, S09, S19, S24, S26) now score ~0.9, not 0.
- **`point` is still effectively untrained/weak** (S03 0.0, S22 0.05) and `wave`
  is weak (S25 0.24). Those remain real gaps.

**As used in fusion (Stage 3 cue-agreement): 0.75 macro** — strong on
`raise_hand`, `beckoning`, `thumbs_up/down`, `both_hands_up`; weak on `point`
and `wave`. This is the honest current picture; the deploy README's headline
84.1% comes from a more favorable split.

> **This is exactly the "model conflict across machines" problem** the project
> keeps hitting: the same filename `best_TCN.pth` meant different weights at
> different times, and a stale evaluation report drew the wrong conclusion. The
> fix going forward is the checkpoint hash manifest + WORKLOG (Stage 0).

### Known failures
- `point`: static pointing is barely distinguishable from idle in keypoints →
  often → idle. Affects table rows that rely on `point` (some F03/F05).
- `wave`: a subtle wave is under-detected, often → raise_hand.
- These are why fusion can't lean on gesture alone — and why emotion/context
  carry more weight in the learned model.

---

## 2.3 Motion — skeleton LSTM

### Role
Classify body motion → 4-way (`sitting, standing, walking, stepping_back`).
Motion disambiguates approach/retreat and posture (e.g. standing frustration vs
fleeing danger).

### Architecture & why
- **3-layer LSTM**, 84-dim per-frame features, **30-frame window** (~1 s @ 30 fps).
- LSTM (temporal) because motion *is* a trajectory — a single frame can't tell
  walking from standing.

### Input pipeline
MediaPipe Pose **world landmarks** → mapped to a **14-joint NTU-style skeleton**
→ hip-centered, shoulder-width normalized, X/Y axis-corrected → per-frame this is
42 numbers; stacked with **per-frame velocities** (42) → **84-dim**. Window of 30.
Crucially, normalization **removes global translation by design** (hip-centered),
which matters for a failure below.

### Output
4-dim softmax: `sitting, standing, walking, stepping_back`. **Note: no `run`
class** — the handover text mentioned 5 classes incl. running, but the actual
deployed head is 4. The design table's running rows can at best map to
walking/stepping_back. (Recorded in `MODEL_AUDIT.md`.)

### Training data & procedure
- **Pretrain on NTU RGB+D** skeletons (4-class), then **fine-tune on the real
  intent clips** (`videos/struct`) to close the domain gap. Fine-tune windows are
  resampled to a uniform 30 Hz grid so window semantics match NTU.
- Deployed: `best_model_finetuned.pt`.

### Results
| Checkpoint | Data | Accuracy |
|---|---|---|
| `best_model` | NTU only | 96.7% val (synthetic benchmark) |
| **`best_model_finetuned` (deployed)** | NTU + real | **76.8% acc / 73.2% macro-F1** on held-out test subjects (all 1,061 clips) |

The test split happens to contain **zero `sitting` clips**, so 76.8% is a harder
3-class-effective estimate, not inflated by the easy class. **As used in fusion
(Stage 3 cue-agreement): 0.81 macro.** This is the weakest of the four cues.

### Known failure (important, and well-diagnosed)
**`stepping_back` fails specifically in the kitchen, for every subject** (train
*and* test): S19 0%, S23 5%, S26 0% — predicted "standing" instead. The same
gesture in the classroom (S06) scores 100%. So it is **not** a
subject-generalization gap — it's an environment/recording issue: the kitchen
framing (camera distance, counter occlusion, smaller visible limb swing) plus the
deliberate removal of global translation means a backward step loses most of its
signal. Documented fix idea: expose z-velocity to fusion. For now, fusion learns
to compensate using the other cues — a concrete example of *why* fusion helps.

---

## 2.4 Context — CLIP zero-shot scene classifier

### Role
Identify the environment → scene probabilities. Context rarely identifies an
intent alone (a kitchen hosts many intents) but it *conditions* the others
(raise_hand means help in a classroom, greeting in a kitchen).

### Architecture & why — this one is NOT trained
- **CLIP ViT-B/32, zero-shot.** There is no trained head and no local checkpoint.
  Each frame's image embedding is matched against **text-prompt ensembles** per
  scene ("a photo of a classroom", …), plus a face-closeup "abstain" probe.
  Adding a new environment is a config edit (a prompt list), not a retraining run.
- Chosen after benchmarking against a fine-tuned EfficientNet-B0 CNN:

| Model | classroom | kitchen | overall |
|---|---|---|---|
| EfficientNet-B0 (trained) | 97.1% | 63.1% | 82.2% |
| **CLIP ViT-B/32 (zero-shot)** | 100% | **98.8%** | **99.5%** |

The CNN had a severe **kitchen domain gap** (63%); CLIP's broad pretraining has
no such gap. So we traded a small trained model for a big pretrained one — the
one place in the project where "zero-shot foundation model" beat "train your own."

### Input / output
RGB frame (sampled every N frames, not every frame) → 5-dim softmax:
`classroom, kitchen, hospital, cloth_store, museum`. Only classroom/kitchen occur
in the current data; the other three exist for future environments.

### Results
99.5% overall on captured frames; **as used in fusion (Stage 3 cue-agreement):
0.97 macro** — second-strongest cue and effectively a solved problem for the two
environments we have.

### Known limitations
- Costlier per call than the tiny nets, so in deployment it runs ~1 Hz and holds
  its value between updates (the scene barely changes). Not exported to ONNX — it
  stays in PyTorch on the Jetson.
- The 3 unused scenes could dilute confidence if a very ambiguous frame appears;
  not an issue in practice for classroom/kitchen.

---

## 2.5 Putting the four together — the reliability ranking

The Stage-3 cue-agreement (deployed checkpoints, curated data) gives the honest
ordering that the fusion model implicitly learns to respect:

| Cue | As-used-in-fusion agreement | Trust | Main weakness |
|---|---|---|---|
| Context (CLIP) | 0.97 | highest | only 2 environments exist |
| Emotion (MobileNetV2) | 0.94 | high | Fear/Disgust thin; needs visible face |
| Motion (LSTM) | 0.81 | medium | kitchen `stepping_back` ≈ 0 |
| Gesture (TCN) | 0.75 | lower | `point` ≈ 0, `wave` weak |

**Why this ranking matters for the thesis:** it predicts what fusion will do.
Emotion and context are dependable, so the model leans on them; gesture is
unreliable alone, so the model uses emotion/context to *correct* it. That is
exactly what the Stage-6 masking sweep shows — masking emotion costs ~28 points,
masking gesture costs less — and it is the mechanism behind the G1 result.

### Where these numbers live
- Per-model training/eval: `modalities/<name>/reports/`.
- Audit (I/O contract, frameworks, checkpoints): `docs/MODEL_AUDIT.md`.
- As-used-in-fusion agreement: computed in Stage 3, summarized in `docs/WORKLOG.md`.

---

### Open points you might want to change
- **Gesture `point`** is the biggest perception gap and it hurts several F03/F05
  table rows. If fusion error analysis (Stage 6) blames it, the handover allows a
  targeted gesture retrain — that would be an HPC job, then re-extract only the
  gesture column into `features_v2.parquet`.
- **Motion kitchen `stepping_back`**: before retraining, the modality README
  suggests checking the raw kitchen framing — it may be a recording issue that
  more data won't fix. Alternatively, expose z-velocity as an extra signal.
- **Stale reports**: I'd recommend we regenerate the gesture REALWORLD report on
  the *current* `best_TCN.pth` so no future reader trusts the pre-both-hands-up
  numbers. Quick to do; say the word.
