# Stage 2 — The four unimodal perception models

**Goal of this stage:** four independent models, each turning raw RGB frames into
a **probability vector** over its own cue vocabulary. They are the "senses" of
the system. The fusion model (Stage 5) never sees pixels — it only sees these
four vectors, so their quality and their *failure patterns* set the ceiling for
everything after.

These four models were built by the team **before** the fusion work, each in its
own `modalities/<name>/` folder with its own training pipeline. Stage 2 of the
fusion project was to **audit** them (not retrain) — discover exactly what each
one eats and emits, and record it. The audit table is `docs/MODEL_AUDIT.md`;
this document is the readable explanation.

**A note on which numbers to trust.** A model's "own" reported accuracy and its
accuracy *as used in fusion* can differ, and several reports in the repo come
from different splits or older checkpoints. Wherever they disagree, this doc
gives the honest one and says why. The most consistent signal is the
**cue-agreement measured in Stage 3** — how often each deployed model's argmax
matches the scenario's intended cue, over the curated data. That is literally
what feeds fusion, so it appears in every section as *"as used in fusion."*

---

## 2.0 What the four have in common

- Each emits a **softmax probability vector**, never a hard label. An uncertain
  "0.55 wave / 0.45 raise_hand" carries information that `argmax` throws away.
- Each model's final layer emits **raw logits**; **softmax is applied outside**
  the model (in the extraction/inference code). So "model output" = logits,
  "cue vector" = softmax(logits).
- Each is **small enough for a Jetson** — except context, which is a large
  pretrained model that earns its place (§2.4).
- Two of them (gesture, motion) consume **MediaPipe keypoints**, not pixels — so
  at deployment a single MediaPipe Holistic pass serves both (Stages 3 and 7).
- Class orders are **fixed by each model's head** and are NOT the design table's
  order. Fusion stores each cue in its *native* order and remaps once
  (`DECISIONS.md`). This is exactly the kind of silent mismatch that caused
  cross-machine bugs, so it is pinned down.

### Deployed checkpoints (verified 2026-07-24 against `checkpoint_manifest.sha256`)

| Modality | Deployed artifact | Selected by | Not deployed (present on disk) |
|---|---|---|---|
| Emotion | `finetuned_MobileNetV2.pth` | `config.py:76` | EfficientNet-B0, MNASNet, MobileNetV3, all `*_LSTM`, `best_*`, `compare_*` |
| Gesture | `best_TCN.pth` + `model_config.json` | `config.py:74` | `best_TCN_pre_bhu`, `_pretune`, `_prev` |
| Motion | `best_model_finetuned.pt` | README / deploy pkg | `best_model` (NTU-only), `best_model_6class`, all `last_*` |
| Context | **CLIP ViT-B/32 zero-shot** (no checkpoint) | `SCENE_BACKEND="clip"`, `config.py:67` | `best_EfficientNet_B0.pth` — the *legacy CNN CLIP replaced* |

> The context folder contains a trained `best_EfficientNet_B0.pth` that is **not
> used**. It is the CNN baseline that zero-shot CLIP replaced (§2.4). Keep it for
> the comparison, never point inference at it.

---

## 2.1 Emotion — MobileNetV2 face CNN

**Role.** Read the facial expression → 7-way emotion. Emotion is the cue that
most often *flips* an intent (thumbs-down = help if sad, frustration if angry),
so it is the highest-value cue — and it turned out to be the most reliable.

### Architecture
`torchvision.mobilenet_v2` (ImageNet-pretrained) with the classifier head
replaced (`src/models.py:15`):

```python
model = tvm.mobilenet_v2(weights="DEFAULT")
model.classifier[1] = nn.Linear(model.last_channel, 7)   # 1280 -> 7
```

**2,232,839 params · 9.18 MB (fp32).** MobileNetV2 is a stack of *inverted
residual blocks*: 1×1 expand → 3×3 **depthwise** conv (each channel processed
separately — the cheap trick) → 1×1 project, with residual skips, and **ReLU6**
activations (capped at 6, friendly to INT8 hardware — a Jetson-minded choice).
Global average pool → 1280-dim feature → linear head.

The param count is 2.23M rather than the textbook ~3.5M because the 1280→1000
ImageNet head (~1.28M params) was replaced by 1280→7 (8,967 params). The feature
extractor is untouched ImageNet MobileNetV2; only the last linear layer is new.

**Why this backbone — and why it is *not* simply "won the Stage-1 table."** The
Stage-1 search (`src/models.py:47` registry: MobileNetV2, MobileNetV3-Large,
EfficientNet-B0, MNASNet1.0) ranks candidates by macro-F1 **on RAF-DB alone**,
within the Jetson size budget (`SIZE_BUDGET_MB = 20`). On that table
EfficientNet-B0 wins (69.76% macro-F1 vs MobileNetV2's 61.34%) — so the
RAF-DB benchmark, taken at face value, recommends the wrong model.

RAF-DB is curated, close-up portrait photography; this project's footage has
subjects 2–5 m from camera with 40–90 px faces, a distribution RAF-DB does not
represent. Backbone selection cannot stop at RAF-DB accuracy, so every
candidate was fine-tuned once more, with an identical recipe, on real face
crops from this project's own footage (`scripts/46_finetune_emotion_backbone.py`,
warm-started from each architecture's own RAF-DB checkpoint), then re-scored
on 856 held-out real-world test clips (`scripts/45_compare_emotion_backbones.py`,
clip-level mean-softmax, same protocol for every model):

| Model | RAF-DB-only acc / macro-F1 | Real-data fine-tuned acc / macro-F1 |
|---|---|---|
| **MobileNetV2 (deployed)** | 43.22% / 35.83% | **77.45% / 66.81%** |
| MobileNetV3-Large | 34.11% / 28.38% | 69.51% / 61.00% |
| EfficientNet-B0 | 37.85% / 31.82% | 66.12% / 56.80% |

Under the deployment-realistic criterion the RAF-DB ranking does not merely
weaken, it **inverts**: MobileNetV2 beats EfficientNet-B0 by +11.3 points
accuracy / +10.0 points macro-F1 after fine-tuning, despite being the
*smallest* candidate (2.23M params vs EfficientNet-B0's 4.02M). MobileNetV2 is
deployed because it wins on real footage after fine-tuning, not because it won
the RAF-DB search — the RAF-DB table alone would have picked the wrong model.
Full numbers: `results/realworld_eval_merged/emotion_backbone_comparison.json`;
narrative writeup: `modalities/emotion/reports/comparison/COMPARISON_REPORT.md`.

### Input pipeline
frame → **MediaPipe face detection with a robust 3-pass fallback**
(`detect_face_box`): plain → CLAHE contrast-boosted → tiled quadrants at full
resolution (~1.5× effective zoom). This exists because the footage has subjects
2–5 m away with **40–90 px faces**; a single pass misses ~30%, the tiled pass
recovers ~100% at ~14 ms/frame. Then: tight crop → 224×224 → ImageNet
normalization (mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, mandatory
for the pretrained backbone).

The model never sees a whole frame — only a face crop. **No face → no emotion
cue** (marked missing downstream, never faked).

### Output
7-dim softmax in **RAF-DB folder order** (`config.py:26`):
`[Surprise, Fear, Disgust, Happy, Sad, Anger, Neutral]` — *not* the table's order.

### Training
Dataset: **RAF-DB** (~15K labelled faces), heavily imbalanced (Happy 4,772 vs
Fear 281 — `config.py:39`). That imbalance drives every choice. Recipe in
`src/engine.py`:

- **Stage 1 (5 epochs, head only):** backbone frozen, **inverse-frequency
  class-weighted** cross-entropy, Adam @ 1e-3.
- **Stage 2 (25 epochs, full fine-tune):** LR 1e-4, **label smoothing 0.1**,
  **mixup α=0.2** (50% of batches), **cosine LR with 2-epoch warmup**, gradient
  clipping at 1.0, AMP.
- **Checkpoint selection metric = macro-F1**, not accuracy (`engine.py:151`) — so
  it cannot win by nailing Happy/Neutral and ignoring Fear.
- Heavy train augmentation (`src/transforms.py`): resize-240→random-crop-224,
  h-flip, ±15° rotation, translate/shear, colour jitter, 5% grayscale,
  **RandomCLAHE** (matching the inference-time contrast step so train and serve
  agree), RandomErasing. Eval transform is plain resize + normalize.
- **Then a second fine-tune on real face crops from `data/`** — the decisive step.

### Results
| Checkpoint | RAF-DB acc/macro-F1 | **Real held-out subjects** |
|---|---|---|
| `best_MobileNetV2` (RAF-DB only) | 84.3% / 76.7% | 58.8% / **38.9%** |
| **`finetuned_MobileNetV2` (deployed)** | 81.9% / 74.3% | **92.5% / 90.1%** |

The RAF-DB-only model is *better on RAF-DB* but **collapses on this project's
footage** — RAF-DB is close-up curated portraits, our faces are 40–90 px at
distance, so it predicts Neutral/Happy for nearly everything. Fine-tuning on real
crops more than doubles macro-F1. The 92.5%/90.1% is measured on subjects
(P03/P05/P07–P09) used in **neither** fine-tuning nor checkpoint selection.

**As used in fusion: 0.94** — the strongest cue, consistent with the above.
(Some repo reports show ~74% "val"; those are the P04 split the checkpoint was
early-stopped on — selection-inflated, and flagged stale in the modality README.)

### Known issues
- **Fear and Disgust are thin** in training data → lower per-class F1.
- Needs a visible face; turning away makes the cue missing (handled, not faked).
- **A referenced feature is not implemented:** `config.py:36` says
  `TRAIN_CLASS_COUNTS` is used at inference to re-inject the natural class prior
  "see `src/postprocess.py`" — **that file does not exist**, and deployed
  inference does a plain softmax/argmax. The comment is aspirational; the
  deployed path is self-consistent, but the prior correction was never wired in.
- **The LSTM variant was tried and rejected:** `finetuned_MobileNetV2_LSTM.pth`
  (CNN + per-clip temporal head) scores **87.5% / 82.9%** on honest test subjects
  — *worse* than the plain CNN — **and its training script is lost** (architecture
  reverse-engineered from the weights). Lesson: expression is near-instantaneous,
  so per-frame CNN + softmax-averaging beats a learned temporal head here. This is
  why emotion is the only one of the four with no temporal model.
- **Checkpoint footgun, documented in code:** `config.py:72` records that a
  2026-07-16 folder copy silently reset the default to the weak
  `best_MobileNetV2`, regressing every downstream script for days. Always print
  the resolved checkpoint path before believing a bad result.

**ONNX:** exported and verified (max |diff| ≈ 1.9e-6).

---

## 2.2 Gesture — TCN over keypoint sequences

**Role.** Recognize the hand/arm gesture → 8-way. Gestures are the most
explicitly communicative cue, but the most ambiguous alone — the same gesture
spans several intents, which is the whole G3 story.

### What it replaced
The original modality was **MediaPipe Hands → static-pose MLP + hand-written
rules** for wave/raise/beckon, deleted in commit `0074d2a`. Two fatal flaws:
dynamic gestures were unlearned heuristics (no metric possible), and a hands-only
view **physically cannot see arm/body gestures**. The v2 rewrite is a learned
keypoint-sequence classifier — *replacing rules with a temporal model* is itself
a methodology result.

### Architecture — TCN (Temporal Convolutional Network)
Chosen from three <1M-param candidates in `src/models.py`: **BiGRU** (recurrent),
**TinyTransformer** (attention), **TCN** (temporal convolutions). TCN won.

**683,272 params · 2.76 MB.** Forward pass (`src/models.py:72`):

```
input             [B, 32, 185]     32 frames × 185 features
 ↓ transpose      [B, 185, 32]     Conv1d wants channels-first
 ↓ proj Conv1d(185→128, k=1)
                  [B, 128, 32]
 ↓ 4 residual dilated blocks, dilations 1, 2, 4, 8
                  [B, 128, 32]
 ↓ global average pool over time   (z.mean(dim=2))
                  [B, 128]         ← penultimate "gesture embedding"
 ↓ head: Dropout(0.3) + Linear(128→8)
                  [B, 8]           ← FINAL layer: class logits
```

Each block (`_TCNBlock`, `src/models.py:42`) is two `Conv1d(kernel=5)` layers
with BatchNorm + ReLU + Dropout, wrapped in a **residual** connection.

**The key idea — dilation.** A dilated convolution skips gaps between the samples
it reads, so its reach grows without extra parameters. Stacking dilations
1→2→4→8 grows the receptive field *exponentially*: with kernel 5 and two convs
per block the total receptive field is **~121 frames**, larger than the whole
32-frame window. So **every output neuron sees the entire gesture**, whether it
is a fast wave oscillation or a slow raise-and-hold.

**Why TCN over BiGRU/Transformer:** no recurrence → fully parallel (fast on
Jetson), tiny, and dilated convs suit fixed-length windows. The **global average
pool** makes it robust to *where* in the window the gesture occurs.

### Input pipeline — the 185-dim feature (the clever part)
Per frame, MediaPipe **Holistic** → pose (33) + left/right hand (21 each) →
`build_features` (`src/features.py:100`):

| Block | Dims | Normalization |
|---|---|---|
| Pose 33 × (x, y, visibility) | 99 | **mid-shoulder = origin**, scaled by **shoulder width** |
| Left hand 21 × (x, y) | 42 | **wrist = origin**, scaled by wrist↔middle-MCP distance |
| Left-hand presence flag | 1 | 1.0 if detected else 0.0 (block zeroed when absent) |
| Right hand 21 × (x, y) | 42 | same |
| Right-hand presence flag | 1 | same |
| **Total** | **185** | |

Three deliberate choices:
1. **Position/scale invariance** — the same gesture looks identical near or far,
   left or right of frame. Hips are deliberately avoided (often off-frame close up).
2. **Hands encode shape only; position comes from the pose.** Hand landmarks are
   wrist-relative, so the hand block says *what shape* the hand is, while *where*
   it is (raised? moving?) comes from pose wrist landmarks 15/16. This
   factorization keeps the two signals from interfering.
3. **Presence flags make multi-dataset training work.** Training mixes close-up
   clips (Jester: pose barely visible) with full-body clips (NTU: hand detail
   poor), and deployment runs at 2–4 m where MediaPipe hand detection flickers.
   The flags tell the model *which block to trust this frame*, so it learns "when
   hands are absent, decide from the pose."

Window = **32 frames (~2 s)**; training clips are uniformly resampled to 32,
inference uses a rolling buffer.

### Output
8-dim softmax, native order:
`[idle, wave, point, thumbs_up, thumbs_down, beckoning, raise_hand, both_hands_up]`.
**`idle` is a real trained class** (the table's "none"), built from thousands of
Jester/NTU negative actions — NOT a confidence threshold. Fusion treats it as
distinct from a *missing* cue.

### How it emits output (asked during review — worth being precise)
It is a **sequence-in, one-label-out** model: 32 frames in → **one prediction for
that window**. "Frame by frame" only describes how frames are *fed*:

| Mode | Runs the model |
|---|---|
| Standalone live (`GestureEngine`) | on **every frame** — rolling ~2 s buffer resampled to 32 and scored, then EMA-smoothed + 0.6 confidence gate + 300 ms debounce |
| **In fusion** | once **per stride step** (every 8/30 s ≈ 3.75 Hz) |

Fusion receives the **8-dim softmax**, not the 128-dim penultimate embedding and
not a hard label — see §2.5 for why.

### Training data
- **20BN-Jester** (close-up): thumbs up/down, wave (shaking hand), beckoning
  (proxy: pulling hand in), plus its other 21 classes subsampled as **`idle` hard
  negatives**.
- **NTU RGB+D 120** (full-body): waving, pointing, thumbs up/down, both-hands-up,
  plus daily actions (drink, clap, wipe face) as more `idle` negatives.
  Split **subject-wise** (NTU filenames encode subject ID).
- **Custom clips for `beckoning` and `raise_hand`** — no good public source;
  `raise_hand` has *zero* NTU source and is entirely custom.
- A **live RealSense test set of all 8 classes, never trained on**.
- Dropout 0.3, label smoothing 0.1, plus temporal augmentation: **speed jitter**
  (`sample_window`, 0.8–1.2×) and **horizontal mirror** (flip x, swap L/R sides).
- Reported validation: **93.2% acc / 92.8% macro-F1** (`model_config.json`).

### Results (regenerated 2026-07-24 on the deployed checkpoint)
`modalities/gesture/reports/evaluation/TCN/REALWORLD_REPORT.md` —
**76.8% acc / 77.7% macro-F1** on the curated clips.

| Strong | Weak |
|---|---|
| raise_hand, beckoning, thumbs_up, thumbs_down, **both_hands_up** | **point** (S03 0%, S22 5% → idle; S29 78%), **wave** (S02 62%, S25 25% → raise_hand) |

**Why `point` fails** — a design consequence, not a bug: a static point is an
extended arm + extended index finger, but the hand block encodes *shape only* and
at 2–4 m MediaPipe hand landmarks are unreliable, while pose alone cannot
separate "arm extended pointing" from "arm resting." **Why `wave` is weak** — a
subtle wave is a small oscillation that resembles a held raised arm.

**As used in fusion: 0.75** — the least reliable cue, which is precisely why the
fusion model learns to lean on emotion/context instead.

### The checkpoint-provenance lesson
Four TCN checkpoints exist: `best_TCN.pth` (deployed), `best_TCN_pre_bhu.pth`,
`_pretune`, `_prev`. The **old** real-world report was generated from the
*pre-both-hands-up* checkpoint and therefore showed both `point` **and**
`both_hands_up` as untrained (0%). The deployed checkpoint recognizes
`both_hands_up` fine (0.83–1.00 per scenario). Same filename convention,
different weights, stale report → wrong conclusion. This is the cross-machine
failure mode in miniature, and why Stage 0 added the hash manifest + WORKLOG. The
old report is preserved as `REALWORLD_REPORT_pre_bhu.md`; the current one is
regenerated by `scripts/12_regen_gesture_realworld.py`. Each checkpoint also has
a matching `model_config_*.json` pinning architecture/labels/window.

**ONNX:** exported and verified (max |diff| ≈ 1.3e-6).

---

## 2.3 Motion — skeleton LSTM with attention pooling

**Role.** Classify body motion → 4-way. Motion disambiguates approach/retreat and
posture (standing frustration vs fleeing danger).

### Architecture
**`MotionLSTM(hidden=256, layers=3, dropout=0.35)` · 1,420,013 params · ~5.4 MB.**

> ⚠ `src/model.py`'s docstring says "~270k params, 2-layer, hidden 128" — that was
> the *original* design. The **deployed checkpoint's stored config** is
> hidden 256 / 3 layers (1.42M params). Trust the config baked into the
> checkpoint, not the docstring — another provenance trap.

```
input [B, 30, 84]
 ↓ 1. LayerNorm over the 84 features        positions & velocities differ in scale
 ↓ 2. 3-layer LSTM (hidden 256) → [B, 30, 256]
 ↓ 3. temporal attention pooling → [B, 256]
 ↓ 4. Linear(256→64) → ReLU → Dropout → Linear(64→4)
      [B, 4] logits
```

1. **LayerNorm** (`model.py:34`) stabilizes training given mixed position/velocity scales.
2. **LSTM** — recurrent: walks the 30 frames carrying a hidden state, gates decide
   what to remember/forget. The classic sequence model, and the deliberate
   opposite of gesture's convolutional approach.
3. **Temporal attention pooling** (`model.py:47`) — instead of taking the last
   hidden state, a `Linear(256→1)` scores all 30 frames, softmaxes them into
   weights, and takes a weighted sum. The model **learns which frames matter**
   (for "walking", the peak-velocity frame beats the first barely-moving frame).
   A mini attention mechanism inside a unimodal model — a nice parallel to fusion.
4. Classifier head 256→64→4.

### Input pipeline — 84-dim skeleton + velocity
1. MediaPipe Pose → **world landmarks** (3-D metric, in metres) — *not* the
   image-plane landmarks gesture uses. Motion needs real 3-D because walking and
   stepping-back are movements **in depth**, which a 2-D projection barely shows.
2. Map to a **14-joint NTU-style skeleton** (`MP_TO_NTU`) with X/Y sign correction
   to match NTU's convention.
3. **Normalize** (`normalize_skeleton`): **hip-centered**, **shoulder-width
   scaled** → 14 × 3 = **42** numbers.
4. **Append per-frame velocity** (difference from previous frame) → **+42 = 84**.
5. Window = **30 frames (~1 s)**.

Two points to remember:
- **Velocity is explicit here.** Gesture hands raw positions to the net and lets
  it learn dynamics; motion hard-codes them. Both are valid; motion's choice is
  physically motivated (walking *is* velocity).
- **Normalization deliberately removes global translation** (hip-centering),
  giving invariance to *where* the person stands — but this is also the root of
  its main failure (below).

### Output
4-dim softmax: `[sitting, standing, walking, stepping_back]`.
**There is no `run` class** — the handover text mentioned 5 classes including
running, but the deployed head is 4. Running scenarios can only map to
walking/stepping_back. (Recorded in `MODEL_AUDIT.md`.)

### Training
1. **Pretrain on NTU RGB+D** skeletons (4-class) — large, clean skeleton data.
2. **Fine-tune on the real intent clips**, warm-started from the NTU model (the
   checkpoint records `warm_started_from`). Fine-tune windows are resampled to a
   uniform **30 Hz grid** so window semantics match NTU despite mixed clip fps.
   Deployed config: LR 2e-4, weight-decay 0.006, dropout 0.35, 80 epochs /
   patience 15; kept checkpoint is epoch 20.
3. Scenario S19 was excluded from fine-tuning (its "move backward (run)" has no
   clean mapping onto the 4-class taxonomy).

### Results
| Checkpoint | Data | Accuracy |
|---|---|---|
| `best_model` | NTU only | 96.7% val (synthetic benchmark) |
| **`best_model_finetuned` (deployed)** | NTU + real | **76.8% acc / 73.2% macro-F1** (held-out test subjects, all 1,061 clips) |

The test split contains **zero `sitting` clips**, so 76.8% is a harder,
3-class-effective estimate rather than one inflated by the easy class.
**As used in fusion: 0.81.**

### The signature failure (well diagnosed)
**`stepping_back` fails in the kitchen for *every* subject** — S19 0%, S23 5%,
S26 0%, all predicted "standing" — while the same motion in the classroom (S06)
scores 100%. Because it fails for *train* subjects too, this is **not** a
subject-generalization gap; it is an environment/recording issue. Cause: kitchen
framing (camera distance, counter occlusion, smaller visible backward step)
combined with the **deliberate removal of global translation** means a backward
step with little limb swing loses almost all signal. Proposed fix (modality
README): expose **z-velocity** to fusion, restoring the depth-translation signal
the normalization discards. Meanwhile fusion compensates using the other cues —
a concrete demonstration of *why fusion helps*.

**Deployment:** `MotionInference` keeps its own 30-frame buffer; call `reset()`
when the tracked person changes. **ONNX:** verified (max |diff| ≈ 3.8e-6).

---

## 2.4 Context — CLIP zero-shot scene classifier

**Role.** Identify the environment. Context rarely identifies an intent alone (a
kitchen hosts many intents) but it *conditions* the others — raise_hand means
help in a classroom, greeting in a kitchen.

### The headline: this model was never trained
No training run, no checkpoint, no dataset, no loss. A pretrained foundation
model used as-is, whose "class definitions" are **English sentences in a config
file**. The team *did* train a CNN for this — and then replaced it (see Results).

### How CLIP works
CLIP (Contrastive Language–Image Pretraining) was trained on ~400M image–caption
pairs, learning to place **images and text in one shared embedding space**. That
gives classification **without a classifier**:

```
frame ─► CLIP image encoder (ViT-B/32) ─► image embedding (512-d, L2-normalized)
                                                  │
text prompts ─► CLIP text encoder ─► class embeddings (512-d, normalized)
                                                  │
                     cosine similarity × 100 → logits → softmax
```

Class scores are literally **how similar the image is to each class's text
description**. Change the text, change the classifier. **ViT-B/32** = Vision
Transformer, Base, 32×32 patches (224² image → 7×7 = 49 patch tokens, transformer
attends over them). Note that three of the four models now use attention
somewhere — context's ViT, motion's pooling, and fusion's core.

### Two design refinements in our implementation (`src/zero_shot.py`)
**(a) Prompt ensembles.** Each class gets **6 phrasings**, not one
(`config.py:73`), all encoded, normalized, and **averaged into one class
prototype**. CLIP is sensitive to phrasing; a single prompt is brittle, six are
stable. The prompts deliberately describe *objects in the scene* ("a stove and a
sink in a kitchen", "a whiteboard at the front of a classroom") — exactly what
distinguishes these environments.

**(b) An abstain probe.** Three extra prompts (`config.py:118`):
`"a close-up photo of a person's face"`, `"a selfie of a person"`, `"a portrait
…"`. If a frame matches these better than any scene, the result is **"uncertain"**
instead of a guess. This handles the real failure case — someone walks up to the
camera, their face fills the frame, and **there is no scene to classify**.
Downstream that becomes an honest *missing cue*, which fusion is built to handle.
The perception layer knowing when it doesn't know is a deliberate feature.

Plus **temporal smoothing**: a 15-frame rolling average of the probability
vectors, since the scene is essentially static.

### Input / output
Input: RGB frame → CLIP preprocessing (resize/center-crop 224, CLIP
normalization), sampled every N frames — **not** every frame (in the Jetson
pipeline ~1 Hz, held between updates).
Output: 5-dim softmax over `[classroom, kitchen, hospital, cloth_store, museum]`.

Only classroom/kitchen appear in current data; the other three are **free future
capacity** — adding "hospital" cost six lines of English, no data, no retraining.
The V3 table already names Hospital/Museum/Shop as future environments.

### Results — why the trained CNN was replaced
`reports/zero_shot/ZERO_SHOT_REPORT.md`, 4,436 frames:

| Model | classroom | **kitchen** | overall | latency |
|---|---|---|---|---|
| EfficientNet-B0 (trained) | 97.1% | **63.1%** | 82.2% | 25.3 ms |
| **CLIP ViT-B/32 (zero-shot)** | 100% | **98.8%** | **99.5%** | 23.7 ms |

The trained CNN had a **severe kitchen domain gap** — it had learned the specific
classrooms it saw and did not transfer to our kitchen footage. CLIP, trained on
400M diverse web images, has effectively seen every kitchen. And CLIP was
*slightly faster*, because text embeddings are computed **once at startup** —
per-frame cost is one image-encoder pass plus 5 dot products.

**As used in fusion: 0.97** — second-strongest cue; abstain rate 0.2%.

### Trade-offs to be able to defend
**Why is zero-shot acceptable in a research project?** Because context is the one
cue whose categories are *visually generic* — a kitchen is a kitchen everywhere —
so a foundation model's breadth beats a small model's specificity. Contrast with
emotion, where the general model (RAF-DB portraits) **collapsed** on far-field
faces and fine-tuning was essential. Two opposite conclusions, each backed by its
own benchmark: a genuinely interesting thesis point about *when* to fine-tune
versus use zero-shot.

**The cost:** CLIP is by far the largest model (~600 MB of weights vs 2–9 MB), so
it runs at ~1 Hz and is **not exported to ONNX/TensorRT** — it stays PyTorch on
the Jetson, loading offline from the bundled `hf_cache/`.

**Limitation:** it classifies the *room*, not the *situation*. With only two
environments in the data, this cue carries little intent information alone —
unimodal-context scores just **14.6%** on intent (Stage 4). Its value is entirely
as a *conditioner* of the other cues.

---

## 2.5 Putting the four together

### How each model handles time
| Model | Temporal mechanism | Explicit dynamics? |
|---|---|---|
| Emotion | none — per-frame, softmax-averaged over the window | no |
| Gesture | **dilated temporal convolutions** (TCN) | no — learns dynamics from raw positions |
| Motion | **LSTM** + attention pooling | **yes** — appends per-frame velocity |
| Context | none — per-frame, held between updates | no |

Three genuinely different approaches to temporal modelling across four cues —
worth highlighting in the thesis.

### Full comparison
| | Emotion | Gesture | Motion | Context |
|---|---|---|---|---|
| Architecture | MobileNetV2 CNN | TCN (dilated conv) | LSTM + attention | CLIP ViT-B/32 |
| Params | 2.23M | 683K | 1.42M | ~150M (pretrained) |
| Input | 224² face crop | 32 × 185 keypoints | 30 × 84 skeleton+velocity | 224² frame |
| Training | RAF-DB → real fine-tune | Jester + NTU + custom | NTU → real fine-tune | **none (zero-shot)** |
| Output | 7-dim | 8-dim | 4-dim | 5-dim |
| **Cue agreement** | **0.94** | **0.75** | **0.81** | **0.97** |
| ONNX | ✓ | ✓ | ✓ | ✗ (stays PyTorch) |

**24 dimensions of probability in total — the entire input to fusion.**

### What is passed to fusion, and why
Fusion receives, per stride step, the **four softmax vectors** — not embeddings,
not hard labels:
- **Probabilities, not hard labels:** "0.55 wave / 0.45 raise_hand" carries
  information `argmax` destroys.
- **Small probability vectors, not the penultimate embeddings** (e.g. gesture's
  128-dim): keeps the fusion input tiny (24 dims), keeps the four models
  **decoupled** (retrain one without changing fusion's input size), and keeps the
  representation interpretable.

### The reliability ranking predicts the fusion result
| Cue | Agreement | Trust | Main weakness |
|---|---|---|---|
| Context | 0.97 | highest | only 2 environments exist |
| Emotion | 0.94 | high | Fear/Disgust thin; needs a visible face |
| Motion | 0.81 | medium | kitchen `stepping_back` ≈ 0 |
| Gesture | 0.75 | lower | `point` ≈ 0, `wave` weak |

This ordering *predicts* what fusion does: lean on emotion and context, and use
them to **correct** unreliable gesture. That is exactly what the Stage-6 masking
sweep shows (masking emotion costs ~28 points; masking gesture costs less), and
it is the mechanism behind the G1 result.

### Where the evidence lives
- Per-model training/eval reports: `modalities/<name>/reports/`
- I/O contract, frameworks, checkpoints: `docs/MODEL_AUDIT.md`
- As-used-in-fusion agreement: computed in Stage 3, summarized in `docs/WORKLOG.md`

---

### Open points you might want to change
- **Gesture `point`** is the biggest perception gap and hurts several F03/F05
  table rows. If Stage-6 error analysis blames it, the handover permits a targeted
  gesture retrain (an HPC job) → re-extract only the gesture columns →
  `features_v2.parquet`.
- **Motion kitchen `stepping_back`:** before retraining, check the raw kitchen
  framing — it may be a recording issue that more data will not fix. Alternative:
  expose z-velocity as an extra signal.
- **Emotion prior re-injection** (`postprocess.py`) was designed but never
  implemented. Decide whether to implement or delete the stale reference.
