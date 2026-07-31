# Stage 3 — Feature extraction & the feature table

**Goal of this stage:** run the four perception models over every clip **once**,
and freeze their outputs into a single table. From this point on, **all fusion
development reads the table and never touches a video again.**

That one sentence is the most important efficiency decision in the project. The
alternative — decoding videos and running MediaPipe every time you try a new
fusion idea — would make each experiment take hours instead of seconds. After
Stage 3, training a fusion model takes ~20 seconds.

Scripts: `scripts/02_extract_perframe.py`, `scripts/03_build_features.py`.
Code: `fusion/extraction/`. Output: `data/features/features_v1.parquet`.

---

## 3.1 The core problem: four models that disagree about time

Each perception model consumes a different amount of video:

| Model | Native unit |
|---|---|
| Emotion | **1 frame** (a face crop) |
| Gesture | **32 frames** (~2 s window) |
| Motion | **30 frames** (~1 s window) |
| Context | **1 frame**, but the scene barely changes |

The fusion model needs all four cues **describing the same moment**, as one
aligned row. So Stage 3's job is to define a **common timeline** and make every
model report onto it — without violating the statistics each model was trained
on.

---

## 3.2 The two-pass design

A naive implementation runs all four models in one loop per clip. We split it in
two, and the reason is practical:

```
PASS 1  (expensive, run once)          PASS 2  (cheap, re-runnable)
video ──► MediaPipe + CNN/CLIP    ──►  per-frame cache ──► windowing ──► parquet
          ~4.8 s per clip                 (.npz)            ~seconds
```

**Pass 1 — `scripts/02_extract_perframe.py`** decodes each clip once and stores
*raw per-frame outputs* to `data/features/perframe/<clip_id>.npz`:

```
emotion_probs  [T, 7]      NaN rows where no face was detected
gesture_feats  [T, 185]    the 185-dim keypoint features (not yet classified)
joints25       [T, 25, 3]  NTU-layout world joints for motion
pose_valid     [T]         bool — MediaPipe found a pose
face_valid     [T]         bool — a face was detected
context_probs  [Tc, 5]     raw CLIP scene softmax, sampled ~3 Hz
context_frames [Tc]        which frame each context sample came from
fps, n_frames              scalars
```

**Pass 2 — `scripts/03_build_features.py`** reads those caches, applies the
windowing (§3.4), runs the gesture TCN and motion LSTM on the windows, and writes
the parquet.

**Why split?** Three concrete payoffs:
1. **Windowing is a design choice we want to revisit.** The window-size sweep
   (Stage 6) re-ran Pass 2 at 0.5× and 2× lookbacks in ~80 seconds each. With a
   single-pass design that would have been three full re-extractions (~7 hours).
2. **Resumability.** Pass 1 skips clips whose `.npz` already exists, so an
   interrupted 2-hour run continues where it stopped.
3. **Retraining one model is cheap.** If gesture is retrained, only its columns
   need regenerating (the 185-dim features are already cached) → `features_v2`.

This mirrors a decision the gesture modality had already made internally (store
raw landmarks, engineer features at load time) — extraction is the expensive
step, so cache before the expensive step, not after.

### One MediaPipe pass serves two models
Gesture needs **image-plane** pose+hand landmarks; motion needs **3-D world**
landmarks. MediaPipe **Holistic** produces both in one call, so Pass 1 runs it
once per frame and derives both feature sets. This is the single biggest
efficiency gain in the whole system and it carries straight into the Jetson
pipeline (Stage 7), where MediaPipe is the latency bottleneck.

---

## 3.3 The critical detail: time-based, not frame-based windows

Stage 1 found that **104 kitchen clips are 24/30 fps phone video**, not the
15 fps the annotations claimed. That single fact dictates the windowing design.

**If windows were counted in frames**, a "32-frame window" would mean 2.13 s on a
15 fps clip but only 1.07 s on a 30 fps clip — the models would see *different
amounts of real time* depending on which camera recorded the clip, and the fusion
model would be trained on inconsistent statistics.

**So every span is defined in seconds**, and each clip's *probed* fps converts
seconds → frame indices:

```python
times = np.arange(T) / fps                       # real timestamp per frame
mask  = (times > t_end - GES_SPAN) & (times <= t_end) & pose_valid
```

The selected frames are then **uniformly resampled** to whatever frame count the
model expects (32 for gesture, 30 for motion) with `uniform_indices`. So a 15 fps
clip contributes ~32 real frames to a 2.13 s window, while a 30 fps clip
contributes ~64 frames resampled down to 32 — **both represent the same 2.13
seconds**. That is the invariant that matters.

*Bonus:* this same property later made the Jetson deployment robust. MediaPipe
caps the live frame rate at ~18 fps, but because windows are time-based, a lower
frame rate just coarsens the resampling instead of breaking window semantics.

---

## 3.4 The window grid — and why each span is what it is

```
stride S = 8/30 s ≈ 0.267 s   → a new fused intent ~3.75 times per second
```

The stride is inherited from the deployment target (S=8 frames at 30 fps). Each
cue then looks *back* from the step time by its own span:

| Cue | Lookback | Why exactly this |
|---|---|---|
| **Gesture** | **2.133 s** → resampled to 32 frames | matches `GestureEngine`'s live buffer (`ENGINE_BUFFER_FRAMES=64` @30 fps → WINDOW=32). The deployed engine's own statistics. |
| **Motion** | **2.0 s** → resampled to 30 frames | gives dt ≈ 1/15 s per resampled frame, matching how the model was fine-tuned on 15 fps clips |
| **Emotion** | **0.267 s** (mean softmax over face frames) | expression is near-instantaneous; averaging a long span would blur a changing expression |
| **Context** | **1.0 s** (mean of ~3 Hz CLIP samples) | the scene is static; averaging suppresses per-frame noise |

The guiding principle: **each cue's span is chosen to match the statistics that
model was trained/tuned on**, not a single arbitrary window for all four. Forcing
one window on all four would have degraded at least two of them.

**Emotion has a fallback:** if no face was detected in the last 0.267 s, the span
widens to the gesture span (2.133 s) before the cue is declared missing. A person
briefly turning their head shouldn't wipe out the strongest cue.

**Minimum-frame guards:** gesture needs ≥8 valid pose frames in its span, motion
≥15. Below that the cue is reported missing rather than computed from too little
data — better an honest "unknown" than a confident guess from 3 frames.

---

## 3.5 Two kinds of "missing" (an important distinction)

The V3 table marks some scenarios as having a **designed** missing cue (e.g. S08:
hands occupied carrying books → gesture not capturable). Separately, a cue can
fail **at runtime** (the face wasn't detected in this particular window). These
are different things and the table keeps them apart:

| Column | Meaning |
|---|---|
| `missing_designed` | from `labels.csv` — the *scenario* was designed with this cue absent |
| `emo_obs`, `ges_obs`, `mot_obs`, `ctx_obs` | booleans — was this cue actually computed for **this window** |
| `emo_cov`, `ges_cov`, … | coverage ratio — what fraction of the span had usable frames |

When a cue is unobserved, its probability columns are **NaN**, not zeros. Zeros
would be a lie (a valid-looking probability vector); NaN plus `*_obs = False`
says "we don't know," and Stage 5's fusion model has an explicit mechanism for
exactly that.

Measured observation rates over the whole dataset:

| Cue | Observed |
|---|---|
| emotion | 100% |
| gesture | 99.84% |
| motion | 99.75% |
| context | 100% |

Very high — the curated clips are clean. The genuine missing-modality robustness
therefore has to be *trained* by deliberate masking (Stage 5), because the real
data alone hardly ever exercises it.

---

## 3.6 The output: `features_v1.parquet`

**15,892 rows × 43 columns**, one row per (clip, window):

```
identity   clip_id, scenario_id, person_id, split_subject, v3_row, status
label      intent, context_gt, missing_designed
timing     window_idx, t_end
cues       emo_Surprise … emo_Neutral        (7)
           ges_idle … ges_both_hands_up      (8)
           mot_sitting … mot_stepping_back   (4)
           ctx_classroom … ctx_museum        (5)
flags      emo_obs, ges_obs, mot_obs, ctx_obs
coverage   emo_cov, ges_cov, mot_cov, ctx_cov
```

**Column names carry the class name in each model's native order** — so a
mismatch is visible rather than silent. (`emo_Surprise` is first because that's
RAF-DB's order, not the table's.)

### The free data multiplier
1,061 clips → **15,892 windows = 15× more training samples**, at zero recording
cost. Median 14 windows per clip (min 1 for a very short clip, max 42).

This matters enormously given only 22 recorded scenarios. But it comes with a
caveat we respect throughout: **windows from one clip are highly correlated** —
they overlap in time and show the same person doing the same thing. So:
- the **actor-disjoint split** (Stage 1) matters even more, and
- **clip-level accuracy** (majority vote over a clip's windows) is the headline
  metric; window-level accuracy is secondary and slightly optimistic.

### Window-level intent distribution
| Intent | Windows | | Intent | Windows |
|---|---|---|---|---|
| F04 | 2,413 | | F07 | 1,452 |
| F01 | 2,242 | | F08 | 1,167 |
| F05 | 2,241 | | F10 | 881 |
| F03 | 2,109 | | F06 | 703 |
| F02 | 2,002 | | F09 | 682 |

Note **F10 = 881 windows all come from S28**, which is flagged
`status=recombination_pool` and therefore **excluded from supervised training**
(Stage 1). So in practice F10 has *zero* training rows until Stage 5's synthetic
recombination fills it — which is exactly why macro-F1 is capped at 0.9 in the
Stage-4 baselines.

### The manifest — reproducibility across machines
`data/features/manifest.json` records, alongside the parquet:
- **sha256 of every checkpoint used** (emotion, gesture, motion) + the context
  backend identity,
- the **class orders** for all four cues,
- the **window parameters** (all five spans),
- git commit, hostname, timestamp, row/clip counts.

This is the direct answer to the two-PC problem: given a parquet, you can prove
*which weights produced it*. If the HPC regenerates features and a hash differs,
you know before you train, not after you get confusing results.

---

## 3.7 Practical problems solved along the way

**Colliding module names.** The four modalities each use generic top-level module
names (`config`, `src`, `model`, `inference`) resolved via their own `sys.path`
hacks — importing two of them in one process collides. `fusion/extraction/modloader.py`
imports each modality file under a unique alias with the right paths prepended,
then purges the generic names from `sys.modules`. (Context's `zero_shot.py`
permanently inserts its own root into `sys.path`, so it is loaded last.)

**Reusing the models' own code.** Extraction deliberately imports the modalities'
*existing* functions — emotion's `detect_face_box` and `get_transform`, gesture's
`build_features`, motion's `normalize_skeleton` and `JOINT_SUBSET` — rather than
reimplementing preprocessing. Reimplementation is how train/serve skew is born.

**Environment.** `pyarrow` was missing from the venv, and the venv has no `pip`
(it is uv-managed): `uv pip install pyarrow --python .venv/Scripts/python.exe`.

**Run cost.** Pass 1: 1,061 clips, ~4.8 s/clip, ~2.2 hours, **0 failures**.
Pass 2: ~2 minutes.

---

## 3.8 The by-product: measuring how good the perception models really are

Because the table pairs every cue vector with the scenario's *intended* cue, we
can measure each model's agreement with the intended label directly — the
"as used in fusion" numbers quoted throughout Stage 2:

| Cue | Agreement | |
|---|---|---|
| Context | 0.97 | |
| Emotion | 0.94 | |
| Motion | 0.81 | kitchen `stepping_back` ≈ 0 |
| Gesture | 0.75 | `point` ≈ 0, `wave` weak |

This is a more honest picture than each modality's own report, because it uses
the **deployed** checkpoint, on the **curated** data, in the **exact window
configuration** fusion will use. It is also what exposed the stale gesture
report (Stage 2) — the extraction said `both_hands_up` worked while the report
said it was untrained.

---

## 3.9 What Stage 3 produced

```
data/features/
├── perframe/<clip_id>.npz      1,061 per-frame caches (Pass 1)
├── features_v1.parquet         15,892 windows × 43 cols  ← THE table
└── manifest.json               checkpoint hashes, class orders, window params
```

**In one sentence:** we ran four models with incompatible time granularities over
1,061 clips exactly once, aligned their outputs onto a shared time-based grid,
and froze the result into a 15,892-row table with full provenance — turning every
later fusion experiment from an hours-long job into a seconds-long one.

---

### Open points you might want to change
- **`features_v2` trigger:** if a unimodal model is retrained, only its columns
  need regenerating from the cached `.npz` — the Pass-1 caches stay valid unless
  the *feature definition* changes (gesture's 185-dim spec, motion's 84-dim spec).
- **The window sweep suggests 1.07 s spans score better than 2.13 s** (Stage 6).
  The deployed model keeps 2.13 s to match the unimodal engines' validated
  buffers; revisit after Jetson profiling.
- **Direction was skipped for v1** (your decision, Stage 1). If added later, it
  would become a 5th cue column here, computed from the cached `joints25`
  trajectories — no video re-decode needed.
