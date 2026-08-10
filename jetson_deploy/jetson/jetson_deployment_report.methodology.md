# HRI Jetson Pipeline — Methodology (A to Z)

Companion document to `jetson_deployment_report.md`. That report tells you
*what the numbers were*; this document explains *what actually runs, in
what order, to produce them* — the full pipeline from camera frame to
displayed INTENT, end to end.

## 0. One-paragraph summary

Every loop iteration in `live_demo.py`, a camera frame is read, MediaPipe
extracts a face box and body/hand landmarks from it, four independent ONNX
models each turn their slice of that frame into a class-probability
distribution (emotion / gesture / motion / scene-context), those
distributions are pushed into a rolling 3-second window per modality, and
periodically that window is collapsed into one fixed 33-column feature row
and handed to a fusion model (LightGBM or hand-written rules) that outputs
one of ten intent labels (F01–F10). Everything runs on-device, CPU + GPU,
no cloud calls.

## Pipeline diagram

```
 RealSense / webcam
        |
        v
  camera.py: CameraSource.read()          -- one BGR frame
        |
        v
  landmarks.py: LandmarkExtractor.process()
        |-- MediaPipe Holistic  -> pose_landmarks, pose_world_landmarks,
        |                          left/right_hand_landmarks
        |                          (skipped on 2 of 3 frames if
        |                           --holistic-every 3; last result reused)
        `-- MediaPipe FaceDetection -> face_box   (always runs, it's cheap)
        |
        +------------------+------------------+------------------+
        v                  v                  v                  v
  cue_emotion.py     cue_gesture.py     cue_motion.py       cue_context.py
  face_box crop      pose+hands ->      pose_world ->       full frame ->
  -> MobileNetV2      185-dim/frame ->   NTU25 remap ->      EfficientNet-B0
  -> 7 emotions        32-frame window   14-joint norm ->    -> 2 scene classes
                        -> TCN -> EMA     30-frame window     (only every Nth
                        -> 8 gestures     + velocity -> LSTM   frame, --context-every)
                                          -> 4 motion classes
        |                  |                  |                  |
        v                  v                  v                  v
        +------------------+------------------+------------------+
                            |
                            v
        feature_aggregator.py: FusionFeatureBuilder
        each modality keeps its own rolling 3s window (--window-sec) of
        (timestamp, prob_vector) pairs; build_row() collapses each window
        into: mean class probabilities, one confidence stat (max or mean,
        fixed per modality), valid_fraction, missing flag
        -> one 33-column feature row, on demand
                            |
                            v
        fusion.py: FusionModel.predict(row)     -- gated by --fusion-hz,
        gbt (LightGBM) or rule_based             NOT every frame
                            |
                            v
                    INTENT  (F01..F10)
                 shown in overlay / printed
```

## 1. Camera capture — `runtime/camera.py`

`CameraSource` tries an Intel RealSense first (`pyrealsense2`, color stream
via `rs.align` so color is spatially aligned to depth even though depth
itself isn't used downstream); if no RealSense is found or it fails to
open, it silently falls back to any V4L2 webcam (`cv2.VideoCapture`,
`--webcam N` to pick the index). Either path returns a plain BGR `numpy`
frame from `.read()` — everything downstream is camera-source-agnostic.

## 2. Landmark extraction — `runtime/landmarks.py`

One shared `LandmarkExtractor` instance serves both the gesture and motion
cues so pose estimation is only paid for once per frame, not twice:

- **MediaPipe Holistic** (`model_complexity` 0=lite/1=full) — the single
  most expensive call in the whole pipeline (~110-200ms on this Jetson's
  CPU; legacy `mediapipe.solutions` API, CPU-only, no GPU delegate
  available). Produces `pose_landmarks` (image-space, for gesture),
  `pose_world_landmarks` (metric space, for motion), and both hands'
  landmarks.
  - `--holistic-every N` amortizes this cost: Holistic only actually runs
    every Nth call; on skipped calls, the previous result is reused
    (`FrameLandmarks.pose_stale=True` marks this). See §8 for the measured
    latency win.
- **MediaPipe FaceDetection** — a second, much cheaper model, always run
  every frame regardless of `--holistic-every`, producing the pixel-space
  `face_box` the emotion cue crops from. `--robust-face` swaps in a
  slower CLAHE + 4-tile fallback pass for far-field faces the single-pass
  detector misses; default (`--fast-face`) is single-pass only.

## 3. The four cue models

All four load through the same three-tier session loader (§4) and are
otherwise independent — none of them know or care whether they're running
on TensorRT or CPU onnxruntime.

### 3a. Emotion — `runtime/cue_emotion.py`
Input: the `face_box` crop of the current frame (skipped entirely,
`valid=False`, if no face was detected this frame). Resize 224×224,
scale to [0,1], ImageNet-normalize, NCHW. Model: ONNX **MobileNetV2**.
Output: softmax over 7 classes (`Surprise, Fear, Disgust, Happy, Sad,
Anger, Neutral`). No temporal state — a pure per-frame classifier.

### 3b. Gesture — `runtime/cue_gesture.py`
Input: this frame's pose + both hands' landmarks (image-space).
Normalized per-frame into a 185-dim feature vector (pose centered on the
shoulder midpoint and scaled by shoulder width; each hand centered on its
wrist and scaled by wrist→middle-MCP distance; a presence flag per hand).
Feature vectors accumulate in a 64-frame (~2.1s) deque; once at least 32
frames exist, the last 64 are uniformly resampled down to a fixed 32-frame
window and fed to an ONNX **TCN**. Output softmax (8 raw classes: idle +
7 named gestures) is EMA-smoothed (`α=0.25`) across calls, and gated by a
confidence threshold (0.60) — below threshold, the cue reports "Unknown"
(idle) instead of a low-confidence guess. `valid=False` when no pose is
visible at all (buffer is reset); `stable=False` until the 32-frame window
first fills (a genuine cold-start warm-up, not an error state).

### 3c. Motion — `runtime/cue_motion.py`
Input: this frame's pose **world** landmarks (metric-space, camera-relative
3D). Remapped from MediaPipe's 33-point skeleton to the 25-joint NTU
layout the model was trained on, subset to 14 joints, hip-centered and
shoulder-width-normalized. Each frame contributes both a position vector
and a velocity vector (this frame's normalized position minus last
frame's) into a 30-frame rolling window, fed to an ONNX **LSTM**. Output:
softmax over 4 classes (`sitting, standing, walking, stepping_back`). Like
gesture, `stable=False` until the 30-frame window fills; unlike gesture,
when no person is visible it still advances the buffer on the last known
(or zero) position rather than resetting, matching the original engine's
behavior.

### 3d. Context — `runtime/cue_context.py`
Input: the **entire** frame (no crop, no landmarks needed — every frame is
"valid" for this cue by definition). Same 224×224/[0,1]/ImageNet-normalize
preprocessing as the other CNN. Model: ONNX **EfficientNet-B0**. Output:
softmax over scene classes (`classroom, kitchen`, from
`onnx_models/context_labels.json`). This is the second-heaviest call after
Holistic, so `live_demo.py` only runs it every Nth processed frame
(`--context-every`, default 3) — on skipped frames the fusion feature
window simply doesn't get a new context sample that tick (its own rolling
window still smooths over the gaps).

## 4. Inference backend — `runtime/ort_session.py` + `runtime/trt_session.py`

Every cue's `.onnx` file is loaded through one function, `load_session()`,
which tries three tiers in order and is invisible to the cue code (they
only ever call `.get_inputs()[0].name` and `.run(None, {name: array})`):

1. **TensorRT** — if `trt_engines/<model_name>.engine` exists, load it via
   `trt_session.TRTSession` (raw TensorRT + `cuda-python` for GPU memory
   management). Fastest path, GPU.
2. **onnxruntime GPU** — if no engine file, but onnxruntime has a
   `CUDAExecutionProvider`/`TensorrtExecutionProvider` built in.
3. **onnxruntime CPU** — the always-available fallback.

In this deployment, `trt_engines/*.engine` exist for all four models (FP16,
built via `export/build_trt_engines.sh` → `trtexec --fp16`), so tier 1 is
what's actually running unless those files are deleted. TensorRT's FP16
conversion was numerically verified against the FP32 ONNX baseline before
being trusted (argmax agreement + logit-diff check on real camera frames —
see the earlier benchmark report's Quantization section).

## 5. Feature aggregation — `runtime/feature_aggregator.py`

This is the layer that turns "four independent per-frame classifiers" into
"one fixed-shape row a fusion model can consume." Each modality gets its
own `ModalityWindow`: a deque of `(timestamp, prob_vector_or_None)` pairs,
pruned to the last `--window-sec` (default 3.0s) on every push. Frames
where a cue was `valid=False` (no face / no person) push `None`, which
still counts toward `valid_fraction` and `missing` but not toward the mean.

`build_row()` collapses each modality's window into 8-9 columns:
- mean probability per class (e.g. `emotion_Happy`, `gesture_wave`, …)
- one confidence statistic — **max** confidence for emotion/motion, **mean**
  confidence for gesture/context (fixed by the fusion model's training
  schema, not a runtime choice)
- `{name}_valid_fraction` — what fraction of the window actually had data
- `missing_{name}` — 1.0 if the window is entirely empty

Total: 33 columns, matching
`portable_fusion_models/portable_fusion/gbt/feature_names.json` exactly.
**Caveat carried over from the source file's own docstring**: the original
training-time feature-extraction code (that built the offline
`clip_features.parquet` the fusion models were trained on) isn't part of
this bundle — this rolling-window aggregation is a schema-faithful
reconstruction, not an extraction of that original code. If live fusion
accuracy looks off relative to each cue repo's own offline eval numbers,
this windowing (and `--window-sec`) is the first place to check.

## 6. Fusion model — `runtime/fusion.py`

Thin wrapper around `portable_fusion_models/portable_fusion/{gbt,rule_based}`,
imported unchanged (this is the one runtime module allowed to reach outside
`jetson_deploy/`). `--backend gbt` (default) loads the trained LightGBM
model; `--backend rule_based` uses hand-written rules instead (no
`lightgbm` dependency needed). Either way, `predict(row)` wraps the
33-column dict in a one-row DataFrame (columns reordered to
`feature_names`) and returns one of the ten intent labels, F01–F10.

## 7. The main loop — `live_demo.py`

Per iteration, in order: read a camera frame → `landmarks.process()` →
emotion → gesture → motion → context (if scheduled this tick) →
`feature_builder.push()` (always) → **if** `now - last_fusion_time >=
1/fusion_hz`: `feature_builder.build_row()` + `fusion_model.predict()`,
i.e. a new INTENT (this is the step gated by `--fusion-hz`, default twice
a second — see the report's §2 for why "how fast can it compute one
intent" and "how often do you see it update" are two different numbers).
Every stage's wall-clock time is recorded (`--metrics-file`) regardless of
whether you're watching the GUI window or running `--no-show` headless.

## 8. What actually controls speed vs. accuracy

| Flag | What it trades | Measured effect (this device) |
|---|---|---|
| `--holistic-complexity 0` | Lite vs. full MediaPipe pose model; some pose/gesture/motion accuracy | ~25-30% faster than default `1` |
| `--context-every N` | Scene-context freshness vs. its ~85ms(CPU)/~15ms(TRT) cost | amortizes the 2nd-heaviest stage |
| `--holistic-every N` | Pose/hand landmark freshness (stale up to N-1 frames) vs. the ~150-200ms Holistic cost | median frame time 158ms→54ms at N=3 |
| `--fusion-hz` | How often INTENT itself refreshes, independent of compute speed | default gates updates to ~500-540ms apart regardless of how fast the rest is |
| TensorRT engines (`trt_engines/`) | none measured — FP16 verified lossless on real frames | 3.96×-5.78× on the two CNNs (emotion, context); *slower* (0.64-0.68×) on the two tiny sequence models (gesture, motion) — GPU dispatch overhead exceeds their compute at that size |

## 9. Benchmarking / reporting tooling — `benchmark_report.py`

Runs `live_demo.py` for a bounded `--run-seconds` window with
`--metrics-file`/`--metrics-duration`, **concurrently** samples
`tegrastats` for the same window (fixed to sample *during* the run, not
after it exited), and writes:
- `jetson_deployment_report.md` — the human-readable report
- `jetson_deployment_report.raw.csv` — one row per processed frame,
  including `intent_latency_ms` (whole-flow compute time, only on the rows
  that actually produced an intent) and `since_last_intent_ms` (wall-clock
  gap between successive INTENT updates, i.e. the `--fusion-hz` effect)
- `jetson_deployment_report.intent_summary.csv` — per-intent rollup of the
  above
- `jetson_deployment_report.tegrastats.raw.txt` — raw power/thermal/GPU
  samples, one line per ~500ms

Any `live_demo.py` flag from §8 can be passed straight through to
`benchmark_report.py` so the report reflects whatever configuration you're
actually planning to deploy with.

## 10. How to run it

```bash
cd jetson_deploy
source .venv-runtime/bin/activate

# Plain live demo, GUI window
python live_demo.py --holistic-complexity 0 --context-every 5 --holistic-every 3

# Headless (SSH, no X server)
python live_demo.py --holistic-complexity 0 --context-every 5 --holistic-every 3 --no-show

# Full benchmark report + CSVs, GUI window visible while it measures
python benchmark_report.py --run-seconds 30 --backend gbt \
    --output reports/jetson_deployment_report.md \
    --holistic-complexity 0 --context-every 5 --holistic-every 3
```
