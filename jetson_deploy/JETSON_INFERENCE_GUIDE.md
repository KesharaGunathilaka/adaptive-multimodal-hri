# Jetson Orin Nano — running the full HRI fusion pipeline

End-to-end inference: RealSense/webcam → 4 perception models → attention fusion
→ intent (F01–F10) → robot action (A01–A15).

Written 2026-07-20 on WIN-3060 after verifying the pipeline on real clips in
both backends. Latency numbers below are from the laptop (RTX 3060) — **re-run
the profiler on the Jetson and record the numbers in `docs/WORKLOG.md`.**

---

## 1. What runs where

| Stage | Model | Runs as | Notes |
|---|---|---|---|
| Keypoints | MediaPipe Holistic | CPU (mediapipe) | ONE pass feeds gesture **and** motion. The bottleneck (~55 ms/frame). |
| Emotion | MobileNetV2 | PyTorch or ONNX | Needs a face crop from MediaPipe FaceDetection first. |
| Gesture | TCN | PyTorch or ONNX | 32-frame window of 185-dim features. |
| Motion | LSTM | PyTorch or ONNX | 30-frame window of 84-dim skeleton features. |
| Context | CLIP ViT-B-32 zero-shot | **PyTorch only** | Deliberately not exported; loads offline from `hf_cache/`. |
| Fusion | Attention (exclude-mode) | PyTorch or ONNX | 24-dim cues + 4 observed flags → 10 intents. |
| Policy | rule lookup | Python | τ gate + F02 emergency bypass. |

## 2. One-time setup

```bash
cd jetson_deploy

# 1. PyTorch/torchvision MUST be NVIDIA's JetPack-matched wheels — the generic
#    pip wheels do not work on Jetson ARM64+CUDA. Install those FIRST.

# 2. Then the rest. CRITICAL PIN: protobuf must stay 3.20.3 — anything newer
#    breaks mediapipe with "FieldDescriptor object has no attribute 'label'".
pip install -r requirements.txt
pip install open_clip_torch transformers        # context modality
pip install onnxruntime-gpu                     # only if using --backend onnx

# 3. Point HuggingFace at the bundled cache so CLIP runs with no network.
export HF_HOME=$PWD/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Verify the environment before trusting any result:

```bash
python -c "import mediapipe, numpy as np; \
mediapipe.solutions.holistic.Holistic(model_complexity=1).process(np.zeros((480,640,3),'uint8')); \
print('mediapipe OK')"
```
If that line fails, your protobuf is wrong — `pip install protobuf==3.20.3`.

## 3. Running it

```bash
cd jetson_deploy

# RealSense (default), PyTorch backend, on-screen overlay
python fusion/pipeline.py --source realsense

# Plain USB webcam, ONNX backend (usually faster), no GUI (headless/SSH)
python fusion/pipeline.py --source 0 --backend onnx --no-display

# Replay a recorded clip — best first test, output is reproducible
python fusion/pipeline.py --source ../data/raw/clips/classroom/S01_F04/S01_F04_c001.mp4 \
                          --no-display

# Log every decision for later analysis
python fusion/pipeline.py --source realsense --json run.jsonl
```

Useful flags: `--tau 0.6` (raise the confidence gate), `--tau-emergency 0.25`
(make F02 fire more eagerly), `--max-seconds 30`, `q` to quit the display.

### Expected output

```
[  3.20s] F04 -> A05 p=0.94 88ms
```
= at t=3.2 s the fused intent is **F04 (help request)**, the robot action is
**A05 (offer task guidance, supportive tone)**, fusion confidence 0.94, and the
step took 88 ms. An `EMERGENCY` suffix means the F02 bypass fired.

Sanity checks that should reproduce (verified on WIN-3060):
- `S01_F04_c001.mp4` (raise hand, neutral, sitting) → steady **F04 → A05**
- any `S19_F02` clip (fear, both hands up, step back, kitchen) → **F02 → A02, EMERGENCY**

## 4. Latency profiling (do this on the Jetson)

```bash
python fusion/profile_latency.py --source 0 --backend onnx --steps 30
```

Reference numbers from **RTX 3060 laptop, ONNX backend** (Jetson will be slower):

| component | mean ms |
|---|---|
| Holistic (per frame) | 55 |
| emotion (face detect + CNN) | 46 |
| context (CLIP, only every 1 s) | 19 |
| gesture / motion / fusion head | 2.4 / 3.0 / <1 |
| **capture → intent latency** | **125 ms (p95 267)** — within the 300 ms budget |

**How to read this.** Frames are processed as they arrive, so the latency for
one intent update is *one* Holistic pass + the step work — not 8× Holistic.
Throughput is separate: Holistic caps input at ~18 fps, so a 267 ms stride
carries ~5 fresh frames. That is fine because **windowing is time-based**
(spans in seconds, uniformly resampled) and the training clips were 15 fps.

If the Jetson comes in over budget, in order of preference:
1. `--backend onnx` (if not already).
2. Holistic `model_complexity=0` in `pipeline.py` (biggest single win).
3. Lower `EMOTION_FRAMES_PER_STEP` (4 → 2) in `pipeline.py`.
4. Raise the stride (`STRIDE_SEC` 8/30 → 12/30). **Never shrink the lookback
   spans** — those must match training.
5. `CONTEXT_EVERY_SEC` 1.0 → 2.0 (scene barely changes).

If the frame rate falls below ~10 fps, the 2 s gesture/motion windows start
missing their minimum frame counts (8 and 15) and those cues will report as
missing — the fusion model handles it, but accuracy drops. Fix the frame rate
rather than lowering the minimums.

## 5. How the pipeline behaves

- **Windowing mirrors training exactly** (`fusion/extraction/windows.py`):
  gesture 2.133 s → 32 frames, motion 2.0 s → 30 frames, emotion last 0.267 s,
  context every 1 s and held. Do not change one side without the other.
- **Missing cues are first-class.** If the face is hidden or MediaPipe loses the
  pose, that cue is marked unobserved and the attention layer *excludes* it
  rather than feeding zeros. The overlay shows `missing: emotion,...`.
- **Temporal smoothing:** majority vote over the last 3 fused outputs, plus
  hysteresis — a new intent must win twice in a row before it becomes active.
  Prevents flicker in demos.
- **Emergency bypass:** F02 skips both smoothing and hysteresis and fires on the
  first step where P(F02) ≥ 0.30. Measured on held-out test subjects: fires on
  22/22 emergency clips, 1.3% false-emergency rate.
- **Confidence gate:** if the top intent scores below τ=0.5, the policy falls
  back to F05 → A06 (hold position, do not interrupt) — the safe default.

## 6. Wiring into a robot

`pipeline.py`'s `HRIPipeline.step()` returns a dict per update:

```python
{"t": 3.2, "intent": "F04", "action": "A05",
 "action_text": "Offer task guidance / answer, supportive tone",
 "confidence": 0.94, "emergency": False, "fallback": False,
 "observed": {"emotion": True, "gesture": True, "motion": True, "context": True},
 "cues": {...}, "step_ms": 88.0}
```

Import it instead of running `main()`:

```python
from fusion.pipeline import HRIPipeline, open_source, STRIDE_SEC
pipe = HRIPipeline(backend="onnx")
# feed frames with pipe.push_frame(frame, t); call pipe.step(t) every STRIDE_SEC
```

Act on `action` (and check `emergency` first — that path is latency-critical).
Action codes are defined in `fusion/policy.py::ACTIONS`.

## 7. Files

```
jetson_deploy/
├── fusion/
│   ├── pipeline.py           streaming pipeline (this guide)
│   ├── profile_latency.py    per-component profiler
│   ├── model.py              attention architecture (COPY of fusion/model/model.py)
│   ├── policy.py             intent->action (COPY of fusion/actions/policy.py)
│   ├── fusion_attn.pt        deployed checkpoint (test clip acc 0.939)
│   └── fusion_config.json    config + metrics of that checkpoint
├── onnx/                     4 exported nets + export_report.json (all ~1e-6 vs native)
├── modalities/               the 4 perception models (code + checkpoints)
└── hf_cache/                 CLIP + SmolVLM2 weights for offline use
```

`model.py` and `policy.py` are **copies** of the main-repo originals. If you
change the originals, re-copy them here and re-run `scripts/08_export_onnx.py`,
otherwise the checkpoint and the graph drift apart.
