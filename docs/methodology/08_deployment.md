# Stage 8 — Deployment to the Jetson Orin Nano

**Goal of this stage:** take four models plus a fusion head that run on a laptop
over recorded clips, and make them run **live, in order, fast enough**, on an
embedded device — then turn the predicted intent into a robot action.

Artifacts: `jetson_deploy/` · guides: `JETSON_INFERENCE_GUIDE.md`,
`TENSORRT_GUIDE.md` · export: `scripts/08_export_onnx.py`.

---

## 8.1 What changes between offline and live

| | Offline (Stages 3–7) | Live (Jetson) |
|---|---|---|
| Input | a finite clip file | an endless camera stream |
| Timing | as slow as you like | must keep up with the camera |
| Clip boundaries | exist | **do not exist** |
| Missing cues | flags in a table | whatever the detectors actually return |
| Output | an accuracy number | an action, repeatedly, in real time |

The third row is the design problem: everything in Stages 3–7 is defined per
*clip*, and live there is no clip. The resolution is a **rolling buffer** — a
~4 s window that slides continuously, which is the streaming equivalent of a clip
(§8.4).

---

## 8.2 The export chain — and what does *not* export

The Jetson path is `PyTorch → ONNX → TensorRT`, but only for **three of the four**
perception models plus the fusion head:

| Component | Exported? | Why |
|---|---|---|
| Emotion MobileNetV2 | ✅ ONNX, max diff **1.9e-6** | plain CNN |
| Gesture TCN | ✅ ONNX, **1.3e-6** | plain conv net |
| Motion LSTM | ✅ ONNX, **3.8e-6** | recurrent, exports cleanly |
| Fusion attention | ✅ ONNX, **1.4e-6** | incl. the exclude-mode padding mask |
| **MediaPipe** | ❌ | a Google **TFLite** graph, not PyTorch — never enters the chain |
| **Context CLIP** | ❌ by choice | ~600 MB, runs at 1 Hz; conversion effort ≫ payoff. Stays PyTorch, loads offline from the bundled `hf_cache/` |

All four exports verified at opset 17 against native PyTorch (tolerance 1e-4,
actual ≈1e-6) — `jetson_deploy/onnx/export_report.json`. This was the schedule's
biggest single risk and it is retired.

**Three corrections to the naive "convert everything to TensorRT" plan**, worth
stating because they are the common misconceptions:

1. **TensorRT does not touch the bottleneck.** MediaPipe Holistic (~55 ms/frame)
   is the slowest stage and is not a PyTorch model. The nets TensorRT *can*
   accelerate already cost 2–46 ms. The biggest latency lever on Jetson is
   Holistic `model_complexity=0`, not TensorRT.
2. **Engines must be built on the Jetson.** A `.engine` (or ORT TensorRT cache) is
   compiled for a specific GPU architecture and TensorRT version. ONNX is
   portable; engines are not. Building on the laptop and copying **will not work**.
3. **FP16 changes the numbers.** ONNX-vs-PyTorch agrees to ~1e-6, but TensorRT
   FP16 logits differ by ~1e-2. That does not change the predicted class —
   `trt_check.py` therefore verifies **argmax agreement**, which is the property
   that matters, not raw closeness.

Two TensorRT routes are provided: **Route A** (ONNX Runtime TensorRT execution
provider — zero build step, `--backend tensorrt`, engines JIT-compiled and cached)
and **Route B** (`build_engines.sh` → native `.engine` via `trtexec`, for peak
performance and benchmark numbers). Route A is recommended.

---

## 8.3 Environment pins that will break things if ignored

- **`protobuf==3.20.3`** — installing `onnx` pulls protobuf 7.x, which **silently
  breaks MediaPipe** (`FieldDescriptor has no attribute 'label'`). Pinned in
  `requirements.txt` alongside `onnx==1.14.1`.
- **PyTorch/torchvision must be NVIDIA's JetPack-matched wheels** — generic pip
  wheels do not work on Jetson ARM64+CUDA.
- **`HF_HOME` / `HF_HUB_OFFLINE`** must point at the bundled cache so CLIP runs
  with no network.

---

## 8.4 The streaming pipeline

`jetson_deploy/fusion/pipeline.py`.

**Per frame:** one MediaPipe Holistic pass feeds **both** gesture (image
landmarks) and motion (world landmarks) — the single biggest efficiency decision
in the system, since MediaPipe is the bottleneck. Results go into rolling
time-based buffers.

**Per stride (8/30 s ≈ 3.75 Hz):** each cue is computed over its lookback span,
fused, smoothed, and turned into an action.

> **The windowing in `pipeline.py` mirrors `fusion/extraction/windows.py`
> exactly** — same spans, same uniform resampling, same aggregation. Any drift
> between them is silent train/serve skew that costs accuracy with no error
> message. Both files carry this warning.

Because the windows are **time-based**, the pipeline degrades gracefully: MediaPipe
caps the live rate at ~18 fps, and a lower frame rate simply coarsens the
resampling rather than breaking window semantics. (The training clips were 15 fps,
so this is well inside the trained regime.)

Three behaviours on top of raw prediction:

- **Temporal smoothing** — majority vote over the last 3 fused outputs.
- **Hysteresis** — a new intent must win twice consecutively before it becomes
  active. Prevents intent flicker in a demo.
- **Emergency bypass** — **F02 skips both** and fires on the first step where
  P(F02) ≥ 0.30. Latency on an emergency is a safety property, so it must not wait
  for smoothing to agree.

Missing cues are first-class: if the face is hidden or the pose is lost, that cue
is marked unobserved and **excluded from attention** rather than zero-filled —
the same mechanism the model was trained with.

---

## 8.5 The policy layer — intent to action

`fusion/actions/policy.py` maps intent (+ context) to actions A01–A15 per the V3
legend, with two safety mechanisms:

- **Confidence gate (τ = 0.5):** if the top intent scores below τ, fall back to
  **F05 → A06** (hold position, do not interrupt) — the safe default.
- **F02 bypass (τ_emergency = 0.30):** deliberately *lower*, because a missed
  emergency is far worse than a false alarm. Context routes the hazard response:
  classroom → A14 (notify supervisor), kitchen → A02/A03 (fire/medical).

Measured on `data/old` test subjects: **22/22 emergency clips fired**, window-level
F02 recall 0.968, false-emergency rate 1.3%.

⚠️ **This must be re-measured on `data/final`** — the gap analysis shows row #23
(fear + motion blur → F02) scoring 0.188, so the safety claim does not transfer
unexamined. See `07_evaluation.md` §7.10.

---

## 8.6 Latency — and a reasoning error worth documenting

Reference profile, **RTX 3060 laptop, ONNX backend** (Jetson will be slower —
re-run `fusion/profile_latency.py` there):

| Component | mean ms |
|---|---|
| MediaPipe Holistic (per frame) | 55 |
| emotion (face detect + CNN) | 46 |
| context CLIP (only every 1 s) | 19 |
| gesture / motion / fusion head | 2.4 / 3.0 / <1 |
| **capture → intent** | **125 (p95 267)** — within the 300 ms budget |

**The error, and the correction.** My first profiler computed per-stride cost as
`8 frames × Holistic + step work` and reported **550 ms — OVER BUDGET**. That is
wrong: frames are processed **as they arrive** (pipelined), so the capture→intent
latency for the frame that triggers a step is *one* Holistic pass plus the step
work — the earlier frames' cost is already paid. Corrected: **125 ms**.

Throughput is a **separate** constraint from latency: Holistic caps input at
~18 fps, so a 267 ms stride carries ~5 fresh frames. That is fine precisely
because windowing is time-based (§8.4).

**Tuning ladder if the Jetson exceeds budget**, in order:
1. `--backend onnx` → `tensorrt`
2. Holistic `model_complexity=0` (the biggest single win)
3. fewer emotion frames per step (4 → 2)
4. larger stride (8/30 → 12/30)
5. context every 2 s instead of 1 s

**Never shrink the lookback spans** — those must match training.

---

## 8.7 Verification status (be precise about this)

| Verified on the laptop | Not yet verified |
|---|---|
| ONNX export, all 4 nets, ~1e-6 vs native | anything on actual Jetson hardware |
| Pipeline end-to-end, torch **and** ONNX backends, on real clips | TensorRT backend (needs the device) |
| S01 clip → steady `F04 → A05`; S19 clip → `F02 → A02 EMERGENCY` every step | live RealSense capture |
| Latency profile on RTX 3060 | Jetson latency profile |

The TensorRT code is **written but untested** — it can only run on the Orin Nano.
Treat the first Jetson run as the real verification.

---

## 8.8 What Stage 8 produced

```
jetson_deploy/
├── JETSON_INFERENCE_GUIDE.md   setup, run, profile, tune, robot integration
├── TENSORRT_GUIDE.md           Route A/B, prerequisites, FP16 accuracy, troubleshooting
├── fusion/
│   ├── pipeline.py             streaming pipeline (torch | onnx | tensorrt)
│   ├── profile_latency.py      per-component profiler + budget check
│   ├── model.py, policy.py     self-contained COPIES of the repo originals
│   ├── fusion_attn.pt          deployed checkpoint
│   └── build_engines.sh, trt_check.py
├── onnx/                       4 exported nets + export_report.json
├── modalities/                 the 4 perception models (code + checkpoints)
└── hf_cache/                   CLIP weights for offline use
```

`model.py` and `policy.py` are **copies**. If the originals change, re-copy them
**and** re-run `scripts/08_export_onnx.py`, or the checkpoint and the graph drift
apart silently.

**In one sentence:** four models with incompatible time granularities, one of them
un-exportable and one of them the bottleneck, made to run as a single streaming
system that emits a smoothed intent and a robot action ~3.75 times per second
inside a 300 ms latency budget — with the emergency path deliberately bypassing
every smoothing mechanism.

---

### Open points you might want to change
- **Run everything on the actual Jetson** and record the numbers in `WORKLOG.md`.
- **Confidence-thresholded masking** — the runtime currently masks only on
  *detector failure*, not on low confidence. This is the train/deploy mismatch
  identified in `04_missing_cues.md` §4.3; it needs a τ tuned on val.
- **Re-measure the F02 safety numbers** on `data/final`.
- **The deployed fusion checkpoint is trained on `data/old`** and should be
  replaced once the `data/final` retraining settles.
