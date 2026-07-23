# TensorRT on the Jetson Orin Nano — complete guide

For the HRI fusion pipeline. Read `JETSON_INFERENCE_GUIDE.md` first (setup +
running); this file is only about the optional TensorRT acceleration step.

---

## 0. Is `PyTorch → ONNX → TensorRT` the right path? Yes — with 3 caveats

Your instinct is correct: for the neural networks, `PyTorch → ONNX → TensorRT`
is the standard Jetson path, and the ONNX half is already done (`onnx/`, all 4
nets verified vs native at ~1e-6). But before you invest time, know this:

**Caveat 1 — TensorRT does NOT touch your bottleneck.**
The slowest stage is **MediaPipe Holistic (~55 ms/frame)** — pose + hand
keypoints. MediaPipe is a Google TFLite graph, **not** a PyTorch/ONNX model, so
it never enters the ONNX→TensorRT path. It stays on CPU. The four nets TensorRT
*can* speed up are already cheap (emotion 46 ms, gesture 2, motion 3, fusion
<1). So TensorRT trims the small costs; it will not transform end-to-end
latency. The largest single latency win on Jetson is Holistic
`model_complexity=0` (edit in `fusion/pipeline.py`), not TensorRT.

**Caveat 2 — TensorRT engines MUST be built ON the Jetson.**
A `.engine` (or ORT TensorRT cache) is compiled for one specific GPU
architecture + TensorRT/CUDA version. You **cannot** build it on the RTX 3060
laptop and copy it — it will refuse to load or misbehave. ONNX is portable;
engines are not. Conversion to ONNX = done anywhere (already done). Conversion
to TensorRT = on the Orin Nano only.

**Caveat 3 — Context/CLIP stays PyTorch.**
It was deliberately not exported (runs once per second, high conversion effort,
low payoff). Leave it. TensorRT here covers emotion, gesture, motion, fusion.

So the realistic goal of this step is: shave the ~50 ms of net inference per
step down to ~10–15 ms and get FP16 for free. Worth doing for headroom and for
a clean thesis latency table — but the pipeline already meets the 300 ms budget
without it (125 ms mean / 267 ms p95 on a 3060).

---

## 1. Two routes — pick one

### Route A (recommended): ONNX Runtime TensorRT Execution Provider

No separate build step. ONNX Runtime compiles a TensorRT engine from your
existing `.onnx` files on first run and caches it. **The pipeline already
supports this** — one flag:

```bash
cd jetson_deploy
python fusion/pipeline.py --source realsense --backend tensorrt
```

- First run is SLOW (engine build, a few minutes) — this is normal. Engines are
  cached in `onnx/trt_cache/`; later runs load in seconds.
- FP16 is enabled automatically (`trt_fp16_enable`).
- Falls back to CUDA then CPU per-op for anything TensorRT can't handle.
- Requires `onnxruntime-gpu` **built with TensorRT** (JetPack's version — see §2).

This reuses everything you've already built and needs zero new inference code.
For a final-year project it is the pragmatic choice.

### Route B: native `trtexec` engines (max performance / benchmarking)

Build standalone `.engine` files with `trtexec`, then run them with the
TensorRT Python runtime. Slightly faster than Route A and gives you portable
engine files + `trtexec`'s own benchmark numbers.

```bash
cd jetson_deploy
bash fusion/build_engines.sh      # writes onnx/*.engine (FP16), ON THE JETSON
python fusion/trt_check.py        # verify each engine vs ONNX (argmax agreement)
```

`build_engines.sh` calls `trtexec` per model with fixed batch=1 shapes (the
pipeline always uses batch 1) and prints each engine's throughput. `trt_check.py`
loads every `.engine` with the TensorRT + pycuda runtime and checks that its
argmax matches ONNX (FP16 logits differ by ~1e-2, but the predicted class must
not change).

> The streaming `pipeline.py` uses Route A (the ORT TensorRT EP), not the raw
> `.engine` files — wiring a hand-rolled TensorRT runtime into the frame loop
> is extra risk for little gain. Use Route B for the benchmark table and to
> confirm the engines are valid; use Route A to actually run.

---

## 2. Prerequisites on the Jetson (JetPack)

TensorRT itself comes WITH JetPack (`/usr/src/tensorrt`, `trtexec` in
`/usr/src/tensorrt/bin`). You need the Python bindings to match:

```bash
# TensorRT + CUDA are part of JetPack — verify they're present:
/usr/src/tensorrt/bin/trtexec --version
python -c "import tensorrt; print('TensorRT', tensorrt.__version__)"

# For Route A you need onnxruntime-gpu WITH the TensorRT EP. The generic PyPI
# onnxruntime-gpu wheel does NOT include it for Jetson — use NVIDIA's Jetson
# Zoo wheel matching your JetPack, e.g.:
#   https://elinux.org/Jetson_Zoo#ONNX_Runtime
# Then confirm the provider is present:
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
#   -> must list 'TensorrtExecutionProvider'

# For Route B's trt_check.py you also need pycuda:
pip install pycuda
```

Keep the `protobuf==3.20.3` pin from the main setup — newer protobuf breaks
mediapipe, which the pipeline still needs for keypoints.

If `TensorrtExecutionProvider` is missing, `--backend tensorrt` prints a clear
error and you should either install the right `onnxruntime-gpu` or fall back to
`--backend onnx` (plain CUDA EP — still GPU-accelerated, just not fused).

---

## 3. Recommended workflow

```bash
cd jetson_deploy
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 1. sanity: plain ONNX (CUDA) backend works and gives correct intents
python fusion/pipeline.py --source ../data/raw/clips/classroom/S01_F04/S01_F04_c001.mp4 \
                          --no-display --backend onnx        # expect F04 -> A05

# 2. (optional, Route B) build + verify native engines
bash fusion/build_engines.sh
python fusion/trt_check.py                                   # argmax must agree

# 3. run with TensorRT EP (Route A). First run builds+caches engines (slow).
python fusion/pipeline.py --source realsense --backend tensorrt

# 4. profile all three backends and record the numbers in docs/WORKLOG.md
python fusion/profile_latency.py --source 0 --backend onnx     --steps 30
python fusion/profile_latency.py --source 0 --backend tensorrt --steps 30
```

Expected shape of the result: `tensorrt` should cut the per-step net time
(emotion+gesture+motion+fusion) noticeably vs `onnx`, while Holistic (the
per-frame cost) is unchanged. If end-to-end latency is still over budget, the
fix is Holistic complexity / stride (see `JETSON_INFERENCE_GUIDE.md` §4), not
more TensorRT.

---

## 4. Numerical accuracy note

- ONNX vs native PyTorch (FP32, CPU): ~1e-6 — already verified in
  `onnx/export_report.json`.
- **TensorRT FP16 vs ONNX: expect up to ~1e-2 on logits.** This is normal for
  half precision and does NOT change the predicted intent — `trt_check.py`
  checks argmax agreement, which is the number that matters. If you need bit-
  closer parity, drop `--fp16` (build FP32 engines): slower, rarely worth it on
  Orin Nano.
- The fusion attention engine includes the boolean key-padding mask (missing-
  cue path). `trt_check.py` exercises it with random `obs` flags; confirm it
  reports argmax agreement = 100% before trusting missing-modality behavior.

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `TensorrtExecutionProvider not available` | `onnxruntime-gpu` lacks the TRT EP — install NVIDIA's Jetson wheel (§2), or use `--backend onnx`. |
| First `--backend tensorrt` run hangs for minutes | Normal — engine compilation. Cached afterward in `onnx/trt_cache/`. Don't kill it. |
| Engine fails to load / garbage output | Built on a different machine or after a JetPack/TensorRT upgrade. Delete `onnx/trt_cache/` and `onnx/*.engine`, rebuild on THIS device. |
| `trtexec: not found` | Use the full path `/usr/src/tensorrt/bin/trtexec`, or `export TRTEXEC=...` before `build_engines.sh`. |
| mediapipe `FieldDescriptor ... 'label'` error | protobuf too new — `pip install protobuf==3.20.3`. |
| motion LSTM engine build warns about loops | TensorRT handles ONNX LSTM but may fall back per-op; acceptable — motion is 3 ms either way. Confirm with `trt_check.py`. |
| Out of memory during build | Close other processes; `trtexec` accepts `--memPoolSize=workspace:512M` to cap it. |

---

## 6. Files added for TensorRT

```
jetson_deploy/
├── TENSORRT_GUIDE.md          this file
├── fusion/
│   ├── pipeline.py            --backend tensorrt (Route A, ORT TensorRT EP)
│   ├── build_engines.sh       Route B: trtexec -> onnx/*.engine (run on Jetson)
│   └── trt_check.py           verify .engine argmax vs ONNX
└── onnx/
    ├── *.onnx                 source graphs (portable)
    ├── *.engine               native TensorRT engines (Jetson-only, gitignored)
    └── trt_cache/             ORT TensorRT EP engine cache (Jetson-only, gitignored)
```

`.engine` files and `trt_cache/` are device-specific build artifacts — do not
commit them or copy them between machines. Rebuild on each device.
