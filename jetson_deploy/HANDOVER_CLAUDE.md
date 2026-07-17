# CLAUDE.md — Jetson Orin Nano Inference Testing Handover

> **Place this file at the root of this folder on the Jetson as `CLAUDE.md`**
> (i.e. `~/hri/jetson_deploy/CLAUDE.md`, next to `README.md`). Claude Code /
> the Claude VS Code extension reads it automatically when you open this
> folder as the workspace. It is the single source of truth for continuing
> this work on this machine.
>
> **This machine has no git repo and no internet-dependent context.** Only
> this `jetson_deploy/` folder exists here — it was copied over by USB from
> the main dev machine (Windows, RTX 3060). Don't try `git log`/`git status`
> for history; there isn't any here. Cross-machine history lives in the main
> repo's `docs/WORKLOG.md`, which this machine doesn't have a copy of.

---

## 1. What this folder is

A **self-contained, inference-only** copy of 4 trained perception models for
an Adaptive Human-Robot Interaction final-year project:

| Modality | What it detects | Checkpoint |
|---|---|---|
| `modalities/emotion/` | 7 facial emotions (Surprise/Fear/Disgust/Happy/Sad/Anger/Neutral) | `checkpoints/finetuned_MobileNetV2.pth` |
| `modalities/gesture/` | 8 hand/body gestures (idle/wave/point/thumbs_up/thumbs_down/beckoning/raise_hand/both_hands_up) | `checkpoints/best_TCN.pth` + `model_config.json` |
| `modalities/motion/` | 4 body-motion states (sitting/standing/walking/stepping_back) | `checkpoints/best_model_finetuned.pt` |
| `modalities/context/` | Scene (classroom/kitchen, CLIP zero-shot) + free-form situation caption (SmolVLM2-500M) | no local weights — loads from `hf_cache/` |

These 4 models will eventually feed a **fusion model** (trained separately, on
the Windows/HPC machines) that predicts one of 10 human intents (F01–F10) for
a downstream robot policy. **That fusion model is NOT in this folder** — this
machine's job is narrower: prove each of the 4 models runs correctly and
fast enough, standalone, on the actual target hardware (Jetson Orin Nano +
Intel RealSense D455). Full project context, if ever needed, lives in the
main repo (`HANDOVER_CLAUDE.md` at its root + `docs/`) — not copied here on
purpose, to keep this folder light.

Each modality's own `README.md` (inside its folder) documents known accuracy
numbers and failure modes per model — read the relevant one before judging a
live demo that looks occasionally wrong (e.g. motion's `stepping_back` is
known-weak in kitchen-like framing; this is expected, not a bug to chase).

## 2. Current status — what is DONE and NOT done

**Done (on the Windows dev machine, before this folder was copied over):**
- All 4 models trained, checkpointed, and verified to load correctly
  (see each modality's README for accuracy numbers).
- This `jetson_deploy/` folder assembled and load-tested on x86 (checkpoints
  load, `hf_cache/` verified to work fully offline).
- `JETSON_SETUP_GUIDE.md` (in this same folder) written — step-by-step Jetson
  environment setup: JetPack-matched PyTorch, RealSense SDK built from source
  (D455, RSUSB backend), remaining Python deps, offline HF cache env vars.

**NOT done — this is genuinely the first attempt at any of this:**
- Nothing has been executed on this specific Jetson board yet. JetPack/L4T
  version on this board is **unknown** as of this handover — check it first
  (guide §0) since it changes the PyTorch install command.
- RealSense SDK has never been built on this board.
- No live inference run, no latency numbers, no confirmation any model
  actually hits real-time speed on Orin Nano hardware (that's the whole
  point of this exercise — the models were only ever benchmarked on x86 GPUs
  before now).

## 3. Your job on this machine

Work through **`JETSON_SETUP_GUIDE.md`** in this folder, in order, top to
bottom. It already covers: identifying the JetPack version, system prep
(power mode, `jetson_clocks`), Python venv + JetPack-matched PyTorch, RealSense
SDK build for the D455, remaining pip deps, offline HF cache setup, per-model
sanity checks, and finally live camera + video-file inference commands for
all 4 modalities. It also has a troubleshooting table for the failure modes
already anticipated (wrong torch wheel, no aarch64 `pyrealsense2` wheel, USB2
vs USB3 port issues, etc.) — check there before treating an error as novel.

Concretely, in order:
1. Run the Step 0 diagnostic commands (JetPack/L4T/Python/CUDA version, free
   disk space) and use them to pick the right branch of Step 3 (PyTorch
   install command differs for JetPack 5 vs 6).
2. Follow Steps 1–6 (system prep → venv/PyTorch → RealSense build → remaining
   deps → HF offline env vars → sanity import checks).
3. Run each modality live (Step 8) with the RealSense D455 connected via
   USB3. Confirm the printed checkpoint path at each script's startup matches
   the table in §1 above — this project already lost time once to a folder
   copy silently resetting a default checkpoint to a weaker baseline, so
   treat that startup line as load-bearing, not decoration.
4. If a camera isn't handy for a given check, `video.py` per modality (guide
   §9) runs on a file instead — useful for isolating "is the model broken"
   from "is the camera/USB setup broken."
5. **Measure and record inference latency per modality** (wall-clock per
   frame/window is enough — `time.perf_counter()` around the model call, or
   just watch `jtop` during a live run for GPU%/frame-rate). This number is
   the actual deliverable of this exercise: the eventual fusion pipeline
   needs all 4 models to run within a shared real-time budget on this exact
   hardware, and nobody has measured that yet.

## 4. How to record what you find

There's no git here and no direct path back to the main repo's
`docs/WORKLOG.md` from this machine. Instead:

- Keep a running log in **`JETSON_TEST_LOG.md`** (template already in this
  folder, same entry format as the main repo's `docs/WORKLOG.md` so it drops
  in cleanly later) — what you ran, what broke, how you fixed it, and the
  latency/accuracy numbers you observed per modality.
- When the user is back at the Windows machine, they'll copy
  `JETSON_TEST_LOG.md` back over (USB, same as the outbound trip) and merge
  its findings into the main repo's `docs/WORKLOG.md` — so write it assuming
  a future reader who wasn't in this session, not just as scratch notes.
- If you hit a genuine environment problem the setup guide didn't anticipate
  (a real JetPack/library incompatibility, not a typo), document the fix in
  `JETSON_TEST_LOG.md` in enough detail to be copied into
  `JETSON_SETUP_GUIDE.md` later — the guide should get more accurate over
  time, not just this one run.

## 5. Guardrails

- **Don't attempt ONNX/TensorRT export or model quantization here.** That's
  a separate, later phase of the main project (Phase 3 in the main repo's
  handover doc) — this machine's job right now is a native-PyTorch
  correctness + latency smoke test, not the optimized deployment build.
  Getting pulled into TensorRT conversion now is scope creep for this session.
- **Don't modify checkpoints, model code, or retrain anything here.** If a
  model looks wrong, that's a finding to record (§4), not something to fix
  on this machine — fixes happen on the training machine with the full
  dataset, then a new checkpoint gets copied over.
- **Don't assume internet access.** The context modality is deliberately
  pre-cached in `hf_cache/` specifically so it works offline — if anything
  tries to hit the network, that's a misconfiguration (see the guide's
  troubleshooting table), not a reason to just enable Wi-Fi and move on;
  the deployed robot won't have reliable internet either.
- If something in this handover doc or the setup guide is wrong or
  outdated, say so in `JETSON_TEST_LOG.md` rather than silently working
  around it — this doc will be reused for the next board/session.
