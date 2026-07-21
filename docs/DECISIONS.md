# DECISIONS — one-line rationale log (append-only, newest first)

- **2026-07-20 [WIN-3060]** `protobuf==3.20.3` + `onnx==1.14.1` pinned — newer protobuf breaks
  mediapipe 0.10.x; pins recorded in requirements.txt for the HPC/Jetson.
- **2026-07-20 [WIN-3060]** Context/CLIP not exported to ONNX — stays PyTorch on Jetson from the
  bundled HF cache; only revisit if the latency budget fails.
- **2026-07-20 [WIN-3060]** Policy thresholds: τ=0.5 (fallback F05/A06), F02 bypass τ=0.30 —
  gives 22/22 clip-level emergency detection at 1.3% false-emergency on test subjects.
- **2026-07-20 [WIN-3060]** Deployed fusion keeps ×1 lookbacks (matches unimodal engines'
  validated buffers) even though ×0.5 scored higher offline — decide after Jetson profiling.

- **2026-07-17 [WIN-3060]** Fusion model selection uses **masked-val early stopping** (mean of
  unmasked + 4 single-masked val accuracies) — plain val selection picked robustness-poor epochs.
- **2026-07-17 [WIN-3060]** Recombination skips V3 #18 (classroom happy+wave+walk F09): without
  direction it is cue-identical to recorded #1 (F01) — synthesizing it would inject label noise.
- **2026-07-17 [WIN-3060]** Missing cues handled by **attention exclusion** (key-padding mask)
  rather than a learned [MISSING] token — token substitution measurably hurt masked accuracy.

- **2026-07-16 [WIN-3060] (user)** S21/S28 label collision: S21 → F04 training; S28 (53 clips) →
  `recombination_pool` (synthetic-generation inputs), promotable to F10 if gesture model says `idle`.
- **2026-07-16 [WIN-3060] (user)** S05 relabeled F02 → F07 per V3 row #14 (folder name unchanged).
- **2026-07-16 [WIN-3060] (user)** Direction cue skipped for fusion v1; revisit via trajectory
  feature only if error analysis shows F01/F09 or F03/F06 confusion dominates.
- **2026-07-16 [WIN-3060] (user)** `data/` is the canonical **curated** dataset (bad clips manually
  removed, all of S27_F06 dropped); `videos/struct/` kept only as the uncurated archive.
- **2026-07-16 [WIN-3060]** `clips.csv` fps/resolution/duration rewritten from OpenCV probe —
  104 kitchen clips had wrong fps (24/30 fps phone video, not 15); windowing must be time-based
  using per-clip probed fps.
- **2026-07-16 [WIN-3060]** Fusion feature table stores each cue in its model's **native class
  order** (see MODEL_AUDIT.md) — remapping to table names happens once at extraction, avoiding
  silent order mismatches between machines.
- **2026-07-16 [WIN-3060]** Cross-machine artifact integrity enforced via
  `docs/checkpoint_manifest.sha256` + WORKLOG entries; git carries hashes, binaries copied manually.
- **2026-07-16 [WIN-3060]** Fusion consumes the CLIP zero-shot scene classifier only (5-dim); the
  SmolVLM2 caption path is out of scope for the fusion cue vector.
