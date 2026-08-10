# Phase 2 — real video degradation (not simulated)

Generated 2026-08-07 07:58 · `data/final_merged` · stratified subsample of **70 clips per condition** (of the 979-clip headline test set — full re-extraction across every condition was not tractable; see script docstring, including a 2026-08-06 crash post-mortem: the first version leaked memory across conditions and lost results on crash, fixed by running one subprocess per condition with incremental result writing) · same `full`-recipe fusion model as elsewhere (cached across conditions), 3-seed majority vote.

Unlike `ROBUSTNESS_BATTERY.md` part B (which corrupts probability vectors post-hoc), this corrupts the ACTUAL VIDEO PIXELS before Holistic/face-detection/CLIP ever run.

**8/8 conditions completed.**

| Condition | n | Fusion acc | Rule acc | emo obs | ges obs | mot obs |
|---|---|---|---|---|---|---|
| clean | 63 | 0.6984 | 0.7778 | 1.0 | 1.0 | 1.0 |
| blur_9 | 63 | 0.6349 | 0.6667 | 1.0 | 1.0 | 1.0 |
| blur_21 | 63 | 0.4603 | 0.5397 | 1.0 | 1.0 | 1.0 |
| dark_0.35 | 63 | 0.6667 | 0.7619 | 1.0 | 1.0 | 1.0 |
| dark_0.15 | 63 | 0.5556 | 0.6349 | 0.984 | 0.937 | 0.937 |
| downsample_0.35 | 63 | 0.6508 | 0.6349 | 1.0 | 1.0 | 1.0 |
| downsample_0.15 | 63 | 0.4921 | 0.5079 | 1.0 | 1.0 | 1.0 |
| jpeg_10 | 63 | 0.5714 | 0.5714 | 1.0 | 1.0 | 1.0 |

Clean baseline: fusion 0.6984 vs rules 0.7778 (delta -0.0794).
Largest fusion-vs-rules gap under degradation: `dark_0.35` — fusion 0.6667 vs rules 0.7619 (delta -0.0952).