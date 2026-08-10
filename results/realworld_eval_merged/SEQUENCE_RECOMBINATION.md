# Phase 3 — trajectory recombination for the window-sequence model (R3)

Generated 2026-08-06 23:44 · `data/final_merged` · headline test (n=979) · 44800 synthetic trajectories covering 448/448 combos · 3 seeds.

## Does closing the augmentation gap change R3's verdict?

| Config | Clip acc | Macro-F1 |
|---|---|---|
| R1 clip-pool (reference, Study 1/2) | 0.7191 | — |
| Rules (reference) | 0.712 | — |
| R3 causal, no recombination (Study 2 baseline, reproduced here) | 0.5329 ± 0.0274 | 0.4324 |
| **R3 causal, WITH trajectory recombination** | **0.7136 ± 0.0172** | **0.6216** |
| R3 bidirectional, WITH recombination (offline ceiling) | 0.7031 ± 0.0046 | 0.5989 |

Recombination moves causal R3 by **+0.1807** (0.5329 → 0.7136).

R3 now **ties** R1 (delta -0.0055, within seed noise) — the augmentation gap was indeed most of the story. Temporal modelling does not clearly help OR hurt once coverage is equal; no strong claim either way from this result alone.

vs rules: +0.0016.