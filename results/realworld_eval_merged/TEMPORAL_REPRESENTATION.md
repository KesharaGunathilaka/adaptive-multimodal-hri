# Study 2 — temporal representation (R1 clip-pool vs R2 per-window vs R3 sequence)

Generated 2026-08-05 04:49 · 3 seeds · headline test · clip acc ± std / macro-F1. All configs share the `full` recipe (recombination + dropout 0.3 + jitter 0.15).

Extends `docs/methodology/07_evaluation.md` §7.2 (2026-07-28, old table): `train=window,infer=majority-vote`=0.342±0.028 vs the adopted `train=clip-mean,infer=clip-mean`=0.374±0.030. R1/R2 below re-measure that same comparison on `final_merged` with promoted checkpoints + recombination; R3 is new — an order-aware model over the window sequence R1 averages away.

| Representation | Clip acc | Macro-F1 |
|---|---|---|
| R1 clip-pool (incumbent, = Study 1 `full`/self-attention) | 0.7191 ± 0.0164 | 0.6241 |
| R2 per-window train, majority-vote | 0.7109 ± 0.0096 | 0.6113 |
| R3 sequence, bidirectional (offline ceiling) | 0.5328 ± 0.0376 | 0.429 |
| R3 sequence, causal (deployable, full buffer) | 0.5329 ± 0.0274 | 0.4324 |

## Causal model swept over live buffer length (trailing K windows)

| K windows | ≈ seconds (8/30s stride) | Clip acc | Macro-F1 |
|---|---|---|---|
| 4 | 1.07 | 0.5478 ± 0.0344 | 0.4376 |
| 8 | 2.13 | 0.5574 ± 0.0371 | 0.4463 |
| 16 | 4.27 | 0.556 ± 0.0322 | 0.4481 |
| 40 | 10.67 | 0.5329 ± 0.0274 | 0.4324 |

**R1 vs R2:** R1 (mean-pool) confirms the §7.2 rule and beats R2 by 0.0082 — consistent with established project policy, though the margin is far smaller than §7.2's old-table 0.032 gap now that BOTH R1 and R2 get recombination -- recombination appears to help per-window training almost as much as clip-pooled training.
**R1 vs R3 (causal, full buffer):** R3 scores 0.1862 lower than R1, but this is **NOT a clean representation-only comparison** — R3 trains on real window sequences only (no recombination augmentation exists for the sequence representation; out of scope for v1, see `fusion/model/sequences.py`), while R1/R2 both train on ~46K samples including 44,800 synthetic recombination samples. R3's high val accuracy (~0.86-0.90) alongside a much lower headline-test score is the same real-vs-synthetic-coverage gap recombination was built to close for R1 -- most likely explanation for the gap here is missing augmentation, not that order-awareness is unhelpful. **Do not conclude temporal modeling doesn't work from this table** -- extending recombination to synthetic sequences is the natural next step before that verdict can be drawn.

The trailing-K sweep is still informative on its own terms (all K use the SAME causal model, so it isolates buffer length, not augmentation): accuracy is flat-to-slightly-better from K=8 (2.13s) through the full buffer, and K=4 (1.07s) loses very little — a live system does not need long history for this decision.

**Deployment recommendation (this study):** ship R1 (clip-pool, self-attention, the incumbent) — it is the best-verified, best-augmented option. Revisit R3 only after building a recombination analogue for sequences.