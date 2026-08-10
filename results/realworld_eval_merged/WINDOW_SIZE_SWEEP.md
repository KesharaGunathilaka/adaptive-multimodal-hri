# Study 3 — window-size (lookback span) sweep

Generated 2026-08-05 05:13 · 3 seeds · headline test · `full` recipe (recombination + dropout 0.3 + jitter 0.15), clip-pooled (R1) — matches Study 1/2's protocol, unlike the original single-seed window-level §7.9 sweep.

Prior result (`docs/methodology/07_evaluation.md` §7.9, old classroom-only table, no recombination): x0.5=0.976, x1.0=0.951, x2.0=0.756 — **shorter was better**, deployed spans kept at x1 anyway (matches the unimodal engines' own validated buffers, not chosen for this number).

| Scale | Gesture span | Motion span | Clip acc | Macro-F1 |
|---|---|---|---|---|
| x0.5 | 1.067s | 1.0s | 0.7252 ± 0.0088 | 0.6304 |
| x1.0 | 2.133s | 2.0s | 0.7191 ± 0.0164 | 0.6241 |
| x1.5 | 3.2s | 3.0s | 0.7004 ± 0.0025 | 0.6035 |
| x2.0 | 4.267s | 4.0s | 0.6782 ± 0.0077 | 0.582 |

**Best on final_merged: `x0.5`.** Compare against the deployed x1.0 span and §7.9's old-table trend before changing the deployed lookback.