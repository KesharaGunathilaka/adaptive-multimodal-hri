# Rubric-driven cue recombination — `data/final_merged`

Generated 2026-08-04 22:54 · 44800 synthetic samples from 448/448 combinatorial cue tuples (n_per_combo=100), sourced from TRAIN-split real cue vectors bucketed by GROUND TRUTH class (`fusion/model/recombine_merged.py`).

Reference numbers from the gap decomposition (same protocol, no recombination): rules+real **0.608**, fusion+real (plain) **0.475** — the bar recombination must clear to recover the G1 claim on unseen cue combinations.

## Headline test (excl. row #58's derived clips), 3 seeds

| Config | dropout | jitter | recomb | Clip acc | Clip macro-F1 |
|---|---|---|---|---|---|
| plain | 0.0 | 0.0 | no | 0.5475 ± 0.0251 | 0.438 ± 0.0228 |
| augmented | 0.3 | 0.15 | no | 0.5264 ± 0.027 | 0.4267 ± 0.0367 |
| recomb_only | 0.0 | 0.0 | yes | 0.7085 ± 0.0076 | 0.6147 ± 0.0091 |
| full | 0.3 | 0.15 | yes | 0.7191 ± 0.0164 | 0.6241 ± 0.0226 |

**Best: `full` at 0.7191.** Beats rules+real (0.608) — the G1 claim is recovered on the honest (unseen-combination) split.

## F02 (emergency) rows #23/#53/#54 — the safety-critical check

Gap decomposition found these fail even with ORACLE (perfect) cues; the question is whether recombination (which explicitly includes every F02 combination) fixes them.

|   v3_row | intent   |     mean |   size |
|---------:|:---------|---------:|-------:|
|       23 | F02      | 0.522727 |     44 |
|       53 | F02      | 0.358974 |     39 |
|       54 | F02      | 0.657895 |     38 |