# Rebalancing recombination's intent prior (T02 `point` fix)

Generated 2026-08-08 06:07 · `data/final_merged` · headline test n=979 · 5 seeds per mode · budget fixed at ~44,800 synthetic samples for every mode.

## Why

Fusion scored 0.038 on row #27 and 0.381 on row #55 despite **clean cues** (row #27: anger 0.80, point 0.65, walking 0.81 — rules read the same vectors correctly 64% of the time). Perception is not the cause. Uniform `n_per_combo` weights cue COMBOS equally, which makes each intent's synthetic share proportional to its combo count — and inside a cue family that is lopsided:

| Cue family | Majority branch | Minority branch |
|---|---|---|
| `Anger + point` | F07 6/8 (75%) | **F06 2/8 (25%)** ← row #27 |
| `Happy + point` | F05 6/8 (75%) | **F03 2/8 (25%)** ← row #55 |

## Modes

| Mode | Intent share of synthetic set |
|---|---|
| `uniform` | F01 23%, F02 16%, F03 11%, F04 12%, F05 20%, F06 6%, F07 7%, F08 4%, F10 2% |
| `sqrt` | F01 17%, F02 14%, F03 12%, F04 12%, F05 16%, F06 9%, F07 9%, F08 7%, F10 5% |
| `intent` | F01 11%, F02 11%, F03 11%, F04 11%, F05 11%, F06 11%, F07 11%, F08 11%, F10 11% |
| `family` | F01 23%, F02 16%, F03 12%, F04 12%, F05 16%, F06 9%, F07 6%, F08 4%, F10 2% |

## Results

**Mode selected on VALIDATION macro-F1: `intent`.** Test columns are reported for all modes for transparency, but played no part in the choice.

| Mode | Val macro-F1 | Test acc | Test macro-F1 | Ensemble acc | Ensemble macro-F1 |
|---|---|---|---|---|---|
| `uniform` | 0.7541 | 0.7134 ± 0.0175 | 0.6137 ± 0.0242 | 0.7099 | 0.6118 |
| `sqrt` | 0.7678 | 0.7216 ± 0.0072 | 0.6220 ± 0.0073 | 0.7242 | 0.6258 |
| `intent` **←selected** | 0.7838 | 0.7299 ± 0.0139 | 0.6346 ± 0.0168 | 0.7293 | 0.6331 |
| `family` | 0.7589 | 0.7354 ± 0.0113 | 0.6425 ± 0.0149 | 0.7303 | 0.6334 |
| rules (reference) | — | 0.7120 | 0.6414 | 0.7120 | 0.6414 |

## The two spotlight rows (ensemble)

| Mode | row #27 (F06) | row #55 (F03) |
|---|---|---|
| `uniform` | 0.0377 | 0.3571 |
| `sqrt` | 0.0189 | 0.4762 |
| `intent` | 0.0566 | 0.5714 |
| `family` | 0.0755 | 0.6429 |
| rules | 0.6415 | 0.5714 |

## Per-intent recall (ensemble)

| Mode | F01 | F02 | F03 | F04 | F05 | F06 | F07 | F08 | F10 |
|---|---|---|---|---|---|---|---|---|---|
| `uniform` | 0.919 | 0.521 | 0.663 | 0.568 | 0.825 | 0.250 | 0.825 | 0.778 | 0.830 |
| `sqrt` | 0.929 | 0.512 | 0.716 | 0.568 | 0.825 | 0.264 | 0.835 | 0.789 | 0.883 |
| `intent` | 0.909 | 0.504 | 0.768 | 0.568 | 0.800 | 0.278 | 0.835 | 0.833 | 0.904 |
| `family` | 0.929 | 0.545 | 0.789 | 0.568 | 0.800 | 0.306 | 0.835 | 0.767 | 0.840 |
| rules | 0.763 | 0.727 | 0.590 | 0.515 | 0.762 | 0.722 | 0.814 | 0.711 | 0.830 |

## Selected mode (`intent`) vs rules — paired tests

- McNemar exact: n10=80 (fusion right/rules wrong), n01=63, p=0.1807, favours **fusion**
- Bootstrap over clips: mean diff +0.0173, 95% CI [-0.0062, +0.0419]  (**contains 0**)
- 91.5% of resamples favour fusion

**Not significant on overall accuracy** — read the per-intent recall and spotlight-row tables instead, which is where rebalancing acts.