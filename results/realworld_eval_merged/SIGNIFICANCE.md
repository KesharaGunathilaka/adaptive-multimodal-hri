# Is the fusion-vs-rules headline statistically significant?

Generated 2026-08-08 05:18 · `data/final_merged` · headline test n=979 · `full` recipe (recombination n_per_combo=100 → 44800 synthetic samples, dropout 0.3, jitter 0.15, masked-val selection) · **10 seeds**.

Replaces the unqualified claim in `SCENARIO_TEST_REPORT.md` ("Fusion beats rules by 0.0071 acc — the G1/T05 claim holds"), which rested on a margin less than half its own seed standard deviation.

## Headline

| | Clip acc | Macro-F1 |
|---|---|---|
| Rules (deterministic) | 0.7120 | 0.6414 |
| Fusion, mean ± std over 10 seeds | 0.7133 ± 0.0160 | 0.6146 ± 0.0212 |
| Fusion, 10-seed majority vote | 0.7130 | 0.6140 |

**Fusion accuracy 95% CI over seeds: [0.7019, 0.7247]** — **contains rules' 0.7120**, so a differently-seeded run could plausibly land at or below the rule baseline.

## McNemar's exact test (paired, per clip)

Only clips where exactly one system is correct carry information. `n10` = fusion right / rules wrong; `n01` = rules right / fusion wrong.

| Comparison | n10 | n01 | discordant | p | favours |
|---|---|---|---|---|---|
| 10-seed ensemble vs rules | 73 | 72 | 145 | 1.0000 | fusion |
| seed 0 vs rules | 71 | 42 | 113 | 0.0081 | fusion |
| seed 1 vs rules | 74 | 83 | 157 | 0.5233 | rules |
| seed 2 vs rules | 72 | 71 | 143 | 1.0000 | fusion |
| seed 3 vs rules | 76 | 74 | 150 | 0.9350 | fusion |
| seed 4 vs rules | 71 | 87 | 158 | 0.2326 | rules |
| seed 5 vs rules | 71 | 54 | 125 | 0.1521 | fusion |
| seed 6 vs rules | 71 | 86 | 157 | 0.2638 | rules |
| seed 7 vs rules | 75 | 87 | 162 | 0.3875 | rules |
| seed 8 vs rules | 76 | 57 | 133 | 0.1182 | fusion |
| seed 9 vs rules | 75 | 78 | 153 | 0.8716 | rules |

**1 of 10 seeds beat rules significantly (p<0.05).**

## Bootstrap over clips (ensemble − rules accuracy)

10,000 resamples of the 979 headline clips, resampled as pairs so the within-clip pairing is preserved.

- mean difference **+0.0009**
- 95% CI **[-0.0235, +0.0255]**  (**contains 0**)
- 51.0% of resamples favour fusion

## Verdict

On this test set the difference between fusion and rules is **not statistically significant** (McNemar p=1.0000). The two systems are statistically indistinguishable on overall headline accuracy; any claim of superiority must rest on the axes where they genuinely diverge (missing-cue robustness T03, F02 recall), not on this number.

Note both systems are scored against V3 intent labels that are themselves `rule_intent()`'s output, so rules+oracle = 1.000 by construction (`GAP_DECOMPOSITION_MERGED.md`). Overall accuracy on this dataset is therefore an arena that structurally favours the rule baseline; this test quantifies how much of the remaining difference is signal rather than noise.