# Gap decomposition — complete `data/final_merged` (both contexts)

Generated 2026-08-04 22:45 · 2869 clips · 62 V3 rows · protocol matches `results/realworld_eval_final/GAP_DECOMPOSITION.md` (clip-level, 4s mean-pooled, actor-disjoint val, no train-time augmentation, 3 seeds).

**Ceiling**: 0 colliding cue tuple(s) at the intent level (F09's removal deleted the classroom direction collision) -> clip-weighted ceiling = **1.0** (n=2869).

## Headline test (excl. row #58's 24 train-derived clips)

| Configuration | Clip acc | Clip macro-F1 |
|---|---|---|
| Rules + oracle cues | 1.0 | 0.9 |
| Fusion + oracle cues | 0.619 ± 0.0484 | 0.5143 |
| Rules + real cues | 0.712 | 0.6414 |
| Fusion + real cues | 0.5475 ± 0.0251 | 0.438 |

## Decomposition

- ceiling 1.0 - fusion+oracle 0.619 = **fusion generalisation cost 0.381**
- fusion+oracle 0.619 - fusion+real 0.5475 = **perception cost 0.072**

- rules+oracle scored 1.0 against the 1.0 ceiling — the rubric is fully expressible and the rule implementation is not a strawman (after remapping the legacy F09 branch to F01; see `merged_gap.rule_predict`'s docstring for why that remap belongs there and not in the shared `fusion/baselines/rule_based.py`, which `data/old` still needs).

## Worst 15 rows by oracle-minus-real drop (perception-attributable failures)

|   v3_row | context   | intent   |   oracle_hit |   real_hit |   drop |
|---------:|:----------|:---------|-------------:|-----------:|-------:|
|       57 | kitchen   | F05      |            1 |  0.0357143 |   0.96 |
|       58 | kitchen   | F06      |            1 |  0.421053  |   0.58 |
|       56 | kitchen   | F04      |            1 |  0.529412  |   0.47 |
|       28 | classroom | F07      |            1 |  0.636364  |   0.36 |
|       29 | classroom | F08      |            1 |  0.653061  |   0.35 |
|       60 | kitchen   | F08      |            1 |  0.731707  |   0.27 |
|       31 | classroom | F10      |            1 |  0.826923  |   0.17 |
|       32 | kitchen   | F01      |            1 |  0.851852  |   0.15 |
|       62 | kitchen   | F10      |            1 |  0.857143  |   0.14 |
|       63 | kitchen   | F04      |            1 |  0.891304  |   0.11 |
|       24 | classroom | F03      |            1 |  0.886792  |   0.11 |
|       18 | classroom | F01      |            1 |  0.916667  |   0.08 |
|       59 | kitchen   | F07      |            1 |  0.97619   |   0.02 |
|       22 | classroom | F01      |            1 |  0.980769  |   0.02 |
|       53 | kitchen   | F02      |            0 |  0         |   0    |

## Worst 15 rows where oracle ALSO fails (fusion-generalisation-attributable)

|   v3_row | context   | intent   |   oracle_hit |   real_hit |   drop |
|---------:|:----------|:---------|-------------:|-----------:|-------:|
|       53 | kitchen   | F02      |            0 |  0         |   0    |
|       26 | classroom | F05      |            0 |  0         |   0    |
|       23 | classroom | F02      |            0 |  0         |   0    |
|       55 | kitchen   | F03      |            0 |  0         |   0    |
|       54 | kitchen   | F02      |            0 |  0.0263158 |  -0.03 |
|       27 | classroom | F06      |            0 |  0.0754717 |  -0.08 |
|       25 | classroom | F04      |            0 |  0.230769  |  -0.23 |