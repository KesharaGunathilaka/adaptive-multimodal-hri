# Fusion per-intent breakdown — `full` recipe on `data/final_merged`

Generated 2026-08-05 20:50 · 3 seeds · headline test clips (n=979) · same code path as `scripts/30_merged_recombination.py`'s winning `full` config.

Headline clip acc **0.7191 ± 0.0164**, macro-F1 **0.6241 ± 0.0226** — reproduces `RECOMBINATION.md`.

## Per intent (mean over 3 seeds, 9 present classes)

|     |   precision |   recall |     f1 |   support |
|:----|------------:|---------:|-------:|----------:|
| F01 |      0.8003 |   0.9158 | 0.854  |       198 |
| F02 |      0.8197 |   0.5179 | 0.6344 |       121 |
| F03 |      0.6817 |   0.7158 | 0.698  |        95 |
| F04 |      0.8387 |   0.5732 | 0.6807 |       132 |
| F05 |      0.5059 |   0.8208 | 0.625  |        80 |
| F06 |      0.4134 |   0.3472 | 0.3705 |        72 |
| F07 |      0.5776 |   0.8213 | 0.6773 |        97 |
| F08 |      0.8894 |   0.7741 | 0.8277 |        90 |
| F10 |      0.9462 |   0.8121 | 0.8737 |        94 |

## Confusion matrix (rows = truth, pooled over 3 seeds)

|          |   F01 |   F02 |   F03 |   F04 |   F05 |   F06 |   F07 |   F08 |   F10 |
|:---------|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| true_F01 |   544 |     8 |     3 |     0 |     6 |    14 |    19 |     0 |     0 |
| true_F02 |    30 |   188 |    22 |     2 |    70 |    41 |     0 |     0 |    10 |
| true_F03 |    12 |     1 |   204 |    10 |    41 |     3 |     6 |     8 |     0 |
| true_F04 |    81 |    29 |    37 |   227 |     3 |     7 |    12 |     0 |     0 |
| true_F05 |     5 |     3 |    21 |     6 |   197 |     5 |     0 |     0 |     3 |
| true_F06 |     0 |     0 |     3 |     3 |     8 |    75 |   127 |     0 |     0 |
| true_F07 |     3 |     0 |     9 |     0 |    17 |     5 |   239 |    18 |     0 |
| true_F08 |     5 |     0 |     0 |    23 |    20 |     0 |    13 |   209 |     0 |
| true_F10 |     0 |     1 |     0 |     0 |    30 |    22 |     0 |     0 |   229 |