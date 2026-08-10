# Study 1 — fusion architecture ablation (clip-pooled input, R1)

Generated 2026-08-05 04:02 · 3 seeds · headline test (excl. row #58 derived clips) · clip acc ± std / macro-F1 · recombination n_per_combo=100.

Reference: **rule-based baseline = 0.712** headline clip-acc. The question is whether any *architecture* beats it and the incumbent self-attention head on the same pooled input.

| Model | Params | plain | full |
|---|---|---|---|
| Rule-based (baseline) | — | 0.712 / 0.6414 | 0.712 / 0.6414 |
| Concat-MLP (floor) | 12,618 | 0.5036 ± 0.0197 / 0.3991 | 0.6837 ± 0.0154 / 0.5819 |
| GBT (LightGBM) | — | 0.5168 ± 0.0088 / 0.4169 | 0.684 ± 0.0097 / 0.6008 |
| Self-attention / transformer | 70,090 | 0.5475 ± 0.0251 / 0.438 | 0.7191 ± 0.0164 / 0.6241 |
| Cross-attention | 69,834 | 0.5203 ± 0.0209 / 0.4172 | 0.699 ± 0.0096 / 0.5932 |
| Channel attn (CAM) | 4,896 | 0.5451 ± 0.0293 / 0.4417 | 0.6581 ± 0.0097 / 0.5484 |
| GMU (gated) | 8,970 | 0.5294 ± 0.0062 / 0.4336 | 0.7065 ± 0.0024 / 0.6049 |
| LMF (low-rank tensor) | 6,730 | 0.5536 ± 0.0284 / 0.4388 | 0.6994 ± 0.0056 / 0.6023 |

**Best architecture (full config): `self_attention` at 0.7191** vs rule baseline 0.712. Learned fusion beats the explicit rubric on real cues.

Cells are headline clip-acc±std / macro-F1. `full` = recombination + dropout(0.3) + jitter(0.15); `plain` = real cues only.