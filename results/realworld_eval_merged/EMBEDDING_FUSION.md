# Embedding-level fusion — does richer representation beat probabilities?

Generated 2026-08-08 16:04 · `data/final_merged` · headline test n=979 · 10 seeds · PCA n_components=32 per modality (fit on train, observed rows only) · plain (no recombination/augmentation) — isolates the representation question.

## Results

| Config | Dims | Acc (mean±std) | Macro-F1 (mean±std) | Ensemble acc | Ensemble F1 |
|---|---|---|---|---|---|
| `probs_only` | 24 | 0.5552 ± 0.0245 | 0.4398 ± 0.0233 | 0.5567 | 0.4442 |
| `embed_only` | 128 | 0.5292 ± 0.0291 | 0.4271 ± 0.0248 | 0.5260 | 0.4257 |
| `embed_probs` | 152 | 0.5104 ± 0.0379 | 0.4065 ± 0.0377 | 0.5066 | 0.4019 |
| rules (reference) | — | 0.7120 | 0.6414 | 0.7120 | 0.6414 |

## Significance

| Comparison | n10 | n01 | p | favours | boot mean diff | boot 95% CI |
|---|---|---|---|---|---|---|
| probs_only_vs_rules | 79 | 231 | 0.0000 | right | -0.1552 | [-0.1890, -0.1205] |
| embed_only_vs_rules | 89 | 271 | 0.0000 | right | -0.1859 | [-0.2217, -0.1491] |
| embed_probs_vs_rules | 98 | 299 | 0.0000 | right | -0.2054 | [-0.2431, -0.1665] |
| embed_only_vs_probs_only | 27 | 57 | 0.0014 | right | -0.0307 | [-0.0490, -0.0123] |
| embed_probs_vs_probs_only | 34 | 83 | 0.0000 | right | -0.0502 | [-0.0715, -0.0296] |

## Reading this

`_vs_rules` favours=left means the config beat rules; `_vs_probs_only` favours=left means the embedding config beat the probability baseline. A CI containing 0 means not significant at this seed count — read `scripts/49_significance.py`'s standing lesson (σ≈0.016 needs ~10+ seeds to resolve gaps below ~0.023) before treating any single-digit point difference here as established.