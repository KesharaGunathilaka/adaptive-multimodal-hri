# Embedding-level fusion with OUT-OF-FOLD recombination pools

Generated 2026-08-08 18:56 · `data/final_merged` · headline test n=979 · 10 seeds · PCA n_components=32 (fit on DEPLOYED train embeddings, unchanged vs `EMBEDDING_FUSION_FULL.md`) · gesture/emotion recombination pools now drawn from `scripts/60`'s out-of-fold embeddings instead of in-fold; motion/context unchanged (never contaminated).

Reference (in-fold pools, `EMBEDDING_FUSION_FULL.md`): `embed_full` 0.6904±0.0070 vs `probs_full` 0.6993±0.0135, McNemar p=1.0000 (exact tie).

## Results

| Config | Dims | Acc (mean±std) | Macro-F1 (mean±std) | Ensemble acc | Ensemble F1 |
|---|---|---|---|---|---|
| `probs_full` | 24 | 0.6993 ± 0.0135 | 0.5987 ± 0.0164 | 0.7028 | 0.6013 |
| `embed_full_outfold` | 128 | 0.6742 ± 0.0112 | 0.5767 ± 0.0111 | 0.6823 | 0.5833 |
| rules (reference) | — | 0.7120 | 0.6414 | 0.7120 | 0.6414 |

## Significance

| Comparison | n10 | n01 | p | favours | boot mean diff | boot 95% CI |
|---|---|---|---|---|---|---|
| probs_full_vs_rules | 69 | 78 | 0.5095 | right | -0.0094 | [-0.0337, +0.0153] |
| embed_full_outfold_vs_rules | 86 | 115 | 0.0480 | right | -0.0296 | [-0.0582, -0.0010] |
| embed_full_outfold_vs_probs_full | 39 | 59 | 0.0544 | right | -0.0202 | [-0.0398, +0.0000] |

## Reading this

If out-of-fold pools move `embed_full_outfold` meaningfully vs the in-fold `embed_full` (0.6904), that confirms pool contamination was suppressing the embedding representation's real value. If the number barely moves, contamination was not the binding constraint -- the flat tie is a property of the representation/architecture, not of dirty pools, and the dimensionality-overfitting risk flagged in `EMBEDDING_PROBE.md` is the more likely explanation.