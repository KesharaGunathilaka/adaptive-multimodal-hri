# Embedding-level fusion at the deployed (full-recipe) operating point

Generated 2026-08-08 16:45 · `data/final_merged` · headline test n=979 · 10 seeds · PCA n_components=32 (z-scored) · recombination + dropout 0.3 + jitter 0.15 + masked-val selection · in-fold pools (known contamination risk, see `POOL_PURITY.md`/`EMBEDDING_PROBE.md`, not fixed in this pass).

Reference: probs-only full recipe already established at **0.7133 ± 0.0160** (`SIGNIFICANCE.md`, `scripts/49`). This run reproduces that number fresh (`probs_full`) via the SAME generic training loop as `embed_full`, for a paired, apples-to-apples comparison rather than citing the old number across different code.

## Results

| Config | Dims | Acc (mean±std) | Macro-F1 (mean±std) | Ensemble acc | Ensemble F1 |
|---|---|---|---|---|---|
| `probs_full` | 24 | 0.6993 ± 0.0135 | 0.5987 ± 0.0164 | 0.7028 | 0.6013 |
| `embed_full` | 128 | 0.6904 ± 0.0070 | 0.5962 ± 0.0086 | 0.7017 | 0.6063 |
| rules (reference) | — | 0.7120 | 0.6414 | 0.7120 | 0.6414 |

## Significance

| Comparison | n10 | n01 | p | favours | boot mean diff | boot 95% CI |
|---|---|---|---|---|---|---|
| probs_full_vs_rules | 69 | 78 | 0.5095 | right | -0.0094 | [-0.0337, +0.0153] |
| embed_full_vs_rules | 95 | 105 | 0.5246 | right | -0.0103 | [-0.0388, +0.0184] |
| embed_full_vs_probs_full | 44 | 45 | 1.0000 | right | -0.0009 | [-0.0194, +0.0184] |

## Reading this

If `embed_full` beats `probs_full` significantly here despite losing significantly in the plain test (`EMBEDDING_FUSION.md`), that means recombination is doing more useful work with the richer representation than dropout/jitter alone could exploit -- plausible, since embeddings give the recombination sampler more real structure to draw from per class. If it still loses or ties, the representation itself is the bottleneck, not the lack of augmentation, and the pool-purity contamination (`POOL_PURITY.md`) is the more likely explanation -- worth testing with out-of-fold pools before concluding embeddings don't help at all.