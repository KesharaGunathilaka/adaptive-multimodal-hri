# Gesture Model - Stage 1: Architecture Comparison

Generated: 2026-07-15 04:01 · shortened schedule (40 epochs max, patience 8), shared recipe (class-weighted CE + label smoothing, AdamW, cosine warmup).

| model           |   params_k |   size_mb |   train_min |   accuracy |   balanced_accuracy |   f1_weighted |   f1_macro |
|:----------------|-----------:|----------:|------------:|-----------:|--------------------:|--------------:|-----------:|
| TCN             |      683.3 |      2.61 |         2.8 |     0.9261 |              0.9386 |        0.9263 |     0.9286 |
| BiGRU           |      606.1 |      2.31 |         2.9 |     0.9264 |              0.9339 |        0.9266 |     0.928  |
| TinyTransformer |      294.4 |      1.12 |         2.8 |     0.9123 |              0.9383 |        0.913  |     0.9231 |

**Winner (val macro-F1): TCN**

- Curves: `comparison_curves.png`

## Next step

```
python scripts/train.py --model TCN
```