# Embedding probe — pipeline sanity check + purity-gap comparison

Generated 2026-08-08 15:55 · linear probe (`LogisticRegression`) fit on TRAIN pooled embeddings, scored on train/val/test.

## 1. Pipeline sanity — does the probe recover known accuracy?

| Modality | Probe train | Probe val | Probe test | Argmax test (reference) |
|---|---|---|---|---|
| emotion | 0.9973 | 0.7980 | 0.7523 | 0.7617 |
| gesture | 1.0000 | 0.9966 | 0.8495 | 0.8643 |
| motion | 0.8511 | 0.7241 | 0.5653 | 0.6710 |

Close agreement between probe-test and argmax-test confirms the extraction pipeline (`scripts/54`/`55`) produced embeddings that carry the same classification signal as the deployed classifier heads -- a large mismatch here would point at a pipeline bug, not a property of embeddings.

## 2. Does the train/test gap widen in embedding space?

| Modality | Fine-tuned on train? | Probe train−test gap | Argmax train−test gap (ref) | Worse? |
|---|---|---|---|---|
| emotion | **yes** | +0.2450 | +0.1972 | **YES** |
| gesture | **yes** | +0.1505 | +0.1357 | **YES** |
| motion | no | +0.2858 | -0.0160 | **YES** |

If gesture/emotion (fine-tuned on train) show a probe gap wider than their argmax gap, embedding-level recombination pools inherit `POOL_PURITY.md`'s contamination at least as badly as probability pools, likely worse given the extra capacity a 1280/128-d vector has to encode train-specific detail a 7/8-way softmax cannot.