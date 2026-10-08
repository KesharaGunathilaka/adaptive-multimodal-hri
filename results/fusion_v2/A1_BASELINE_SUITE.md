# WP1 - comparison against published fusion methods

All methods share the v4 taxonomy, cue features, full-rubric augmentation (24000 synthetic rows), seed list and the same 773 recorded test clips. Only the fusion mechanism varies. Neural methods: 10 seeds, majority vote.

| method | accuracy | macro-F1 | per-seed mean +/- std | vs ours (p) |
|---|---|---|---|---|
| `rules` | **0.7646** | 0.7385 | - | 1.68e-07 |
| `unimodal-emotion` | **0.4191** | 0.2753 | - | 7.44e-73 |
| `unimodal-gesture` | **0.4075** | 0.2500 | - | 3.9e-71 |
| `unimodal-context` | **0.1268** | 0.0395 | - | 8.35e-140 |
| `late-product` | **0.5653** | 0.4237 | - | 1.2e-40 |
| `concat-mlp` | **0.8008** | 0.7720 | 0.8032 +/- 0.0046 | 0.00149 |
| `gbt` | **0.7050** | 0.6897 | 0.6977 +/- 0.0054 | 1.35e-16 |
| `channel-attn` | **0.7853** | 0.7506 | 0.7893 +/- 0.0040 | 6.57e-05 |
| `lmf` | **0.8137** | 0.7843 | 0.8098 +/- 0.0091 | 0.267 |
| `gmu` | **0.7969** | 0.7628 | 0.7977 +/- 0.0041 | 0.000912 |
| `cross-attn` | **0.8176** | 0.7920 | 0.8150 +/- 0.0086 | 0.804 |
| `self-attn` **(ours)** | **0.8202** | 0.7968 | 0.8182 +/- 0.0061 | - |

`p` is a McNemar exact test of that method against ours on the same clips. Unimodal floors, late-product and rules are deterministic given the cue features, so they have no seed variance.
