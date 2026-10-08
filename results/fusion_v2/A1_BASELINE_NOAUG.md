# WP1 - comparison against published fusion methods

All methods share the v4 taxonomy, cue features, full-rubric augmentation (0 synthetic rows), seed list and the same 773 recorded test clips. Only the fusion mechanism varies. Neural methods: 10 seeds, majority vote.

| method | accuracy | macro-F1 | per-seed mean +/- std | vs ours (p) |
|---|---|---|---|---|
| `rules` | **0.7646** | 0.7385 | - | 1.51e-28 |
| `unimodal-emotion` | **0.3079** | 0.2698 | - | 2.71e-27 |
| `unimodal-gesture` | **0.4722** | 0.3454 | - | 0.00184 |
| `unimodal-context` | **0.1035** | 0.0208 | - | 2.54e-70 |
| `late-product` | **0.5847** | 0.5263 | - | 5.15e-09 |
| `concat-mlp` | **0.4916** | 0.4271 | 0.4964 +/- 0.0246 | 0.00956 |
| `gbt` | **0.4994** | 0.4717 | 0.4994 +/- 0.0000 | 0.331 |
| `channel-attn` | **0.5045** | 0.4764 | 0.5172 +/- 0.0562 | 0.416 |
| `lmf` | **0.5175** | 0.4583 | 0.5288 +/- 0.0310 | 0.88 |
| `gmu` | **0.5019** | 0.4741 | 0.5221 +/- 0.0497 | 0.268 |
| `cross-attn` | **0.5330** | 0.4923 | 0.5220 +/- 0.0337 | 0.0488 |
| `self-attn` **(ours)** | **0.5149** | 0.4661 | 0.5283 +/- 0.0379 | - |

`p` is a McNemar exact test of that method against ours on the same clips. Unimodal floors, late-product and rules are deterministic given the cue features, so they have no seed variance.
