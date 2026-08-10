# Stage 6 — The fusion model

**Goal of this stage:** turn four probability vectors into one intent, in a way
that (a) beats the Stage-5 baselines, (b) survives missing cues, and (c) is small
enough to run on a Jetson.

Code: `fusion/model/` — `model.py` (architecture), `datasets.py` (augmentation),
`recombine.py` (synthetic samples), `train.py` (training loop).
Runners: `scripts/05_train_fusion.py`, `06_robustness_experiments.py`,
`07_finalize_fusion.py`.

---

## 6.1 The job, precisely

**Input:** 24 numbers + 4 flags — emotion (7), gesture (8), motion (4), context
(5), plus one `observed` flag per cue.
**Output:** one of 10 intents (F01–F10).

That is a tiny learning problem, which is deliberate. All the heavy perception
already happened in Stage 2; keeping the fusion input small means:
- it trains in ~20 seconds, so ablations are cheap;
- the four perception models stay **decoupled** — retrain any one of them and the
  fusion input shape is unchanged;
- it is trivially deployable (70K params, sub-millisecond inference).

---

## 6.2 Architecture — why tokens instead of concatenation

The naive approach (Stage 5's concat-MLP) glues the 24 numbers into one vector.
That works, but it treats the input as an undifferentiated blob: the model has to
*learn* that dimensions 0–6 are emotion and 7–14 are gesture, and it has no
natural way to say "this cue is absent."

`AttentionFusion` (`fusion/model/model.py`) instead gives each cue its own
**token**:

```
emotion[7] ─ Linear(7→64)  ─┐
gesture[8] ─ Linear(8→64)  ─┤   + learned modality embedding (which cue is this?)
motion[4]  ─ Linear(4→64)  ─┤
context[5] ─ Linear(5→64)  ─┘
                              → [CLS] + 4 cue tokens  =  sequence of 5 × 64
                              → 2 TransformerEncoder layers (4 heads, FFN 128,
                                 dropout 0.2, pre-norm)
                              → take the CLS token
                              → LayerNorm + Linear(64→10) → intent logits
```

**70,090 parameters.**

Design choices worth defending:

- **One projection per modality.** Each cue gets its *own* `Linear`, not a shared
  one, because a gesture probability and an emotion probability mean different
  things. (`nn.ModuleDict` in `__init__`.)
- **Learned modality embeddings.** Added to each token so the model always knows
  which cue it is looking at, independent of position.
- **A CLS token.** A learned "summary slot" that attends to all four cues; its
  output is what the classifier reads. Borrowed from BERT/ViT — it gives the model
  a place to accumulate a *joint* representation rather than forcing one cue to
  carry the answer.
- **Attention is the point.** Self-attention lets each cue's contribution depend
  on the others — literally *"if gesture = thumbs_down, look at emotion to decide
  F04 vs F07 vs F08."* That is the G3 mechanism expressed in architecture.
  Concatenation can approximate this; attention states it directly and, crucially,
  gives an **interpretable attention map** for the thesis.
- **Pre-norm (`norm_first=True`)** — more stable for tiny transformers.

---

## 6.3 Missing cues — and a negative result worth keeping

The model supports two mechanisms, selected by `missing_mode`:

| Mode | Mechanism |
|---|---|
| `token` | a **learned `[MISSING]` embedding** (one per modality) replaces the projection when a cue is unobserved |
| `exclude` | the cue's token is **removed from attention** via `src_key_padding_mask` — the transformer marginalises over it |

`token` was the original design (it is what the handover specified). Then the
T03 masking sweep produced an uncomfortable result:

> **The attention model with learned `[MISSING]` tokens degraded *worse* under cue
> masking than the plain concat-MLP.** With gesture masked: attention 0.634 vs
> MLP 0.878.

That is the opposite of the architecture's selling point. Two fixes were tested:

1. **Masked-validation model selection.** Early stopping on unmasked validation
   accuracy was picking epochs that happened to be robustness-poor. Selecting on
   the *mean of unmasked + each single-masked* validation accuracy
   (`select_masked=True` in `train.py`) fixed most of it.
2. **`missing_mode="exclude"`.** Marginalising instead of substituting.

After both, on `data/old` (3 seeds, test clip accuracy):

| Masked | attention (exclude) | concat-MLP (also dropout-trained) |
|---|---|---|
| none | 0.939 ± 0.000 | 0.927 ± 0.020 |
| emotion | 0.654 ± 0.029 | 0.658 ± 0.026 |
| gesture | 0.720 ± 0.060 | 0.805 ± 0.053 |
| motion | 0.809 ± 0.032 | 0.833 ± 0.015 |
| context | 0.776 ± 0.006 | 0.776 ± 0.006 |
| emotion+gesture | 0.297 ± 0.064 | 0.297 ± 0.047 |
| gesture+context | 0.752 ± 0.038 | 0.732 ± 0.030 |
| motion+context | 0.740 ± 0.006 | 0.720 ± 0.000 |

**Conclusion (a genuine thesis ablation):** at this input size, missing-cue
robustness comes from **dropout training plus marginalisation**, *not* from a
learned `[MISSING]` token. The attention model matches a dropout-trained MLP on
every masking condition — it does not beat it. The architecture earns its place
on interpretability and on clean-input accuracy, not on robustness.

The failed variant is kept in `results/fusion_v1/` deliberately: a negative
result that was measured and diagnosed is worth more than one quietly deleted.

Note also that **masking emotion costs ~28 points** — by far the largest drop,
consistent with emotion being the most reliable cue (Stage 2). And
emotion+gesture masked collapses to ~0.30, which matches the table's own ceiling
under that masking (~0.407): with only motion+context left, most intents are
genuinely unidentifiable, and the table says the safe answer is F05.

---

## 6.4 Augmentation — three kinds, train-only

Applied on the fly in `fusion/model/datasets.py`. None of these ever touch val or
test.

**1. Modality dropout** (`dropout_p`, 0.2–0.3). Each observed cue is independently
masked with probability *p*, capped at **2 dropped cues** per sample, and never
all four. This is what actually trains the missing-cue behaviour — recall from
Stage 3 that the real data is ~99.8 % complete, so without this the model would
never see a missing cue during training.

**2. Confidence jitter** (`jitter_sigma`, 0.15). With probability 0.5 per cue, add
Gaussian noise in **log-probability space** and re-normalise. This simulates a
miscalibrated or noisy detector, and it is deliberately done in log space so the
result stays a valid probability distribution.

**3. Cue recombination** (`recombine.py`). Synthesise new training samples by
drawing **real cue vectors from different clips** and combining them into tuples
the table defines but the recordings do not contain, labelled by the V3 rubric.
On `data/old` this produced **7,600 synthetic windows** covering 19 unrecorded
train rows — including every F10 row, which otherwise had *zero* supervision
(Stage 3).

Two safeguards in the implementation:
- Sources are **train-split windows only**, so no val/test cue vector can leak in.
- **V3 row #18 is deliberately skipped**: without a direction cue it is
  cue-identical to recorded row #1 but carries the opposite label, so synthesising
  it would inject pure label noise.

---

## 6.5 Training and model selection

`train.py::train_fusion` — AdamW (lr 1e-3, weight-decay 1e-4), cross-entropy with
label smoothing 0.05, batch 512, up to 80 epochs, early stopping patience 10.

The non-obvious part is **model selection**, and it turned out to matter more
than any hyper-parameter: with `select_masked=True` the score used for early
stopping is the mean of validation accuracy **unmasked and with each single cue
masked**. Selecting on unmasked accuracy alone reliably picked checkpoints that
were fragile under masking (§6.3).

---

## 6.6 Ablations on `data/old`

3 seeds, actor-disjoint test subjects, clip-level majority vote:

| Config | Augmentation | Clip acc | Clip macro-F1 |
|---|---|---|---|
| `attn_base` | none | 0.943 ± 0.011 | 0.628 ± 0.033 |
| `attn_do` | + modality dropout | 0.943 ± 0.025 | 0.629 ± 0.040 |
| `attn_do_jit` | + confidence jitter | **0.951 ± 0.017** | 0.629 ± 0.038 |
| `attn_full` | + cue recombination | 0.943 ± 0.006 | **0.649 ± 0.005** |

Readings:
- Every variant beats the concat-MLP (0.931) and all Stage-5 baselines.
- **Recombination buys macro-F1, not accuracy** — exactly as predicted, because
  its contribution is supervision for rare/unrecorded intents (F10), which
  accuracy barely notices but macro-F1 does. It also gives the **lowest seed
  variance** (±0.006), i.e. the most stable model.
- Augmentation does not raise clean accuracy much. Its value is robustness and
  rare-class coverage, which is what it was for.

**Deployed model** (`scripts/07_finalize_fusion.py` →
`jetson_deploy/fusion/fusion_attn.pt`): `missing_mode="exclude"`, dropout 0.3,
jitter 0.15, recombination, masked-val selection. Test clip accuracy **0.939 on
all three seeds** — unusually stable, which is what you want in a deployed
artifact.

---

## 6.7 Honest status on `data/final`

Everything in §6.6 is measured on `data/old`, where the test split holds out
**people**. On `data/final`, where it holds out **cue combinations**, the same
architecture scores **0.325** (Stage 5 §5.3), and the oracle diagnostic shows
**0.500 even with perfect cues**.

So the architecture is **not** currently validated on the hard question. What
Stage 6 established is:
- the model works well when the cue tuple is familiar and the cues are noisy;
- its missing-cue handling is sound and matches its ceiling under masking;
- its weakness is **compositional generalisation**, and that weakness is a
  property of *what it was trained on* (~19 tuples), not of the architecture.

The lever is §6.4's third augmentation. Recombination on `data/old` was scoped to
*unrecorded rows*; the finding in `GAP_DECOMPOSITION.md` says it must instead be
scoped to **spanning the combinatorial cue space** — supplying the rubric
semantics the model cannot induce, while training on *real noisy cue vectors* so
it keeps the robustness rules lack. That is the next experiment, and §6.6's table
should be regenerated on `data/final` once it exists.

---

## 6.8 What Stage 6 produced

```
fusion/model/
├── model.py       AttentionFusion (70,090 params, token|exclude missing modes)
├── datasets.py    modality dropout + confidence jitter (train-only)
├── recombine.py   cue recombination from real cue vectors
└── train.py       training loop + masked-val selection + masked evaluation

results/fusion_v1/   ablation table, robustness sweep, deployed checkpoint
jetson_deploy/fusion/fusion_attn.pt   the deployed model
```

**In one sentence:** a 70K-parameter attention network over four cue tokens that
beats every baseline on familiar situations, handles missing cues at its
theoretical ceiling, and — as measured honestly on unseen situations — still
needs the rubric supplied to it through augmentation.

---

### Open points you might want to change
- **Regenerate §6.6 on `data/final`** with the 4 s clip-level aggregation; the
  ablation ordering may change on the harder split.
- **Attention-weight figure**: the interpretability argument for choosing
  attention over an MLP is currently unexercised. A per-row attention map (which
  cue the CLS token attends to for F04 vs F07) is cheap and would be a strong
  thesis figure.
- **`missing_mode="token"` is retained** only as the ablation evidence; the
  deployed path is `exclude`. Do not switch without re-running the sweep.

---

## 6.9 Architecture zoo + temporal representation, on `data/final_merged` (2026-08-05)

§6.7's "the lever is recombination, not architecture" prediction was tested directly once
`data/final_merged` + the promoted unimodal checkpoints + recombination (§6.4's third
augmentation, finally scoped correctly per §6.7) were all in place. Full results/code:
`results/realworld_eval_merged/FUSION_ARCHITECTURES.md`,
`fusion/model/fusion_zoo.py` (GMU, LMF, cross-attention, channel-attention/CAM), `fusion/model/gbt.py`.

**Architecture roster, headline clip-acc, `full` recipe, 3 seeds:**

| Model | Params | Clip acc |
|---|---|---|
| Rule-based | — | 0.712 |
| Self-attention (incumbent, unchanged from §6.2) | 70,090 | **0.7191** |
| GMU (gated) | 8,970 | 0.7065 (std 0.0024 — far tighter than the incumbent's 0.0164) |
| LMF (low-rank tensor) | 6,730 | 0.6994 |
| Cross-attention | 69,834 | 0.699 |
| GBT (LightGBM) | — | 0.684 |
| Concat-MLP (floor) | 12,618 | 0.6837 |
| Channel-attention (CAM) | 4,896 | 0.6581 |

**Prediction confirmed**: architectures cluster within ~0.06 of each other once recombination is
applied — capacity was never the bottleneck, matching §6.7's diagnosis exactly. The incumbent
stays deployed; GMU is the noted fallback for a Jetson params/latency squeeze (8x smaller, far
more seed-stable, ~0.01 less accurate).

**Temporal representation** (`results/realworld_eval_merged/TEMPORAL_REPRESENTATION.md`): the
clip-pooled input (R1, i.e. everything in §6.2-6.8) still wins over per-window training (R2,
0.7109) and an order-aware sequence transformer (R3, ~0.533) — but R3's number is **confounded**,
not a fair test: it has no recombination analogue (synthetic per-clip cue vectors don't extend to
synthetic *trajectories*), so it trains on ~30x fewer augmented samples than R1/R2. Not a verdict
against temporal modeling; a flagged follow-up (build recombination-for-sequences first).

**Lookback span** (`results/realworld_eval_merged/WINDOW_SIZE_SWEEP.md`): re-running §7.9's old
sweep on the current pipeline confirms shorter spans still win (x0.5 = 0.7252 > deployed x1.0's
0.7191), with a much gentler slope than the old table thanks to recombination.

---

## 6.10 ⚠ Statistical power — read this before quoting any table above (2026-08-08)

Every comparison in §6.9, and the headline in §6.6, was measured at **3 seeds**. Measuring the
per-seed spread properly (`scripts/49_significance.py`, 10 seeds of the deployed `full` recipe)
gives a standard deviation of **0.0160** on headline clip accuracy. That fixes how large a
difference 3 seeds can actually resolve:

| Seeds | 95% CI half-width on the mean | Smallest resolvable difference |
|---|---|---|
| 3 | ±0.0397 | ~0.079 |
| 5 | ±0.0199 | ~0.040 |
| 10 | ±0.0114 | ~0.023 |
| 20 | ±0.0075 | ~0.015 |

Against that threshold, the differences the tables above are read from:

| Claimed comparison | Difference | Resolvable at 3 seeds? |
|---|---|---|
| self-attention vs GMU | 0.0126 | no |
| self-attention vs cross-attention | 0.0201 | no |
| self-attention vs concat-MLP floor | 0.0354 | no |
| window x0.5 vs x1.0 | 0.0061 | no |
| R1 clip-pool vs R2 per-window | 0.0082 | no |
| fusion vs rules | 0.0071 | no |

**None of them are.** The architecture roster's entire spread, best (0.7191) to worst (0.6581),
is 0.061 — still below the ~0.079 that 3 independent seeds can distinguish. This does not make
the conclusions wrong, but it does mean **they are not established by the evidence as collected**:
"self-attention is the best head", "shorter spans are better", and "clip-pool beats per-window"
are all currently indistinguishable from seed noise.

Two things follow, and both are cheap:
1. **Compare configs paired by seed**, not as independent means — train every config on the same
   seed set and test the per-seed *differences*. Common seed effects cancel, so this resolves far
   smaller gaps at the same cost. It could not be applied retroactively here because the Study
   1–3 JSONs saved only aggregated mean/std; **per-seed values should be persisted from now on**
   (`scripts/49_significance.py` does).
2. **Raise the seed count** for any comparison that is going into the thesis as a claim.

The headline itself has already been re-measured this way: §6.6's 0.7191 was a 3-seed artifact
(seeds 0,1,2 happen to include seed 0, the best of ten), and the honest figure is
**0.7133 ± 0.0160**, statistically indistinguishable from rules — see
`results/realworld_eval_merged/SIGNIFICANCE.md`.
