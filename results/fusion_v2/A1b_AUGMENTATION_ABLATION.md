# Augmentation ablation — is it a general lever or a thumb on the scale?

Every method in the baseline suite was trained *with* our cue-combination
augmentation, so a reviewer is entitled to ask whether the augmentation was tuned
to flatter our model. This runs the identical suite with augmentation switched
off. Same labels, features, seeds and test clips; 10 seeds; 773 recorded test
clips.

| method | with augmentation | without | gain |
|---|---|---|---|
| unimodal — context | 0.1268 | 0.1035 | +0.023 |
| unimodal — emotion | 0.4191 | 0.3079 | +0.111 |
| unimodal — gesture | 0.4075 | 0.4722 | −0.065 |
| late fusion (product) | 0.5653 | 0.5847 | −0.019 |
| **rules** (not trained) | 0.7646 | 0.7646 | — |
| gradient-boosted trees | 0.7050 | 0.4994 | **+0.206** |
| channel attention | 0.7853 | 0.5045 | **+0.281** |
| cross-attention | 0.8176 | 0.5330 | **+0.285** |
| GMU | 0.7969 | 0.5019 | **+0.295** |
| low-rank fusion (LMF) | 0.8137 | 0.5175 | **+0.296** |
| self-attention (ours) | 0.8202 | 0.5149 | **+0.305** |
| early fusion (concat-MLP) | 0.8008 | 0.4916 | **+0.309** |

## Reading

**The augmentation is a general lever, not a thumb on the scale.** Every learned
fusion method gains between **+0.21 and +0.31** from it, and the gain for our
model (+0.305) sits in the middle of that range rather than at the top. Early
fusion actually gains the most. There is no evidence the augmentation was tuned to
favour the proposed architecture.

**Without it, learned fusion is not viable on this dataset.** Every learned method
lands between 0.49 and 0.53, far below the hand-coded rule cascade at 0.765. Only
with combination coverage do learned methods overtake rules. This is the clearest
statement of what the data strategy contributes: on 42 recorded training
scenarios covering 39 of the 80 possible cue-combinations, a learned model simply
does not see enough of the space to compete with an explicit rubric.

**Without it, the architecture ranking is noise.** Pairwise against ours:
gbt p=0.33, channel-attn p=0.42, gmu p=0.27, lmf p=0.88 — nothing separates the
methods. The differences that appear in the main table only emerge once every
method has adequate combination coverage.

**The two unimodal exceptions are informative, not anomalies.** Gesture-only
(−0.065) and late-product (−0.019) do slightly *worse* with augmentation: both
ignore cue interactions by construction, so the extra combinations add label noise
from their point of view without adding usable structure. Only methods that model
cue interaction can exploit combination coverage.

## Consequence for the paper's claim

Combined with the taxonomy ablation (+0.067 from the label space) and the
architecture comparison (cross-attention and LMF statistically tied with ours),
the picture is consistent across three independent experiments: **the contribution
is the data and label design, not the fusion mechanism.** The architecture is
interchangeable among modern attention/low-rank variants; the taxonomy and the
combination coverage are what move the number.
