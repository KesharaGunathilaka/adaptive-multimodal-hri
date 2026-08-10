"""Alternative fusion heads for the architecture ablation (Study 1,
`scripts/38_fusion_architectures.py`).

Every head here consumes the SAME input as `model.AttentionFusion` — a 24-dim
concatenated cue-probability vector (emotion|gesture|motion|context, zero-filled
where a cue is missing) plus a 4-dim observed flag — and returns 10-class
logits, so they all train through the identical `train.train_fusion` loop via
its `model_factory` argument. Missing cues are handled per-architecture (gates
zeroed / product-identity / attention-masked) rather than left as an ambiguous
all-zero sub-vector.

Roster (canonical references):
  * GMUFusion            — Gated Multimodal Unit (Arevalo et al., 2017)
  * LMFusion             — Low-rank Multimodal Fusion (Liu et al., 2018)
  * CrossAttentionFusion — learned-query cross-attention into modality tokens
                           (MulT / attention-bottleneck family, Tsai 2019 /
                           Nagrani 2021), distinct from AttentionFusion's
                           CLS self-attention
  * ChannelAttentionFusion — Squeeze-and-Excitation channel recalibration over
                           the 4 modality channels (Hu et al., 2018; CBAM
                           Woo et al., 2018), adapted from CV feature maps
"""
import torch
import torch.nn as nn

from .model import MODALITIES, MODALITY_DIMS

# byte offsets of each modality inside the 24-dim vector (emotion 0:7, gesture
# 7:15, motion 15:19, context 19:24) — derived from MODALITY_DIMS so the two
# never drift.
_OFFSETS = {}
_i = 0
for _m in MODALITIES:
    _OFFSETS[_m] = (_i, _i + MODALITY_DIMS[_m])
    _i += MODALITY_DIMS[_m]
TOTAL_DIM = _i  # 24


def _split(x):
    """[B,24] -> list of 4 per-modality sub-vectors in MODALITIES order."""
    return [x[:, a:b] for m, (a, b) in _OFFSETS.items()]


class ConcatMLPFusion(nn.Module):
    """The concat-MLP floor (`baselines.learned.ConcatMLP`) lifted into the
    `forward(x, obs)` contract so it trains through the shared loop and receives
    recombination + augmentation identically to the attention heads — the fair
    floor for Study 1. Concatenates the 24 cue probs with the 4 obs flags."""

    def __init__(self, n_classes=10, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TOTAL_DIM + len(MODALITIES), 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes))

    def forward(self, x, obs):
        return self.net(torch.cat([x, obs], dim=1))


class GMUFusion(nn.Module):
    """Gated Multimodal Unit. Each modality is projected to a hidden vector
    h_m = tanh(W_m x_m); a per-modality gate z_m = sigmoid(V_m · [all cues])
    decides how much of h_m enters the fused representation. Missing cues have
    their gate forced to 0 so they contribute nothing (obs=0)."""

    def __init__(self, d=64, n_classes=10, dropout=0.2):
        super().__init__()
        self.enc = nn.ModuleDict(
            {m: nn.Linear(dim, d) for m, dim in MODALITY_DIMS.items()})
        self.gate = nn.ModuleDict(
            {m: nn.Linear(TOTAL_DIM, d) for m in MODALITIES})
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, n_classes))

    def forward(self, x, obs):
        fused = 0.0
        for k, m in enumerate(MODALITIES):
            a, b = _OFFSETS[m]
            h = torch.tanh(self.enc[m](x[:, a:b]))          # [B,d]
            z = torch.sigmoid(self.gate[m](x))              # [B,d]
            z = z * obs[:, k:k + 1]                         # kill missing cues
            fused = fused + z * h
        return self.head(fused)


class LMFusion(nn.Module):
    """Low-rank Multimodal Fusion. Each modality vector is bias-augmented
    ([x_m, 1]) and mapped by a rank-R factor to [R, d_out]; the fused tensor is
    the elementwise product of the four factor outputs summed over the rank,
    avoiding the full outer-product's exponential blow-up. Missing modalities
    are replaced by the multiplicative identity (all-ones) so they drop out of
    the product cleanly."""

    def __init__(self, d=48, rank=4, n_classes=10, dropout=0.2):
        super().__init__()
        self.rank, self.d = rank, d
        self.factors = nn.ModuleDict(
            {m: nn.Linear(dim + 1, rank * d) for m, dim in MODALITY_DIMS.items()})
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, n_classes))

    def forward(self, x, obs):
        B = x.shape[0]
        prod = x.new_ones(B, self.rank, self.d)
        for k, m in enumerate(MODALITIES):
            a, b = _OFFSETS[m]
            xa = torch.cat([x[:, a:b], x.new_ones(B, 1)], dim=1)   # [B,dim+1]
            f = self.factors[m](xa).view(B, self.rank, self.d)     # [B,R,d]
            present = obs[:, k].view(B, 1, 1)                      # 1 present
            f = torch.where(present.bool(), f, torch.ones_like(f))
            prod = prod * f
        fused = prod.sum(dim=1)                                    # [B,d]
        return self.head(fused)


class CrossAttentionFusion(nn.Module):
    """Learned-query cross-attention. Each modality -> one token; a small set of
    learned query tokens cross-attends into the modality key/values over a few
    layers, then the pooled query drives the head. Distinct from
    AttentionFusion (which self-attends a CLS+modality set) — here the fused
    representation is a dedicated query stream reading the modalities. Missing
    cues are removed via a key-padding mask (architectural marginalization)."""

    def __init__(self, d=64, n_heads=4, n_layers=2, n_queries=1, ff=128,
                 dropout=0.2, n_classes=10):
        super().__init__()
        self.proj = nn.ModuleDict(
            {m: nn.Linear(dim, d) for m, dim in MODALITY_DIMS.items()})
        self.mod_emb = nn.Parameter(torch.randn(len(MODALITIES), d) * 0.02)
        self.query = nn.Parameter(torch.randn(1, n_queries, d) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                              batch_first=True),
                "norm1": nn.LayerNorm(d), "norm2": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, ff), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(ff, d)),
            }) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def forward(self, x, obs):
        B = x.shape[0]
        toks = torch.stack(
            [self.proj[m](x[:, a:b]) + self.mod_emb[k]
             for k, (m, (a, b)) in enumerate(_OFFSETS.items())], dim=1)  # [B,4,d]
        pad = ~obs.bool()                                    # True = ignore
        q = self.query.expand(B, -1, -1)
        for L in self.layers:
            a, _ = L["attn"](q, toks, toks, key_padding_mask=pad)
            q = L["norm1"](q + a)
            q = L["norm2"](q + L["ff"](q))
        return self.head(q.mean(dim=1))                      # pool query tokens


class ChannelAttentionFusion(nn.Module):
    """Squeeze-and-Excitation over MODALITY channels. Each modality is projected
    to a d-vector, forming a [B,4,d] map; a squeeze (mean over d) + bottleneck
    MLP + sigmoid produces one gate per modality channel that recalibrates the
    map before flatten+classify. This is the faithful CV channel-attention idea
    (recalibrate channels), with 'channel' = modality. Missing cues zero their
    gate (obs=0)."""

    def __init__(self, d=64, reduction=2, n_classes=10, dropout=0.2):
        super().__init__()
        self.proj = nn.ModuleDict(
            {m: nn.Linear(dim, d) for m, dim in MODALITY_DIMS.items()})
        c = len(MODALITIES)
        self.se = nn.Sequential(
            nn.Linear(c, max(1, c // reduction)), nn.ReLU(),
            nn.Linear(max(1, c // reduction), c))
        self.head = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(c * d), nn.Dropout(dropout),
            nn.Linear(c * d, n_classes))

    def forward(self, x, obs):
        maps = torch.stack(
            [torch.relu(self.proj[m](x[:, a:b]))
             for m, (a, b) in _OFFSETS.items()], dim=1)      # [B,4,d]
        squeeze = maps.mean(dim=2)                           # [B,4]
        gate = torch.sigmoid(self.se(squeeze)) * obs         # kill missing
        maps = maps * gate.unsqueeze(-1)                     # recalibrate
        return self.head(maps)


ZOO = {
    "gmu": GMUFusion,
    "lmf": LMFusion,
    "cross_attention": CrossAttentionFusion,
    "channel_attention": ChannelAttentionFusion,
}
