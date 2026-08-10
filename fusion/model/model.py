"""Attention fusion over modality tokens (handover §8.1).

Each modality's probability vector -> linear projection to d -> + learned
modality embedding -> 4 tokens; a learned per-modality [MISSING] token
replaces the projection whenever the cue is unobserved (runtime dropout,
scenario-designed [MISSING], or training-time modality dropout). A CLS token
prepends the sequence; 2 transformer encoder layers; CLS -> 10-class head.
~110K params — trains in seconds on the 3060, CPU-capable.
"""
import torch
import torch.nn as nn

MODALITY_DIMS = {"emotion": 7, "gesture": 8, "motion": 4, "context": 5}
MODALITIES = list(MODALITY_DIMS)


class AttentionFusion(nn.Module):
    def __init__(self, d=64, n_heads=4, n_layers=2, ff=128, dropout=0.2,
                 n_classes=10, missing_mode="token", modality_dims=None):
        """missing_mode: 'token' substitutes a learned [MISSING] embedding for
        unobserved cues; 'exclude' removes them from attention entirely via
        key-padding mask (architectural marginalization).

        `modality_dims`: optional dict overriding the module-level
        `MODALITY_DIMS` (default None -> the original 7/8/4/5-dim probability
        contract, unchanged). Added 2026-08-08 for embedding-level fusion
        (`fusion/model/embed_fusion.py`), where each "modality" is a
        PCA-reduced penultimate embedding instead of a class-probability
        vector -- the architecture (per-modality projection -> CLS attention)
        is unchanged, only the input width per token differs."""
        super().__init__()
        assert missing_mode in ("token", "exclude")
        self.missing_mode = missing_mode
        self.modality_dims = modality_dims or MODALITY_DIMS
        self.modalities = list(self.modality_dims)
        self.proj = nn.ModuleDict(
            {m: nn.Linear(dim, d) for m, dim in self.modality_dims.items()})
        self.mod_emb = nn.Parameter(torch.randn(len(self.modalities), d) * 0.02)
        self.missing = nn.Parameter(torch.randn(len(self.modalities), d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def forward(self, x, obs):
        """x: [B, sum(modality_dims)] concatenated per-modality vectors, in
        `self.modalities` order, zero-filled when missing; obs: [B, n_modalities]
        float 1/0 observed flags."""
        B = x.shape[0]
        tokens, i = [], 0
        for k, m in enumerate(self.modalities):
            dim = self.modality_dims[m]
            t = self.proj[m](x[:, i:i + dim])                      # [B, d]
            t = torch.where(obs[:, k:k + 1].bool(), t,
                            self.missing[k].expand(B, -1))
            tokens.append(t + self.mod_emb[k])
            i += dim
        n_tok = len(self.modalities) + 1
        seq = torch.cat([self.cls.expand(B, -1, -1),
                         torch.stack(tokens, dim=1)], dim=1)        # [B, n_tok, d]
        if self.missing_mode == "exclude":
            pad = torch.zeros(B, n_tok, dtype=torch.bool, device=x.device)
            pad[:, 1:] = ~obs.bool()
            return self.head(self.encoder(seq, src_key_padding_mask=pad)[:, 0])
        return self.head(self.encoder(seq)[:, 0])
