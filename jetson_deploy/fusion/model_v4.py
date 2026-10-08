"""Attention fusion architecture (v4) — SELF-CONTAINED COPY for the Jetson
package.

Source of truth: ../../fusion/model/model.py (AttentionFusion, modality_dims-
parameterised version) in the main repo. If you change the architecture there,
re-copy this file AND re-export the ONNX (scripts/B1_export_jetson_v4.py), or
the checkpoint will load into a mismatched graph.

v4 differs from the retired model.py (still present for reference) in three
ways, all decided by measuring the four cue models on held-out footage
(docs/FUSION_V4_METHOD_GUIDE.md):
  * MOTION IS REMOVED. Only 3 tokens: emotion, gesture, context.
  * emotion is 5-dim SUPERCLASS probabilities (alarm/hostile/positive/low/
    neutral), not the 7-dim fine-grained RAF-DB softmax.
  * context is 2-dim (classroom/kitchen only) — the fusion head has no
    representation for other scenes; mark obs[2]=0 for any other environment.
"""
import torch
import torch.nn as nn

MODALITY_DIMS = {"emotion": 5, "gesture": 8, "context": 2}
MODALITIES = list(MODALITY_DIMS)


class AttentionFusion(nn.Module):
    def __init__(self, d=64, n_heads=4, n_layers=2, ff=128, dropout=0.2,
                 n_classes=9, missing_mode="exclude"):
        super().__init__()
        assert missing_mode in ("token", "exclude")
        self.missing_mode = missing_mode
        self.proj = nn.ModuleDict(
            {m: nn.Linear(dim, d) for m, dim in MODALITY_DIMS.items()})
        self.mod_emb = nn.Parameter(torch.randn(len(MODALITIES), d) * 0.02)
        self.missing = nn.Parameter(torch.randn(len(MODALITIES), d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def forward(self, x, obs):
        """x: [B,15] = emotion_super(5) | gesture(8) | context(2), concatenated
        in that order. obs: [B,3] float 1/0, same order."""
        B = x.shape[0]
        tokens, i = [], 0
        for k, m in enumerate(MODALITIES):
            dim = MODALITY_DIMS[m]
            t = self.proj[m](x[:, i:i + dim])
            t = torch.where(obs[:, k:k + 1].bool(), t,
                            self.missing[k].expand(B, -1))
            tokens.append(t + self.mod_emb[k])
            i += dim
        n_tok = len(MODALITIES) + 1                    # +1 for CLS
        seq = torch.cat([self.cls.expand(B, -1, -1),
                         torch.stack(tokens, dim=1)], dim=1)
        if self.missing_mode == "exclude":
            pad = torch.zeros(B, n_tok, dtype=torch.bool, device=x.device)
            pad[:, 1:] = ~obs.bool()
            return self.head(self.encoder(seq, src_key_padding_mask=pad)[:, 0])
        return self.head(self.encoder(seq)[:, 0])
