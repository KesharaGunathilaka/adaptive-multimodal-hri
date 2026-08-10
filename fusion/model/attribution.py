"""Cue attribution for `AttentionFusion` decisions -- the CAM-analogue
(2026-08-07, following the user's question about adding Class Activation
Mapping to the fusion model zoo).

**Literal CAM/Grad-CAM does not apply here.** Both produce a SPATIAL heatmap
over a CNN's feature maps ("which pixels drove this class"); the fusion input
is a 24-dim probability vector with no spatial extent to map. Applying CAM
literally would not be meaningful. (Separately: the "CAM" head in
`fusion/model/fusion_zoo.py::ChannelAttentionFusion` is Channel Attention,
i.e. SE/CBAM-style channel recalibration -- unrelated to Class Activation
Mapping despite the shared acronym; worth renaming in the thesis to avoid
exactly this confusion.)

**The concept transfers cleanly, though, and needs no external attribution
method.** In a CNN, the class score is a nonlinear function of spatial
features, so Grad-CAM/SHAP APPROXIMATE which input regions mattered. Here,
the CLS token's attention weight onto each modality token IS ALREADY the
model's own learned mixing coefficient -- no approximation needed, just
extraction. This is the "attention-weight figure... currently unexercised...
would be a strong thesis figure" gap flagged in `06_fusion_model.md`'s open
points.

Implementation note: `nn.TransformerEncoderLayer.forward` hardcodes
`need_weights=False` internally on its fast path, so attention weights are
NOT retrievable via a forward hook on the layer itself. This module manually
replicates the `norm_first` layer forward (the `_sa_block`/`_ff_block`
pattern PyTorch's implementation uses) to call the inner `self_attn` module
directly with `need_weights=True`.
"""
from __future__ import annotations

import torch

from .model import MODALITIES, MODALITY_DIMS

TOKEN_NAMES = ["CLS"] + list(MODALITIES)


@torch.no_grad()
def cls_attention(model, x: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """model: a trained `AttentionFusion`. x:[B,24] obs:[B,4].

    -> [B, n_layers, 5] float -- the CLS token's (row 0) attention weight onto
    each of the 5 tokens (self, emotion, gesture, motion, context) at every
    encoder layer, head-averaged. The LAST layer's weights (`[:, -1, 1:]`) are
    "how much did the final decision attend to each modality" -- the headline
    attribution figure.
    """
    device = x.device
    B = x.shape[0]
    tokens, i = [], 0
    for k, m in enumerate(MODALITIES):
        dim = MODALITY_DIMS[m]
        t = model.proj[m](x[:, i:i + dim])
        t = torch.where(obs[:, k:k + 1].bool(), t, model.missing[k].expand(B, -1))
        tokens.append(t + model.mod_emb[k])
        i += dim
    seq = torch.cat([model.cls.expand(B, -1, -1), torch.stack(tokens, dim=1)], dim=1)

    pad = None
    if model.missing_mode == "exclude":
        pad = torch.zeros(B, 5, dtype=torch.bool, device=device)
        pad[:, 1:] = ~obs.bool()

    weights_per_layer = []
    h = seq
    for layer in model.encoder.layers:
        normed = layer.norm1(h)
        attn_out, attn_w = layer.self_attn(
            normed, normed, normed, attn_mask=None, key_padding_mask=pad,
            need_weights=True, average_attn_weights=True)
        weights_per_layer.append(attn_w[:, 0, :].cpu())        # CLS row, [B,5]
        h = h + layer.dropout1(attn_out)
        h2 = layer.norm2(h)
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h2))))
        h = h + layer.dropout2(ff)
    return torch.stack(weights_per_layer, dim=1)                # [B, n_layers, 5]


def batched_cls_attention(model, X, obs, device, bs=2048) -> torch.Tensor:
    """X:[N,24] obs:[N,4] numpy -> [N, n_layers, 5] torch, batched."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        ob = torch.from_numpy(obs[i:i + bs]).to(device)
        out.append(cls_attention(model, xb, ob))
    return torch.cat(out, dim=0)
