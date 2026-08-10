"""Shared constants/helpers for embedding-level fusion scripts (57, 58) --
factored out so both use the identical clip-to-embedding join logic rather
than two copies drifting apart.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fusion.model.model import MODALITIES

ROOT = Path(__file__).resolve().parents[1]
EMBED_PATH = ROOT / "data" / "final_merged" / "features" / "embeddings_pooled.npz"
PREF_OF = {"emotion": "emo", "gesture": "ges", "motion": "mot", "context": "ctx"}
PROB_DIMS = {"emotion": 7, "gesture": 8, "motion": 4, "context": 5}


def load_embed_table(clip_ids: np.ndarray) -> dict:
    """embeddings_pooled.npz, reindexed to `clip_ids` order (missing clips
    get zero embeddings + obs=0, matching the standard missing-cue contract)."""
    z = np.load(EMBED_PATH, allow_pickle=True)
    idx = {cid: i for i, cid in enumerate(z["clip_id"].astype(str))}
    pos = np.array([idx.get(cid, -1) for cid in clip_ids])
    have = pos >= 0
    out = {}
    for m in MODALITIES:
        pref = PREF_OF[m]
        dim = z[f"{pref}_embed"].shape[1]
        emb = np.zeros((len(clip_ids), dim), np.float32)
        obs = np.zeros(len(clip_ids), np.float32)
        emb[have] = z[f"{pref}_embed"][pos[have]]
        obs[have] = z[f"{pref}_obs"][pos[have]]
        out[f"{pref}_embed"] = emb
        out[f"{pref}_obs"] = obs
    out["has_embed"] = have
    return out
