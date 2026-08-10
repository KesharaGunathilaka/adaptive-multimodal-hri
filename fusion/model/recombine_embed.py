"""GT-indexed recombination pools for embedding-level fusion (2026-08-08) --
the embedding analogue of `recombine_merged.py`, reusing its combo/label
logic (`all_combos`, `label_combo`, `allocate`) unchanged so the synthetic
label rubric is identical; only the PAYLOAD differs (a PCA-reduced
penultimate embedding vector instead of a class-probability vector).

Same design principle as `recombine_merged.build_pools`: pools are indexed by
GROUND TRUTH class, not model argmax, so a pool entry is "a real embedding a
genuine occurrence of this class produces" -- confusable cases included, not
excluded.

**Known limitation, not fixed here** (`POOL_PURITY.md`, `EMBEDDING_PROBE.md`):
pools built from TRAIN embeddings inherit train/test contamination for the
two fine-tuned-on-train cues (gesture, emotion), and the linear-probe finding
suggests this may be WORSE in embedding space than in probability space. This
module builds the straightforward in-fold version first (mirrors
`recombine_merged.build_pools` exactly); `build_pools_out_of_fold` is the
flagged follow-up if the in-fold result looks contaminated (suspiciously high
val/train, disappointing test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .recombine_merged import all_combos, label_combo

TO_RULE_MODALITY = {"emo": "emotion", "ges": "gesture", "mot": "motion", "ctx": "context"}


def build_embed_pools(embed_table: dict, clips: pd.DataFrame, train_clip_ids,
                      gt_col: dict, mask_col: dict) -> dict:
    """`embed_table`: output of `scripts/57_embedding_fusion.py::load_embed_table`
    for exactly `train_clip_ids` (same order). -> {(modality, class): [n,dim]
    PCA-space array}, modality in ('emo','ges','mot','ctx')."""
    idx = clips.set_index("clip_id")
    pools: dict[tuple[str, str], np.ndarray] = {}
    for pref in ("emo", "ges", "mot", "ctx"):
        modality = TO_RULE_MODALITY[pref]
        emb = embed_table[f"{pref}_embed"]
        obs = embed_table[f"{pref}_obs"].astype(bool)
        labels = idx.reindex(train_clip_ids)[gt_col[modality]].to_numpy()
        masked = idx.reindex(train_clip_ids)[mask_col[modality]].to_numpy()
        usable = obs & ~masked.astype(bool)
        for cls in pd.unique(labels[pd.notna(labels)]):
            sel = usable & (labels == cls)
            if sel.any():
                pools[(pref, cls)] = emb[sel]
    return pools


def generate_embed(pools: dict, embed_dims: dict, n_per_combo: int = 100,
                   seed: int = 0, balance: str = "uniform",
                   combo_filter: set | None = None):
    """-> (X [N, sum(embed_dims)], obs [N,4], y [N]). Mirrors
    `recombine_merged.generate` exactly (same combos, same labels, same
    `allocate` balancing) but draws PCA-space embedding vectors instead of
    probability vectors. `embed_dims`: {'emotion':k, 'gesture':k, ...} (PCA
    component counts) -- fixes the column layout (MODALITIES order:
    emotion, gesture, motion, context, matching `fusion.model.model.MODALITIES`)."""
    from .recombine_merged import allocate  # local import, avoids a cycle at module load
    from ..baselines import common

    rng = np.random.default_rng(seed)
    combos = all_combos()
    if combo_filter is not None:
        combos = [c for c in combos if c in combo_filter]
    alloc = allocate(combos, balance=balance, target_total=n_per_combo * len(combos))

    # embed_dims is keyed by full modality name (matches MODALITIES /
    # AttentionFusion's modality_dims contract); offsets keyed by the same
    # 'emo'/'ges'/'mot'/'ctx' prefixes `pools`/`need` use, in MODALITIES order
    # (emotion, gesture, motion, context) so the concatenated X matches
    # AttentionFusion's expected column layout.
    full_of = {"emo": "emotion", "ges": "gesture", "mot": "motion", "ctx": "context"}
    offsets, i = {}, 0
    for pref in ("emo", "ges", "mot", "ctx"):
        dim = embed_dims[full_of[pref]]
        offsets[pref] = (i, i + dim)
        i += dim
    total_dim = i

    X, Y, skipped = [], [], []
    for ctx, emo, ges, mot in combos:
        need = {"emo": emo, "ges": ges, "mot": mot, "ctx": ctx}
        missing_pool = [(m, cls) for m, cls in need.items() if (m, cls) not in pools]
        if missing_pool:
            skipped.append((ctx, emo, ges, mot, missing_pool))
            continue
        label = label_combo(ctx, emo, ges, mot)
        n = alloc[(ctx, emo, ges, mot)]
        x = np.zeros((n, total_dim), np.float32)
        for m, cls in need.items():
            pool = pools[(m, cls)]
            pick = rng.integers(0, len(pool), n)
            a, b = offsets[m]
            x[:, a:b] = pool[pick]
        X.append(x)
        Y.append(np.full(n, common.INTENTS.index(label), np.int64))

    X = np.concatenate(X) if X else np.zeros((0, total_dim), np.float32)
    y = np.concatenate(Y) if Y else np.zeros((0,), np.int64)
    obs = np.ones((len(X), 4), np.float32)
    return X, obs, y, skipped
