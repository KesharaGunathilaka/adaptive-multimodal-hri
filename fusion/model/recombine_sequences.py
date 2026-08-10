"""Trajectory recombination for the window-SEQUENCE representation (R3) --
Phase 3 of the 2026-08-06 "why does fusion only tie rules" investigation
(see `docs/WORKLOG.md`). `fusion/model/recombine_merged.py` synthesises
POOLED per-clip cue vectors spanning all 448 combinatorial cue tuples; R3's
window-sequence model never received an analogous augmentation (explicitly
out of scope for v1 -- `fusion/model/sequences.py`'s module docstring), which
the 2026-08-05 Study 2 identified as the likely reason R3 (~0.533) scored far
below R1 (~0.7191): not because temporal modelling doesn't help, but because
R3 trained on ~30x fewer augmented samples than R1/R2.

This closes that gap for sequences: for each of the 448 combos, stitch a
synthetic per-window TRAJECTORY from four REAL per-clip window sequences --
one per modality, each drawn from a real clip whose GROUND TRUTH matches the
combo's class for that modality (same ground-truth-indexing rationale as
`recombine_merged.build_pools` -- bucket by truth, not by what the model
already gets right) -- independently resampled to a shared target length `n`
(drawn from the real per-clip window-count distribution, so synthetic lengths
stay realistic) via `uniform_indices`, the same resampling helper used
everywhere else in this project for window selection.

This preserves each modality's OWN real temporal dynamics (an actual clip's
gesture unfolding over time) while recombining modalities across clips -- the
same "real noise, synthetic combination" principle as the pooled version,
extended along the time axis.

Every synthesised sample has obs=1 for every real (non-padded) timestep, all
four modalities (matching `recombine_merged.generate()`'s convention) --
missing-cue robustness stays `SequenceDataset`'s job via its per-timestep
dropout, not this generator's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..baselines import common
from ..extraction.windows import uniform_indices
from .recombine_merged import COL_OFFSET, LABELS, all_combos, label_combo

GT_COL = {"emo": "gt_emotion", "ges": "gt_gesture", "mot": "gt_motion", "ctx": "gt_context"}
MASK_COL = {"emo": "emotion_masked", "ges": "gesture_masked",
           "mot": "motion_masked", "ctx": "context_masked"}


@dataclass
class SeqReport:
    n_combos: int
    n_generated: int
    n_skipped_empty_pool: int
    skipped: list[tuple]
    label_counts: dict


def build_sequence_pools(windows: pd.DataFrame, clips: pd.DataFrame,
                         real_train: pd.DataFrame) -> dict:
    """(modality, class) -> list of [n_i, dim] real per-window sequences, for
    TRAIN-split clips whose GROUND TRUTH is that class. A clip contributes to
    modality m's pool only using the windows where `{m}_obs==1` -- the
    modality's own real firing pattern within that clip, which may be a
    strict subset of the clip's full window count (e.g. emotion needs a
    detected face every window)."""
    gt = clips.set_index("clip_id")
    train_ids = list(real_train.clip_id)
    w = windows[windows.clip_id.isin(set(train_ids))].sort_values(
        ["clip_id", "window_idx"])
    groups = {cid: g for cid, g in w.groupby("clip_id")}

    pools: dict[tuple[str, str], list[np.ndarray]] = {}
    for m in ("emo", "ges", "mot", "ctx"):
        cols = LABELS[m]
        for cid in train_ids:
            g = groups.get(cid)
            if g is None or cid not in gt.index:
                continue
            label = gt.at[cid, GT_COL[m]]
            masked = gt.at[cid, MASK_COL[m]]
            if masked or label is None or (isinstance(label, float) and pd.isna(label)):
                continue
            if label not in LABELS_native_set(m):
                continue
            fired = g[g[f"{m}_obs"] == 1]
            if fired.empty:
                continue
            seq = fired[cols].to_numpy(np.float32)
            pools.setdefault((m, label), []).append(seq)
    return pools


_NATIVE_CACHE: dict[str, set] = {}


def LABELS_native_set(m: str) -> set:
    if m not in _NATIVE_CACHE:
        _NATIVE_CACHE[m] = {c.split("_", 1)[1] for c in LABELS[m]}
    return _NATIVE_CACHE[m]


def generate_sequences(pools: dict, target_lens: np.ndarray, T: int,
                       n_per_combo: int = 100, seed: int = 0):
    """-> (X[N,T,24] f32, obs[N,T,4] f32, valid[N,T] bool, y[N] i64, SeqReport).
    Right-aligned like `sequences.build_sequences` (padding at the START, real
    data ending at index T-1), so the resulting arrays are drop-in compatible
    with `SequenceDataset`'s `extra` parameter and the causal model's
    "always read index -1" readout."""
    rng = np.random.default_rng(seed)
    combos = all_combos()
    Xs, OBSs, VALIDs, Ys = [], [], [], []
    skipped, counts = [], {}

    for ctx, emo, ges, mot in combos:
        need = {"emo": emo, "ges": ges, "mot": mot, "ctx": ctx}
        missing_pool = [(m, cls) for m, cls in need.items() if (m, cls) not in pools]
        if missing_pool:
            skipped.append((ctx, emo, ges, mot, missing_pool))
            continue
        label = label_combo(ctx, emo, ges, mot)
        yi = common.INTENTS.index(label)

        for _ in range(n_per_combo):
            n = int(rng.choice(target_lens))
            n = max(1, min(n, T))
            x = np.zeros((T, 24), np.float32)
            obs = np.zeros((T, 4), np.float32)
            valid = np.zeros(T, bool)
            valid[T - n:] = True
            for k, (m, cls) in enumerate(need.items()):
                pool = pools[(m, cls)]
                seq = pool[rng.integers(0, len(pool))]
                sel = seq[uniform_indices(len(seq), n)]
                a = COL_OFFSET[m]
                b = a + len(LABELS[m])
                x[T - n:, a:b] = sel
                obs[T - n:, k] = 1.0
            Xs.append(x)
            OBSs.append(obs)
            VALIDs.append(valid)
            Ys.append(yi)
        counts[label] = counts.get(label, 0) + n_per_combo

    X = np.stack(Xs) if Xs else np.zeros((0, T, 24), np.float32)
    obs = np.stack(OBSs) if OBSs else np.zeros((0, T, 4), np.float32)
    valid = np.stack(VALIDs) if VALIDs else np.zeros((0, T), bool)
    y = np.array(Ys, np.int64)
    report = SeqReport(n_combos=len(combos), n_generated=len(X),
                       n_skipped_empty_pool=len(skipped), skipped=skipped,
                       label_counts=counts)
    return X, obs, valid, y, report
