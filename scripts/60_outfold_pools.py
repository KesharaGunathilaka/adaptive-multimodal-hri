"""Build out-of-fold GT-indexed embedding pools for gesture + emotion
(2026-08-08) -- consumes the two fold checkpoints from
`scripts/59_outfold_finetune.py`, scores each fold's HELD-OUT train clips
with the checkpoint that never saw them, and combines the two halves into a
full "clean" train-embedding table for exactly the two contaminated cues
(`POOL_PURITY.md`: gesture, emotion). Motion/context are copied UNCHANGED
from the existing in-fold `embeddings_pooled.npz` -- they were never
fine-tuned on train and are not contaminated (motion's train/test purity gap
was already negative).

Gesture: `WindowFeaturizer(gesture_ckpt=fold_ckpt)` on the ALREADY-CACHED
per-frame arrays -- no video decode, hooked embedding is free (same forward
pass as `ges_probs`, see `windows.py`'s 2026-08-08 hook).

Emotion: the ALREADY-CACHED face crops (`data/final_merged/features/
emotion_crops/`) run through the fold checkpoint with the same
features->avgpool->flatten->classifier split `perframe.py::_emotion_batch`
uses, then mean-pooled per clip -- matching the existing pooling convention
exactly, just with a fold-specific, held-out-clip-only checkpoint.

    .venv/Scripts/python scripts/60_outfold_pools.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.extraction.modloader import load_module  # noqa: E402
from fusion.extraction.windows import WindowFeaturizer  # noqa: E402
from scripts.realworld_eval.merged_unimodal import (PERFRAME_DIR,  # noqa: E402
                                                     load_clips)

GES_DIR = ROOT / "modalities" / "gesture"
EMO_DIR = ROOT / "modalities" / "emotion"
CROPS_DIR = ROOT / "data" / "final_merged" / "features" / "emotion_crops"
FOLD_SPLIT_PATH = ROOT / "data" / "final_merged" / "features" / "outfold_split.json"
EXISTING_EMBED = ROOT / "data" / "final_merged" / "features" / "embeddings_pooled.npz"
OUT_PATH = ROOT / "data" / "final_merged" / "features" / "embeddings_pooled_outfold.npz"


def load_folds() -> dict[int, set]:
    d = json.loads(FOLD_SPLIT_PATH.read_text())
    return {int(k): set(v) for k, v in d.items()}


# ── gesture: out-of-fold clip-level embeddings ──────────────────────────────
def gesture_outfold(clips, fold_held_out_clips, fold_ckpt: Path, device) -> dict:
    """-> {clip_id: (embed[128], probs[8])} for exactly `fold_held_out_clips`,
    scored with `fold_ckpt` (never trained on them)."""
    fz = WindowFeaturizer(device=device, gesture_ckpt=fold_ckpt)
    out = {}
    for cid in fold_held_out_clips:
        p = PERFRAME_DIR / f"{cid}.npz"
        if not p.exists():
            continue
        npz = np.load(p)
        rows = fz.featurize_clip(npz)          # no embed_npz needed -- ges_embed is hooked
        embeds = [r["ges_embed"] for r in rows if r["ges_embed"] is not None]
        probs = [r["ges_probs"] for r in rows if r["ges_probs"] is not None]
        if embeds:
            out[cid] = (np.mean(embeds, axis=0), np.mean(probs, axis=0))
    return out


# ── emotion: out-of-fold clip-level embeddings ──────────────────────────────
@torch.no_grad()
def emotion_outfold(clips, fold_held_out_clips, fold_ckpt: Path, device) -> dict:
    emo = load_module("hri_emo_outfold", EMO_DIR / "inference" / "video.py")
    model = emo.build_model().to(device)
    model.load_state_dict(torch.load(fold_ckpt, map_location=device, weights_only=True))
    model.eval()
    tfm = emo.get_transform()

    out = {}
    for cid in fold_held_out_clips:
        p = CROPS_DIR / f"{cid}.npz"
        if not p.exists():
            continue
        crops = np.load(p)["crops"]
        if len(crops) == 0:
            continue
        batch = torch.stack([tfm(Image.fromarray(c)) for c in crops]).to(device)
        feat = model.features(batch)
        embed = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(feat, 1), 1)
        probs = torch.softmax(model.classifier(embed), dim=1)
        out[cid] = (embed.mean(dim=0).cpu().numpy(), probs.mean(dim=0).cpu().numpy())
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clips = load_clips()
    clips = clips[clips.v3_row.notna()]
    folds = load_folds()
    print(f"folds: {[len(folds[f]) for f in (0, 1)]} scenarios", flush=True)

    ges_results, emo_results = {}, {}
    for fold in (0, 1):
        held_out_rows = folds[fold]          # THIS fold's checkpoint never saw these
        held_out_clips = clips[(clips.split == "train")
                               & clips.v3_row.isin(held_out_rows)].clip_id.tolist()
        print(f"\n[fold{fold}] {len(held_out_clips)} held-out clips", flush=True)

        t0 = time.time()
        ges_ckpt = GES_DIR / "checkpoints" / f"best_TCN_fold{fold}.pth"
        r = gesture_outfold(clips, held_out_clips, ges_ckpt, device)
        ges_results.update(r)
        print(f"  gesture: {len(r)}/{len(held_out_clips)} scored [{time.time()-t0:.0f}s]",
              flush=True)

        t0 = time.time()
        emo_ckpt = EMO_DIR / "checkpoints" / f"finetuned_MobileNetV2_fold{fold}.pth"
        r = emotion_outfold(clips, held_out_clips, emo_ckpt, device)
        emo_results.update(r)
        print(f"  emotion: {len(r)}/{len(held_out_clips)} scored [{time.time()-t0:.0f}s]",
              flush=True)

    # ── assemble the full table (TRAIN clips only, matching build_embed_pools'
    # expected shape), gesture/emotion out-of-fold, motion/context copied ────
    existing = np.load(EXISTING_EMBED, allow_pickle=True)
    ex_idx = {cid: i for i, cid in enumerate(existing["clip_id"].astype(str))}
    train_clip_ids = clips[clips.split == "train"].clip_id.tolist()

    ges_dim, emo_dim = existing["ges_embed"].shape[1], existing["emo_embed"].shape[1]
    out = {"clip_id": np.array(train_clip_ids, dtype=object)}
    ges_embed = np.zeros((len(train_clip_ids), ges_dim), np.float32)
    ges_obs = np.zeros(len(train_clip_ids), np.float32)
    emo_embed = np.zeros((len(train_clip_ids), emo_dim), np.float32)
    emo_obs = np.zeros(len(train_clip_ids), np.float32)
    n_ges_covered = n_emo_covered = 0
    for i, cid in enumerate(train_clip_ids):
        if cid in ges_results:
            ges_embed[i], _ = ges_results[cid]
            ges_obs[i] = 1.0
            n_ges_covered += 1
        if cid in emo_results:
            emo_embed[i], _ = emo_results[cid]
            emo_obs[i] = 1.0
            n_emo_covered += 1
    out["ges_embed"], out["ges_obs"] = ges_embed, ges_obs
    out["emo_embed"], out["emo_obs"] = emo_embed, emo_obs

    # motion/context: copy the EXISTING (in-fold, but never contaminated) values
    for pref in ("mot", "ctx"):
        dim = existing[f"{pref}_embed"].shape[1]
        emb = np.zeros((len(train_clip_ids), dim), np.float32)
        obs = np.zeros(len(train_clip_ids), np.float32)
        pos = np.array([ex_idx.get(cid, -1) for cid in train_clip_ids])
        have = pos >= 0
        emb[have] = existing[f"{pref}_embed"][pos[have]]
        obs[have] = existing[f"{pref}_obs"][pos[have]]
        out[f"{pref}_embed"], out[f"{pref}_obs"] = emb, obs

    np.savez_compressed(OUT_PATH, **out)
    print(f"\n-> {OUT_PATH.relative_to(ROOT)}")
    print(f"gesture coverage: {n_ges_covered}/{len(train_clip_ids)} "
          f"({n_ges_covered/len(train_clip_ids):.4f})")
    print(f"emotion coverage: {n_emo_covered}/{len(train_clip_ids)} "
          f"({n_emo_covered/len(train_clip_ids):.4f})")


if __name__ == "__main__":
    main()
