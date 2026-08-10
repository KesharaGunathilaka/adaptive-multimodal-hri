"""Pass 2: per-frame .npz caches -> window-level probability vectors.

Time-based grid (clips are mixed 15/24/30 fps — see DECISIONS.md 2026-07-16):
    stride           8/30 s  (deployment stride S=8 @ 30 fps ≈ 3.75 Hz)
    gesture lookback 2.133 s -> uniform-resampled to 32 frames (matches
                     GestureEngine's ENGINE_BUFFER_FRAMES=64 @30fps -> WINDOW=32)
    motion  lookback 2.0 s   -> resampled to 30 frames, dt≈1/15 s (matches
                     MotionInference's 30-frame window on 15 fps clips)
    emotion lookback 8/30 s  -> mean softmax of face frames in span
                     (fallback: widen to the gesture span before flagging missing)
    context lookback 1.0 s   -> mean of raw CLIP samples in span

Per cue we emit probabilities + a coverage ratio; a cue with no usable frames
in its span gets NaN probs and observed=0 (runtime-missing, distinct from the
scenario-designed [MISSING] flag which lives in labels.csv).
"""
import json

import numpy as np
import torch

from .modloader import REPO, load_module

GES_DIR = REPO / "modalities" / "gesture"
MOT_DIR = REPO / "modalities" / "motion"

GESTURE_CKPT = GES_DIR / "checkpoints" / "best_TCN.pth"
GESTURE_CFG = GES_DIR / "checkpoints" / "model_config.json"
MOTION_CKPT = MOT_DIR / "checkpoints" / "best_model_finetuned.pt"

STRIDE_SEC = 8 / 30
GES_SPAN = 64 / 30          # 2.133 s
MOT_SPAN = 2.0
EMO_SPAN = 8 / 30
CTX_SPAN = 1.0
GES_MIN_FRAMES = 8          # min valid pose frames in span to run the TCN
MOT_MIN_FRAMES = 15


def uniform_indices(n, target):
    return np.round(np.linspace(0, n - 1, target)).astype(np.int64)


class WindowFeaturizer:
    def __init__(self, device=None, scale=1.0, gesture_ckpt=None, motion_ckpt=None):
        """`scale` multiplies the gesture/motion lookback spans (window-size
        sweep, handover §8.3). Emotion/context spans and the stride stay fixed
        — the sweep varies temporal context, not the output grid.

        `gesture_ckpt`/`motion_ckpt` override the deployed checkpoint (default
        GESTURE_CKPT/MOTION_CKPT) — used to refresh the feature cache after a
        fine-tune without touching the deployed weights until they're compared.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ges_span = GES_SPAN * scale
        self.mot_span = MOT_SPAN * scale
        gesture_ckpt = gesture_ckpt or GESTURE_CKPT
        motion_ckpt = motion_ckpt or MOTION_CKPT

        # ── Gesture TCN (arch/labels pinned by model_config.json) ─────────
        gm = load_module("hri_gesture_models", GES_DIR / "src" / "models.py", [GES_DIR])
        cfg = json.loads(GESTURE_CFG.read_text())
        self.gesture_labels = list(cfg["labels"])
        self.ges_window = int(cfg["window"])
        self.ges_model = gm.build_model(cfg["model"], **cfg.get("model_kwargs", {}))
        self.ges_model.load_state_dict(
            torch.load(gesture_ckpt, map_location=self.device, weights_only=True))
        self.ges_model.to(self.device).eval()

        # ── Motion LSTM (same load path as MotionInference) ───────────────
        mi = load_module("hri_motion_inf", MOT_DIR / "src" / "inference.py",
                         [MOT_DIR / "src"])
        self._mi = mi
        self.motion_labels = [mi.MOTION_LABELS[i] for i in range(mi.NUM_CLASSES)]
        self.mot_window = mi.WINDOW_SIZE
        ckpt = torch.load(motion_ckpt, map_location=self.device, weights_only=True)
        mcfg = ckpt.get("config", {})
        self.mot_model = mi.MotionLSTM(
            hidden_size=mcfg.get("hidden_size", 256),
            num_layers=mcfg.get("num_layers", 3),
            dropout=mcfg.get("dropout", 0.35)).to(self.device)
        self.mot_model.load_state_dict(ckpt["model_state_dict"])
        self.mot_model.eval()

        # ── Penultimate-embedding capture (2026-08-08, embedding-level fusion) ─
        # Forward-pre-hooks on each model's final classifier layer capture its
        # INPUT -- the pooled feature the model itself already computes before
        # collapsing to class logits -- at zero extra compute (same forward
        # pass already running for ges_probs/mot_probs). TCN's `head` is
        # Sequential(Dropout, Linear): in eval() mode Dropout is identity, so
        # hooking the Sequential's input equals hooking the Linear's input.
        # LSTM's `classifier` is Sequential(Linear, ReLU, Dropout, Linear); its
        # input is the 256-d attention-pooled context vector (`model.py`
        # forward: `context = (attn_weights * lstm_out).sum(dim=1)`).
        self._ges_embed_buf = self._mot_embed_buf = None
        self.ges_model.head.register_forward_pre_hook(
            lambda m, args: setattr(self, "_ges_embed_buf", args[0].detach()))
        self.mot_model.classifier.register_forward_pre_hook(
            lambda m, args: setattr(self, "_mot_embed_buf", args[0].detach()))

    # ────────────────────────────────────────────────────────────────────
    def _motion_window_feats(self, joints):
        """[30,25,3] resampled joints -> [30,84] pos+vel (MotionInference recipe)."""
        pos = np.stack([
            self._mi.normalize_skeleton(j[self._mi.JOINT_SUBSET]).flatten()
            for j in joints])                                   # [30,42]
        vel = np.vstack([np.zeros((1, 42), np.float32), np.diff(pos, axis=0)])
        return np.concatenate([pos, vel], axis=1).astype(np.float32)

    def raw_training_windows(self, npz, which):
        """Per-frame cache -> list of RAW (not model-run) window features, for
        fine-tuning. `which` = 'motion' -> [30,84] pos+vel; 'gesture' ->
        [32,185] keypoint features. Uses the IDENTICAL window-selection logic
        as `featurize_clip` (same spans, same min-frame gate, same uniform
        resampling) so a fine-tuned model trains on exactly the distribution
        it will see at inference — train/serve skew here would silently cost
        accuracy with no error message (see fusion/extraction/windows.py
        module docstring's general warning on this)."""
        fps = float(npz["fps"])
        T = int(npz["n_frames"])
        dur = T / fps
        times = np.arange(T) / fps
        pose_valid = npz["pose_valid"]
        span = self.mot_span if which == "motion" else self.ges_span
        min_frames = MOT_MIN_FRAMES if which == "motion" else GES_MIN_FRAMES
        window = self.mot_window if which == "motion" else self.ges_window

        first = max(self.ges_span, self.mot_span)
        ends = np.arange(first, dur + 1e-9, STRIDE_SEC)
        if len(ends) == 0:
            ends = np.array([dur])

        out = []
        for t_end in ends:
            m = (times > t_end - span) & (times <= t_end) & pose_valid
            idx = np.flatnonzero(m)
            if len(idx) < min_frames:
                continue
            sel = idx[uniform_indices(len(idx), window)]
            if which == "motion":
                out.append(self._motion_window_feats(npz["joints25"][sel]))
            else:
                out.append(npz["gesture_feats"][sel])
        return out

    @torch.no_grad()
    def featurize_clip(self, npz, embed_npz=None):
        """Per-frame cache dict -> list of per-window dicts.

        `embed_npz`: optional second npz (`scripts/54_extract_embeddings.py`'s
        output -- `emotion_embed [T,1280]`, `context_embed [Tc,512]`) for the
        SAME clip. When given, each row also gets `emo_embed`/`ctx_embed`,
        pooled over the identical span/coverage windows as `emo_probs`/
        `ctx_probs` (same `nanmean`-over-span logic, just a wider vector) --
        guarantees the embedding columns line up window-for-window with the
        existing probability columns, since both come from one pass over the
        same `ends` sequence. `ges_embed`/`mot_embed` (128-d/256-d) are always
        produced regardless of `embed_npz`, at zero extra cost via the
        forward-pre-hooks registered in `__init__` -- gesture/motion need no
        separate frame-level embedding cache since the model already consumes
        a whole window per call."""
        fps = float(npz["fps"])
        T = int(npz["n_frames"])
        dur = T / fps
        times = np.arange(T) / fps
        pose_valid = npz["pose_valid"]
        face_valid = npz["face_valid"]
        ctx_t = npz["context_frames"] / fps
        emo_embed_frames = embed_npz["emotion_embed"] if embed_npz is not None else None
        ctx_embed_frames = embed_npz["context_embed"] if embed_npz is not None else None

        first = max(self.ges_span, self.mot_span)
        ends = np.arange(first, dur + 1e-9, STRIDE_SEC)
        if len(ends) == 0:                      # clip shorter than the lookback
            ends = np.array([dur])

        rows, ges_batch, mot_batch = [], [], []
        for wi, t_end in enumerate(ends):
            row = {"window_idx": wi, "t_end": round(float(t_end), 3)}

            # gesture
            m = (times > t_end - self.ges_span) & (times <= t_end) & pose_valid
            idx = np.flatnonzero(m)
            row["ges_cov"] = len(idx) / max(1, round(self.ges_span * fps))
            if len(idx) >= GES_MIN_FRAMES:
                sel = idx[uniform_indices(len(idx), self.ges_window)]
                ges_batch.append((wi, npz["gesture_feats"][sel]))

            # motion
            m = (times > t_end - self.mot_span) & (times <= t_end) & pose_valid
            idx = np.flatnonzero(m)
            row["mot_cov"] = len(idx) / max(1, round(self.mot_span * fps))
            if len(idx) >= MOT_MIN_FRAMES:
                sel = idx[uniform_indices(len(idx), self.mot_window)]
                mot_batch.append((wi, self._motion_window_feats(npz["joints25"][sel])))

            # emotion (widen to gesture span if the short span has no face)
            for span in (EMO_SPAN, self.ges_span):
                m = (times > t_end - span) & (times <= t_end) & face_valid
                if m.any():
                    row["emo_probs"] = np.nanmean(npz["emotion_probs"][m], axis=0)
                    row["emo_cov"] = m.sum() / max(1, round(span * fps))
                    # `emo_embed_frames` comes from a SEPARATE detection pass
                    # (`extract_embeddings_only`, not `extract_clip`) -- do not
                    # reuse `m` (built from `npz`'s own `face_valid`) blindly:
                    # on rare frames the two passes disagree about which frame
                    # has a face, and if ALL frames `m` selects are NaN in
                    # `emo_embed_frames`, `nanmean` silently returns NaN for the
                    # whole window instead of raising, which then poisons the
                    # clip-level mean in Pass 2 (found 2026-08-08 in 2/2869
                    # clips via a downstream NaN audit). Require the embedding
                    # itself be present at each selected frame.
                    if emo_embed_frames is not None:
                        em = m & ~np.isnan(emo_embed_frames).any(axis=1)
                        row["emo_embed"] = (np.nanmean(emo_embed_frames[em], axis=0)
                                            if em.any() else None)
                    else:
                        row["emo_embed"] = None
                    break
            else:
                row["emo_probs"], row["emo_cov"], row["emo_embed"] = None, 0.0, None

            # context
            m = (ctx_t > t_end - CTX_SPAN) & (ctx_t <= t_end)
            if not m.any():
                m = ctx_t <= t_end                       # fall back to any earlier sample
            row["ctx_probs"] = (npz["context_probs"][m].mean(axis=0)
                                if m.any() else None)
            row["ctx_cov"] = float(m.sum())
            row["ctx_embed"] = (ctx_embed_frames[m].mean(axis=0)
                                if m.any() and ctx_embed_frames is not None else None)
            rows.append(row)

        # batched model runs -- the forward-pre-hooks (see __init__) populate
        # self._ges_embed_buf / self._mot_embed_buf as a side effect of these
        # SAME calls, so the embeddings cost nothing extra.
        if ges_batch:
            seq = torch.from_numpy(np.stack([g for _, g in ges_batch])).to(self.device)
            probs = torch.softmax(self.ges_model(seq), dim=1).cpu().numpy()
            embeds = self._ges_embed_buf.cpu().numpy()
            for (wi, _), p, e in zip(ges_batch, probs, embeds):
                rows[wi]["ges_probs"] = p
                rows[wi]["ges_embed"] = e
        if mot_batch:
            seq = torch.from_numpy(np.stack([m for _, m in mot_batch])).to(self.device)
            probs = torch.softmax(self.mot_model(seq), dim=1).cpu().numpy()
            embeds = self._mot_embed_buf.cpu().numpy()
            for (wi, _), p, e in zip(mot_batch, probs, embeds):
                rows[wi]["mot_probs"] = p
                rows[wi]["mot_embed"] = e
        for row in rows:
            row.setdefault("ges_probs", None)
            row.setdefault("mot_probs", None)
            row.setdefault("ges_embed", None)
            row.setdefault("mot_embed", None)
        return rows
