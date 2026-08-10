"""Pass 1 of feature extraction: per-frame raw outputs for all 4 modalities.

One decode pass per clip produces a compressed .npz cache:
    emotion_probs [T,7]  float32, NaN rows where no face was detected
    gesture_feats [T,185] float32 (NaN-safe per modalities/gesture/src/features.py)
    pose_img      [T,33,4] float32 raw MediaPipe pose in IMAGE space
                  (x, y normalised to frame, z, visibility), NaN where no pose.
                  gesture_feats divides out the mid-shoulder centre and the
                  shoulder-width scale, so apparent size and screen position —
                  the only signals that distinguish approaching from receding —
                  are unrecoverable from it. Cached here so a direction cue can
                  be derived later without decoding every video again.
    pose_valid    [T]    bool    (MediaPipe Holistic pose found)
    face_valid    [T]    bool
    joints25      [T,25,3] float32, NaN where no pose (NTU layout, metres —
                  same MP->NTU mapping + spine approximation as
                  modalities/motion/inference/video.py)
    context_probs [Tc,5] float32 raw CLIP scene softmax (no temporal smoothing)
    context_frames[Tc]   int64   frame index of each context sample (~3 Hz)
    fps, n_frames        scalars

Windowing/aggregation happens in windows.py — re-runnable without touching
videos again (handover §7.1). MediaPipe Holistic serves BOTH gesture (image
landmarks + hands) and motion (world landmarks) from a single pass.
"""
import os

import cv2
import numpy as np
import torch
from PIL import Image

from .modloader import REPO, load_module

# Context CLIP weights come from the repo-bundled HF cache (verified offline).
os.environ.setdefault("HF_HOME", str(REPO / "jetson_deploy" / "hf_cache"))

EMO_DIR = REPO / "modalities" / "emotion"
GES_DIR = REPO / "modalities" / "gesture"
CTX_SCENE_DIR = REPO / "modalities" / "context" / "scene_classification"

EMOTION_CKPT = EMO_DIR / "checkpoints" / "finetuned_MobileNetV2.pth"

MAX_SIDE = 640          # Holistic/face-detection run on frames capped to this
CONTEXT_HZ = 3.0        # CLIP sampling rate
EMO_BATCH = 64

# MediaPipe pose -> NTU-25 joint mapping (modalities/motion/inference/video.py)
MP_TO_NTU = {0: 3, 11: 4, 12: 8, 13: 5, 14: 9, 15: 6, 16: 10,
             23: 12, 24: 16, 25: 13, 26: 17, 27: 14, 28: 18}


class PerFrameExtractor:
    # `extract_embeddings_only` only: sources at/above this width or height use
    # the cheap small-frame-tiled face-detection fallback instead of
    # `detect_face_box`'s normal full-resolution tiling. 3000 sits between
    # this dataset's phone_1080p (1920) and phone_4k (3840) sources, so 1080p
    # keeps the full-precision path (never the slow one) and only 4K sources
    # take the tradeoff (see `extract_embeddings_only`'s docstring).
    _HIRES_TILE_LIMIT = 3000


    def __init__(self, device=None):
        import mediapipe as mp

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # ── Emotion (self-contained inference script) ────────────────────
        self._emo = load_module("hri_emotion_video", EMO_DIR / "inference" / "video.py")
        self.emotion_labels = list(self._emo.EMOTION_LABELS)
        self.emo_model = self._emo.build_model().to(self.device)
        self.emo_model.load_state_dict(
            torch.load(EMOTION_CKPT, map_location=self.device, weights_only=True))
        self.emo_model.eval()
        self.emo_tf = self._emo.get_transform()
        self.face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

        # ── Gesture feature engineering (reused verbatim) ─────────────────
        gf = load_module("hri_gesture_features", GES_DIR / "src" / "features.py", [GES_DIR])
        self._g_build = gf.build_features

        self.holistic = mp.solutions.holistic.Holistic(
            model_complexity=1, min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # ── Context: CLIP zero-shot, raw per-frame probs (bypass smoothing) ─
        zs = load_module("hri_zero_shot", CTX_SCENE_DIR / "src" / "zero_shot.py",
                         [CTX_SCENE_DIR])
        self.scene = zs.ZeroShotSceneClassifier(device=str(self.device))
        self.context_labels = list(self.scene.classes)

        print(f"PerFrameExtractor ready | device={self.device} | "
              f"emotion={EMOTION_CKPT.name} | context=CLIP zero-shot "
              f"({len(self.context_labels)} scenes)")

    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _emotion_batch(self, crops):
        """List of BGR face crops -> ([N,7] softmax, [N,1280] penultimate
        embedding). The embedding is torchvision MobileNetV2's own
        pooled-feature input to `classifier` (features -> global avgpool ->
        flatten) -- splitting its forward into these two explicit steps costs
        nothing extra (same compute the softmax already required), added
        2026-08-08 for embedding-level fusion (`fusion/fusion-engine-embeddings`
        design note: penultimate-layer features retain the uncertainty argmax
        collapses)."""
        probs_out, embed_out = [], []
        for i in range(0, len(crops), EMO_BATCH):
            batch = torch.stack([
                self.emo_tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                for c in crops[i:i + EMO_BATCH]]).to(self.device)
            feat = self.emo_model.features(batch)
            embed = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(feat, 1), 1)
            probs_out.append(torch.softmax(self.emo_model.classifier(embed), dim=1)
                             .cpu().numpy())
            embed_out.append(embed.cpu().numpy())
        probs = np.concatenate(probs_out) if probs_out else np.zeros((0, 7), np.float32)
        embed = (np.concatenate(embed_out) if embed_out
                else np.zeros((0, 1280), np.float32))
        return probs, embed.astype(np.float32)

    @torch.no_grad()
    def _context_batch(self, frames_bgr):
        """List of BGR frames -> ([N,5] raw CLIP scene softmax (no smoothing),
        [N,512] L2-normalised CLIP image embedding). The embedding was already
        being computed and discarded before the text-similarity step; now
        returned too (2026-08-08, same rationale as `_emotion_batch`)."""
        m = self.scene
        probs_out, embed_out = [], []
        for i in range(0, len(frames_bgr), EMO_BATCH):
            imgs = torch.stack([
                m.preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                for f in frames_bgr[i:i + EMO_BATCH]]).to(self.device)
            emb = m.model.encode_image(imgs).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            logits = 100.0 * emb @ m.text_embs.T
            probs = torch.softmax(logits[:, :len(m.classes)], dim=1)
            probs_out.append(probs.cpu().numpy())
            embed_out.append(emb.cpu().numpy())
        probs = (np.concatenate(probs_out) if probs_out
                else np.zeros((0, len(m.classes)), np.float32))
        embed = (np.concatenate(embed_out) if embed_out
                else np.zeros((0, 512), np.float32))
        return probs, embed.astype(np.float32)

    # ────────────────────────────────────────────────────────────────────
    def extract_clip(self, video_path, frame_transform=None):
        """`frame_transform`, if given, is applied to each raw BGR frame right
        after decode -- before Holistic/face-detection/CLIP ever see it. Used
        by `scripts/44_video_degradation.py` (Phase 2 robustness study,
        2026-08-06) to corrupt REAL pixels (blur/darken/downsample/compress)
        rather than simulating cue noise post-hoc, closing the honesty gap
        flagged in `07_evaluation.md` §7.5 (designed-missing rows were found
        to have 100% detection anyway -- nothing was actually corrupted).
        None (default) reproduces the exact original behaviour."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"cannot open {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        ctx_every = max(1, round(fps / CONTEXT_HZ))

        pose_arr, lh_arr, rh_arr = [], [], []
        joints_list, pose_valid, face_valid = [], [], []
        face_crops, face_frame_idx = [], []
        ctx_frames, ctx_frame_idx = [], []

        t = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_transform is not None:
                frame = frame_transform(frame)
            h, w = frame.shape[:2]
            scale = MAX_SIDE / max(h, w)
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame

            res = self.holistic.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

            # gesture landmark arrays (NaN where absent — features.py contract)
            pose = np.full((33, 4), np.nan, np.float32)
            if res.pose_landmarks:
                for i, lm in enumerate(res.pose_landmarks.landmark):
                    pose[i] = [lm.x, lm.y, lm.z, lm.visibility]
            hands = []
            for hl in (res.left_hand_landmarks, res.right_hand_landmarks):
                hand = np.full((21, 3), np.nan, np.float32)
                if hl is not None:
                    for i, lm in enumerate(hl.landmark):
                        hand[i] = [lm.x, lm.y, lm.z]
                hands.append(hand)
            pose_arr.append(pose); lh_arr.append(hands[0]); rh_arr.append(hands[1])
            pose_valid.append(res.pose_landmarks is not None)

            # motion joints from world landmarks (X/Y sign flip, NTU layout)
            j25 = np.full((25, 3), np.nan, np.float32)
            if res.pose_world_landmarks:
                wl = res.pose_world_landmarks.landmark
                for mp_i, ntu_i in MP_TO_NTU.items():
                    lm = wl[mp_i]
                    j25[ntu_i] = [-lm.x, -lm.y, lm.z]
                j25[0] = (j25[12] + j25[16]) / 2          # spine base
                j25[1] = (j25[4] + j25[8]) / 2            # spine shoulder
                j25[2] = j25[1] * 0.5 + j25[3] * 0.5      # neck
            joints_list.append(j25)

            # emotion face crop (robust multi-pass detector from emotion/video.py)
            box = self._emo.detect_face_box(self.face_det, frame, small)
            ok = False
            if box is not None:
                x, y, bw, bh = box
                crop = frame[y:y + bh, x:x + bw]
                if crop.size:
                    face_crops.append(crop)
                    face_frame_idx.append(t)
                    ok = True
            face_valid.append(ok)

            if t % ctx_every == 0:
                ctx_frames.append(small.copy())
                ctx_frame_idx.append(t)
            t += 1
        cap.release()

        T = t
        emotion_probs = np.full((T, len(self.emotion_labels)), np.nan, np.float32)
        emotion_embed = np.full((T, 1280), np.nan, np.float32)
        if face_crops:
            ep, ee = self._emotion_batch(face_crops)
            emotion_probs[face_frame_idx] = ep
            emotion_embed[face_frame_idx] = ee
        gesture_feats = (self._g_build(np.stack(pose_arr), np.stack(lh_arr),
                                       np.stack(rh_arr))
                         if T else np.zeros((0, 185), np.float32))
        context_probs, context_embed = self._context_batch(ctx_frames)

        return {
            "emotion_probs": emotion_probs,
            "emotion_embed": emotion_embed,
            "gesture_feats": gesture_feats.astype(np.float32),
            "pose_img": np.stack(pose_arr) if T else np.zeros((0, 33, 4), np.float32),
            "pose_valid": np.array(pose_valid, bool),
            "face_valid": np.array(face_valid, bool),
            "joints25": np.stack(joints_list) if T else np.zeros((0, 25, 3), np.float32),
            "context_probs": context_probs,
            "context_embed": context_embed,
            "context_frames": np.array(ctx_frame_idx, np.int64),
            "fps": np.float32(fps),
            "n_frames": np.int64(T),
        }

    # ────────────────────────────────────────────────────────────────────
    def extract_embeddings_only(self, video_path, frame_transform=None):
        """Lean pass for embedding-level fusion (2026-08-08): emotion +
        context penultimate embeddings only, WITHOUT running MediaPipe
        Holistic. Holistic (~150-200ms/frame, methodology doc §8) is the
        dominant cost of `extract_clip` and is not needed here -- emotion's
        face crop comes from the separate, cheap `FaceDetection` model
        (`self.face_det`, ~14ms/frame), and context sampling needs only the
        resized frame. Gesture/motion embeddings are NOT produced here; they
        are derived in Pass 2 from the EXISTING `gesture_feats`/`joints25`
        already cached by `extract_clip`, so no video is re-decoded for them.

        For high-resolution sources (width or height >= `_HIRES_TILE_LIMIT`),
        uses face detection on `small` for ALL THREE of `detect_face_box`'s
        tiers, including its tiled-CLAHE fallback (normally tiled on the
        FULL-resolution frame) -- found live 2026-08-08 on a 4K ("phone_4k",
        3840x2160) clip whose subject faces away from camera for the entire
        168-frame clip: every frame hit that fallback (full-res CLAHE + 4x
        FaceDetection.process() on ~2500x1400 tiles), costing 54 MINUTES for
        one clip; 228 more 4K clips were queued when this was caught. Below
        the resolution limit, `detect_face_box` runs completely UNCHANGED
        (full contract, full-res fallback) -- this was never the slow path at
        moderate resolutions, so there is no reason to trade any precision
        there. When the fast path IS used, `detect_face_box(self.face_det,
        small, small)` (passing `small` for both args) runs the identical
        3-tier logic entirely on the downscaled frame; the returned box is
        rescaled back to original pixel coordinates below. Measured cost of
        this tradeoff on the pathological clip: emotion_probs match on frames
        resolved by tier 1/2 (identical in both paths -- they only ever look
        at `small`), max |diff| 0.287 on the minority of frames that needed
        tier 3, where box-coordinate rescaling amplifies any small-frame
        detection error ~6x. Aggregate window-pooling should absorb most of
        this; `sanity_check` in `scripts/54_extract_embeddings.py` (checked
        on 1/N clips) will flag it if it ever costs more than that.

        Returns a subset of `extract_clip`'s keys: emotion_probs/embed,
        face_valid, context_probs/embed, context_frames, fps, n_frames.
        `frame_transform` behaves as in `extract_clip` (video-degradation
        studies)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"cannot open {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        ctx_every = max(1, round(fps / CONTEXT_HZ))

        face_valid = []
        face_crops, face_frame_idx = [], []
        ctx_frames, ctx_frame_idx = [], []

        t = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_transform is not None:
                frame = frame_transform(frame)
            h, w = frame.shape[:2]
            scale = MAX_SIDE / max(h, w)
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame

            if max(h, w) >= self._HIRES_TILE_LIMIT:
                box = self._emo.detect_face_box(self.face_det, small, small)
                if box is not None:
                    sh, sw = small.shape[:2]
                    sy, sx = h / sh, w / sw
                    bx, by, bbw, bbh = box
                    box = (int(bx * sx), int(by * sy), int(bbw * sx), int(bbh * sy))
            else:
                box = self._emo.detect_face_box(self.face_det, frame, small)
            ok = False
            if box is not None:
                x, y, bw, bh = box
                crop = frame[y:y + bh, x:x + bw]
                if crop.size:
                    face_crops.append(crop)
                    face_frame_idx.append(t)
                    ok = True
            face_valid.append(ok)

            if t % ctx_every == 0:
                ctx_frames.append(small.copy())
                ctx_frame_idx.append(t)
            t += 1
        cap.release()

        T = t
        emotion_probs = np.full((T, len(self.emotion_labels)), np.nan, np.float32)
        emotion_embed = np.full((T, 1280), np.nan, np.float32)
        if face_crops:
            ep, ee = self._emotion_batch(face_crops)
            emotion_probs[face_frame_idx] = ep
            emotion_embed[face_frame_idx] = ee
        context_probs, context_embed = self._context_batch(ctx_frames)

        return {
            "emotion_probs": emotion_probs,
            "emotion_embed": emotion_embed,
            "face_valid": np.array(face_valid, bool),
            "context_probs": context_probs,
            "context_embed": context_embed,
            "context_frames": np.array(ctx_frame_idx, np.int64),
            "fps": np.float32(fps),
            "n_frames": np.int64(T),
        }
