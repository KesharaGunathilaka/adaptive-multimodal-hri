"""Pass 1 of feature extraction: per-frame raw outputs for all 4 modalities.

One decode pass per clip produces a compressed .npz cache:
    emotion_probs [T,7]  float32, NaN rows where no face was detected
    gesture_feats [T,185] float32 (NaN-safe per modalities/gesture/src/features.py)
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
        """List of BGR face crops -> [N,7] softmax."""
        out = []
        for i in range(0, len(crops), EMO_BATCH):
            batch = torch.stack([
                self.emo_tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                for c in crops[i:i + EMO_BATCH]]).to(self.device)
            out.append(torch.softmax(self.emo_model(batch), dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 7), np.float32)

    @torch.no_grad()
    def _context_batch(self, frames_bgr):
        """List of BGR frames -> [N,5] raw CLIP scene softmax (no smoothing)."""
        m = self.scene
        out = []
        for i in range(0, len(frames_bgr), EMO_BATCH):
            imgs = torch.stack([
                m.preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                for f in frames_bgr[i:i + EMO_BATCH]]).to(self.device)
            emb = m.model.encode_image(imgs).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            logits = 100.0 * emb @ m.text_embs.T
            probs = torch.softmax(logits[:, :len(m.classes)], dim=1)
            out.append(probs.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(m.classes)), np.float32)

    # ────────────────────────────────────────────────────────────────────
    def extract_clip(self, video_path):
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
        if face_crops:
            emotion_probs[face_frame_idx] = self._emotion_batch(face_crops)
        gesture_feats = (self._g_build(np.stack(pose_arr), np.stack(lh_arr),
                                       np.stack(rh_arr))
                         if T else np.zeros((0, 185), np.float32))
        context_probs = self._context_batch(ctx_frames)

        return {
            "emotion_probs": emotion_probs,
            "gesture_feats": gesture_feats.astype(np.float32),
            "pose_valid": np.array(pose_valid, bool),
            "face_valid": np.array(face_valid, bool),
            "joints25": np.stack(joints_list) if T else np.zeros((0, 25, 3), np.float32),
            "context_probs": context_probs,
            "context_frames": np.array(ctx_frame_idx, np.int64),
            "fps": np.float32(fps),
            "n_frames": np.int64(T),
        }
