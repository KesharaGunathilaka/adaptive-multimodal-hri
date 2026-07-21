"""Streaming multimodal HRI pipeline for Jetson Orin Nano + RealSense.

Per frame:  one MediaPipe Holistic pass feeds BOTH gesture (image landmarks)
            and motion (world landmarks); frames go into rolling buffers.
Per stride: the 4 cue vectors are computed over their lookback spans, the
            fusion head turns them into an intent, temporal smoothing +
            hysteresis stabilise it, and the policy layer picks a robot action.

The windowing here MIRRORS fusion/extraction/windows.py — same lookback spans,
same uniform resampling, same aggregation. Changing one without the other
creates train/serve skew and the accuracy silently drops.

Deviations from offline extraction, deliberate, for real-time cost:
  * emotion runs on the last EMOTION_FRAMES_PER_STEP frames at each stride
    step (not every frame); offline it averaged every face frame in the span.
  * context (CLIP) recomputes every CONTEXT_EVERY_SEC and is held between
    updates — offline it averaged ~3 Hz samples over a 1 s span.

Usage (see ../JETSON_INFERENCE_GUIDE.md):
    python fusion/pipeline.py --source realsense
    python fusion/pipeline.py --source 0 --backend onnx --display
    python fusion/pipeline.py --source clip.mp4 --no-display --json out.jsonl
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # jetson_deploy/

# ── Windowing constants (must match fusion/extraction/windows.py) ─────────
STRIDE_SEC = 8 / 30                     # ~3.75 Hz intent updates
GES_SPAN = 64 / 30                      # 2.133 s
MOT_SPAN = 2.0
EMO_SPAN = 8 / 30
CONTEXT_EVERY_SEC = 1.0
GES_WINDOW, MOT_WINDOW = 32, 30
GES_MIN_FRAMES, MOT_MIN_FRAMES = 8, 15
EMOTION_FRAMES_PER_STEP = 4
MAX_SIDE = 640

# ── Smoothing / hysteresis (handover §9) ─────────────────────────────────
SMOOTH_N = 3                            # majority vote over last N fused outputs
HYSTERESIS = 2                          # switch only after N consecutive agreeing

INTENTS = [f"F{i:02d}" for i in range(1, 11)]
EMOTION_LABELS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
CONTEXT_LABELS = ["classroom", "kitchen", "hospital", "cloth_store", "museum"]
MP_TO_NTU = {0: 3, 11: 4, 12: 8, 13: 5, 14: 9, 15: 6, 16: 10,
             23: 12, 24: 16, 25: 13, 26: 17, 27: 14, 28: 18}


def _load(alias, path, extra_paths=()):
    """Import a modality file under a unique alias (their generic module names
    collide: config / src / model)."""
    generic = ("config", "src", "model", "inference")
    for n in list(sys.modules):
        if n in generic or n.startswith(tuple(g + "." for g in generic)):
            del sys.modules[n]
    added = [str(p) for p in extra_paths]
    sys.path[0:0] = added
    try:
        spec = importlib.util.spec_from_file_location(alias, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for p in added:
            try:
                sys.path.remove(p)
            except ValueError:
                pass


def uniform_indices(n, target):
    return np.round(np.linspace(0, n - 1, target)).astype(np.int64)


class HRIPipeline:
    def __init__(self, backend="torch", device=None, tau=None, tau_emergency=None):
        import torch

        import mediapipe as mp
        sys.path.insert(0, str(HERE))
        import policy as policy_mod                                  # noqa: E402

        self.policy = policy_mod
        if tau is not None:
            self.policy.TAU = tau
        if tau_emergency is not None:
            self.policy.TAU_EMERGENCY = tau_emergency

        self.torch = torch
        self.backend = backend
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[init] backend={backend} device={self.device}")

        # ── perception models ────────────────────────────────────────────
        emo_dir = ROOT / "modalities" / "emotion"
        self._emo = _load("jd_emotion", emo_dir / "inference" / "video.py")
        ges_dir = ROOT / "modalities" / "gesture"
        gfeat = _load("jd_ges_feats", ges_dir / "src" / "features.py", [ges_dir])
        self._g_build = gfeat.build_features
        mot_dir = ROOT / "modalities" / "motion"
        self._mi = _load("jd_motion", mot_dir / "src" / "inference.py",
                         [mot_dir / "src"])

        if backend == "onnx":
            import onnxruntime as ort
            prov = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if "CUDAExecutionProvider" in ort.get_available_providers()
                    else ["CPUExecutionProvider"])
            odir = ROOT / "onnx"
            self.emo_sess = ort.InferenceSession(str(odir / "emotion_mobilenetv2.onnx"), providers=prov)
            self.ges_sess = ort.InferenceSession(str(odir / "gesture_tcn.onnx"), providers=prov)
            self.mot_sess = ort.InferenceSession(str(odir / "motion_lstm.onnx"), providers=prov)
            self.fus_sess = ort.InferenceSession(str(odir / "fusion_attn.onnx"), providers=prov)
            print(f"[init] ONNX providers: {self.emo_sess.get_providers()}")
        else:
            self._init_torch_models(emo_dir, ges_dir, mot_dir)

        self.emo_tf = self._emo.get_transform()
        self.face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        self.holistic = mp.solutions.holistic.Holistic(
            model_complexity=1, min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # ── context (CLIP) — always PyTorch, never exported ───────────────
        scene_dir = ROOT / "modalities" / "context" / "scene_classification"
        zs = _load("jd_zero_shot", scene_dir / "src" / "zero_shot.py", [scene_dir])
        self.scene = zs.ZeroShotSceneClassifier(device=str(self.device))

        # ── rolling state ────────────────────────────────────────────────
        span = max(GES_SPAN, MOT_SPAN)
        self.buf = deque()               # (t, gesture_feat[185], joints25, pose_ok)
        self.frames = deque(maxlen=8)    # (t, bgr_small, bgr_full) for emotion
        self.buf_span = span
        self.ctx_probs = None
        self.ctx_time = -1e9
        self.history = deque(maxlen=SMOOTH_N)
        self.active_intent = "F05"
        self.candidate = None
        self.candidate_count = 0
        self.timings = Counter()
        self.n_steps = 0

    def _init_torch_models(self, emo_dir, ges_dir, mot_dir):
        torch = self.torch
        self.emo_model = self._emo.build_model().to(self.device).eval()
        self.emo_model.load_state_dict(torch.load(
            emo_dir / "checkpoints" / "finetuned_MobileNetV2.pth",
            map_location=self.device, weights_only=True))

        gm = _load("jd_ges_models", ges_dir / "src" / "models.py", [ges_dir])
        cfg = json.loads((ges_dir / "checkpoints" / "model_config.json").read_text())
        self.ges_model = gm.build_model(cfg["model"], **cfg.get("model_kwargs", {}))
        self.ges_model.load_state_dict(torch.load(
            ges_dir / "checkpoints" / "best_TCN.pth",
            map_location=self.device, weights_only=True))
        self.ges_model.to(self.device).eval()

        ckpt = torch.load(mot_dir / "checkpoints" / "best_model_finetuned.pt",
                          map_location=self.device, weights_only=True)
        mcfg = ckpt.get("config", {})
        self.mot_model = self._mi.MotionLSTM(
            hidden_size=mcfg.get("hidden_size", 256),
            num_layers=mcfg.get("num_layers", 3),
            dropout=mcfg.get("dropout", 0.35)).to(self.device)
        self.mot_model.load_state_dict(ckpt["model_state_dict"])
        self.mot_model.eval()

        from model import AttentionFusion
        fcfg = json.loads((HERE / "fusion_config.json").read_text())
        self.fusion = AttentionFusion(missing_mode="exclude").to(self.device)
        self.fusion.load_state_dict(torch.load(
            HERE / "fusion_attn.pt", map_location=self.device, weights_only=True))
        self.fusion.eval()
        print(f"[init] fusion checkpoint seed={fcfg.get('chosen_seed')} "
              f"test_clip={fcfg.get('test_clip')}")

    # ── per-frame ingest ─────────────────────────────────────────────────
    def push_frame(self, frame_bgr, t):
        h, w = frame_bgr.shape[:2]
        scale = MAX_SIDE / max(h, w)
        small = (cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
                 if scale < 1 else frame_bgr)
        t0 = time.time()
        res = self.holistic.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        self.timings["holistic"] += time.time() - t0

        pose = np.full((1, 33, 4), np.nan, np.float32)
        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                pose[0, i] = [lm.x, lm.y, lm.z, lm.visibility]
        hands = []
        for hl in (res.left_hand_landmarks, res.right_hand_landmarks):
            hand = np.full((1, 21, 3), np.nan, np.float32)
            if hl is not None:
                for i, lm in enumerate(hl.landmark):
                    hand[0, i] = [lm.x, lm.y, lm.z]
            hands.append(hand)
        feat = self._g_build(pose, hands[0], hands[1])[0]

        j25 = np.full((25, 3), np.nan, np.float32)
        if res.pose_world_landmarks:
            wl = res.pose_world_landmarks.landmark
            for mp_i, ntu_i in MP_TO_NTU.items():
                lm = wl[mp_i]
                j25[ntu_i] = [-lm.x, -lm.y, lm.z]
            j25[0] = (j25[12] + j25[16]) / 2
            j25[1] = (j25[4] + j25[8]) / 2
            j25[2] = j25[1] * 0.5 + j25[3] * 0.5

        self.buf.append((t, feat, j25, res.pose_landmarks is not None))
        while self.buf and t - self.buf[0][0] > self.buf_span:
            self.buf.popleft()
        self.frames.append((t, small, frame_bgr))

    # ── cue computation at a stride step ─────────────────────────────────
    def _gesture(self, t):
        idx = [i for i, (ts, _, _, ok) in enumerate(self.buf)
               if ok and t - ts <= GES_SPAN]
        if len(idx) < GES_MIN_FRAMES:
            return None
        sel = np.array(idx)[uniform_indices(len(idx), GES_WINDOW)]
        seq = np.stack([self.buf[i][1] for i in sel])[None].astype(np.float32)
        if self.backend == "onnx":
            out = self.ges_sess.run(None, {self.ges_sess.get_inputs()[0].name: seq})[0]
            return _softmax(out[0])
        with self.torch.no_grad():
            logits = self.ges_model(self.torch.from_numpy(seq).to(self.device))
            return self.torch.softmax(logits[0], 0).cpu().numpy()

    def _motion(self, t):
        idx = [i for i, (ts, _, _, ok) in enumerate(self.buf)
               if ok and t - ts <= MOT_SPAN]
        if len(idx) < MOT_MIN_FRAMES:
            return None
        sel = np.array(idx)[uniform_indices(len(idx), MOT_WINDOW)]
        joints = np.stack([self.buf[i][2] for i in sel])
        pos = np.stack([self._mi.normalize_skeleton(j[self._mi.JOINT_SUBSET]).flatten()
                        for j in joints])
        vel = np.vstack([np.zeros((1, 42), np.float32), np.diff(pos, axis=0)])
        seq = np.concatenate([pos, vel], axis=1)[None].astype(np.float32)
        if self.backend == "onnx":
            out = self.mot_sess.run(None, {self.mot_sess.get_inputs()[0].name: seq})[0]
            return _softmax(out[0])
        with self.torch.no_grad():
            logits = self.mot_model(self.torch.from_numpy(seq).to(self.device))
            return self.torch.softmax(logits[0], 0).cpu().numpy()

    def _emotion(self, t):
        from PIL import Image
        recent = [f for f in self.frames if t - f[0] <= EMO_SPAN] or list(self.frames)[-1:]
        recent = recent[-EMOTION_FRAMES_PER_STEP:]
        crops = []
        for _, small, full in recent:
            box = self._emo.detect_face_box(self.face_det, full, small)
            if box is None:
                continue
            x, y, bw, bh = box
            c = full[y:y + bh, x:x + bw]
            if c.size:
                crops.append(c)
        if not crops:
            return None
        batch = self.torch.stack([
            self.emo_tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
            for c in crops])
        if self.backend == "onnx":
            out = self.emo_sess.run(
                None, {self.emo_sess.get_inputs()[0].name: batch.numpy()})[0]
            return np.stack([_softmax(o) for o in out]).mean(0)
        with self.torch.no_grad():
            p = self.torch.softmax(self.emo_model(batch.to(self.device)), 1)
        return p.mean(0).cpu().numpy()

    def _context(self, t):
        if self.ctx_probs is not None and t - self.ctx_time < CONTEXT_EVERY_SEC:
            return self.ctx_probs
        if not self.frames:
            return self.ctx_probs
        r = self.scene.predict(self.frames[-1][1])
        self.ctx_probs = np.array([r["probs"][c] for c in self.scene.classes],
                                  dtype=np.float32)
        self.ctx_time = t
        return self.ctx_probs

    # ── fused step ───────────────────────────────────────────────────────
    def step(self, t):
        """Run one fusion step. Returns a result dict."""
        t0 = time.time()
        emo, ges = self._emotion(t), self._gesture(t)
        mot, ctx = self._motion(t), self._context(t)

        x = np.zeros(24, np.float32)
        obs = np.zeros(4, np.float32)
        for k, (p, sl) in enumerate([(emo, slice(0, 7)), (ges, slice(7, 15)),
                                     (mot, slice(15, 19)), (ctx, slice(19, 24))]):
            if p is not None:
                x[sl] = p
                obs[k] = 1.0

        if self.backend == "onnx":
            names = [i.name for i in self.fus_sess.get_inputs()]
            logits = self.fus_sess.run(None, {names[0]: x[None], names[1]: obs[None]})[0][0]
            probs = _softmax(logits)
        else:
            with self.torch.no_grad():
                logits = self.fusion(self.torch.from_numpy(x[None]).to(self.device),
                                     self.torch.from_numpy(obs[None]).to(self.device))
                probs = self.torch.softmax(logits[0], 0).cpu().numpy()

        ctx_label = (CONTEXT_LABELS[int(np.argmax(ctx))] if ctx is not None
                     else "unknown")
        raw = self.policy.decide(probs, context_label=ctx_label)

        # ── emergency bypass: F02 skips smoothing AND hysteresis ─────────
        if raw.emergency:
            self.active_intent = "F02"
            self.history.clear()
            self.candidate, self.candidate_count = None, 0
            decision = raw
        else:
            self.history.append(raw.intent)
            voted = Counter(self.history).most_common(1)[0][0]
            if voted != self.active_intent:
                if voted == self.candidate:
                    self.candidate_count += 1
                else:
                    self.candidate, self.candidate_count = voted, 1
                if self.candidate_count >= HYSTERESIS:
                    self.active_intent = voted
                    self.candidate, self.candidate_count = None, 0
            else:
                self.candidate, self.candidate_count = None, 0
            decision = self.policy.decide(
                _one_hot_like(probs, self.active_intent),
                context_label=ctx_label) if self.active_intent != raw.intent else raw

        self.timings["fusion_step"] += time.time() - t0
        self.n_steps += 1
        return {
            "t": round(t, 3),
            "raw_intent": raw.intent,
            "intent": self.active_intent,
            "action": decision.action,
            "action_text": self.policy.ACTIONS[decision.action],
            "confidence": round(float(probs.max()), 3),
            "emergency": bool(raw.emergency),
            "fallback": bool(decision.fallback),
            "observed": {"emotion": bool(obs[0]), "gesture": bool(obs[1]),
                         "motion": bool(obs[2]), "context": bool(obs[3])},
            "cues": {
                "emotion": EMOTION_LABELS[int(np.argmax(emo))] if emo is not None else None,
                "gesture": int(np.argmax(ges)) if ges is not None else None,
                "motion": int(np.argmax(mot)) if mot is not None else None,
                "context": ctx_label,
            },
            "step_ms": round((time.time() - t0) * 1000, 1),
        }


def _softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def _one_hot_like(probs, intent):
    """Keep the policy's threshold logic meaningful for a hysteresis-held
    intent: reuse the real probability of that intent."""
    out = np.zeros_like(probs)
    i = INTENTS.index(intent)
    out[i] = max(float(probs[i]), 0.99)
    return out


# ── sources ──────────────────────────────────────────────────────────────
def open_source(source):
    if source == "realsense":
        import pyrealsense2 as rs
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe.start(cfg)

        def read():
            frames = pipe.wait_for_frames()
            c = frames.get_color_frame()
            if not c:
                return None
            return np.asanyarray(c.get_data())
        return read, pipe.stop
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {source}")

    def read():
        ok, f = cap.read()
        return f if ok else None
    return read, cap.release


def main():
    ap = argparse.ArgumentParser(description="Streaming HRI fusion pipeline")
    ap.add_argument("--source", default="realsense",
                    help="'realsense', a camera index ('0'), or a video path")
    ap.add_argument("--backend", choices=("torch", "onnx"), default="torch")
    ap.add_argument("--display", dest="display", action="store_true", default=True)
    ap.add_argument("--no-display", dest="display", action="store_false")
    ap.add_argument("--json", default=None, help="append results as JSONL")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--tau-emergency", type=float, default=None)
    args = ap.parse_args()

    pipe = HRIPipeline(backend=args.backend, tau=args.tau,
                       tau_emergency=args.tau_emergency)
    read, close = open_source(args.source)
    jf = open(args.json, "a", encoding="utf-8") if args.json else None

    t_start = time.time()
    next_step = STRIDE_SEC
    last = None
    n_frames = 0
    try:
        while True:
            frame = read()
            if frame is None:
                break
            t = time.time() - t_start
            pipe.push_frame(frame, t)
            n_frames += 1
            if t >= next_step:
                last = pipe.step(t)
                next_step += STRIDE_SEC
                print(f"[{last['t']:6.2f}s] {last['intent']} -> {last['action']} "
                      f"p={last['confidence']:.2f} {last['step_ms']:.0f}ms "
                      f"{'EMERGENCY' if last['emergency'] else ''}", flush=True)
                if jf:
                    jf.write(json.dumps(last) + "\n")
            if args.display:
                _draw(frame, last)
                cv2.imshow("HRI fusion", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_seconds and t > args.max_seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        close()
        if jf:
            jf.close()
        cv2.destroyAllWindows()

    dur = time.time() - t_start
    print(f"\n--- {n_frames} frames in {dur:.1f}s ({n_frames/max(dur,1e-9):.1f} fps), "
          f"{pipe.n_steps} fusion steps")
    if pipe.n_steps:
        print(f"    mean holistic {1000*pipe.timings['holistic']/max(n_frames,1):.1f} ms/frame, "
              f"mean fusion step {1000*pipe.timings['fusion_step']/pipe.n_steps:.1f} ms")


def _draw(frame, r):
    if not r:
        return
    h = frame.shape[0]
    colour = (0, 0, 255) if r["emergency"] else (0, 200, 0)
    cv2.rectangle(frame, (0, h - 60), (frame.shape[1], h), (0, 0, 0), -1)
    cv2.putText(frame, f"{r['intent']}  {r['action']}  p={r['confidence']:.2f}",
                (10, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    miss = [k for k, v in r["observed"].items() if not v]
    cv2.putText(frame, r["action_text"][:58] + ("  | missing: " + ",".join(miss) if miss else ""),
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


if __name__ == "__main__":
    main()
