"""Live laptop demo (v4): all 4 cues -> fused intent -> robot action.

Shows on the webcam feed:
    emotion  (7-class + the 5-class superclass the fusion head really sees)
    gesture  (8-class TCN)
    motion   (4-class LSTM)          (shown only; see --hide-motion)
    context  (CLIP scene, 5-class)   <- fusion sees it only if classroom/kitchen
    intent   (F01..F10, smoothed + hysteresis + F02 emergency bypass)
    action   (A01..A14 from policy_v4)

Motion is shown but deliberately NOT an input of the v4 fusion head: its
measured reliability was too low (see JETSON_V4_DEPLOYMENT_GUIDE.md §1).

Run from the repo root with the GPU venv:
    .venv\\Scripts\\python.exe jetson_deploy\\fusion\\live_laptop_v4.py
    ... --source 0 --backend torch            # default webcam
    ... --source path\\to\\clip.mp4            # a video file instead
    ... --json out.jsonl --max-seconds 60
Keys: q/Esc quit, f toggle full screen.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline_v4 as p4                                          # noqa: E402
from pipeline_v4 import (CONTEXT_LABELS, EMOTION_LABELS, ROOT,   # noqa: E402
                         STRIDE_SEC, SUPER_LABELS, HRIPipeline,
                         open_source, uniform_indices)

GESTURE_LABELS = ["idle", "wave", "point", "thumbs_up", "thumbs_down",
                  "beckoning", "raise_hand", "both_hands_up"]
MOTION_LABELS = ["sitting", "standing", "walking", "stepping_back"]
HAND_MIN_FRAMES, HAND_MIN_FRAC = 6, 0.3
MOT_SPAN, MOT_WINDOW, MOT_MIN_FRAMES = 2.0, 30, 15
MP_TO_NTU = {0: 3, 11: 4, 12: 8, 13: 5, 14: 9, 15: 6, 16: 10,
             23: 12, 24: 16, 25: 13, 26: 17, 27: 14, 28: 18}


class _HolisticTap:
    """Proxy that remembers the last Holistic result so we can also derive the
    world-landmark skeleton for the (display-only) motion cue without running
    Holistic twice."""

    def __init__(self, inner):
        self.inner, self.last = inner, None

    def process(self, rgb):
        self.last = self.inner.process(rgb)
        return self.last


class LivePipeline(HRIPipeline):
    def __init__(self, **kw):
        super().__init__(backend="torch", **kw)
        self.holistic = _HolisticTap(self.holistic)
        self.skel = []                       # (t, joints25) for motion
        self.hand_log = []                   # (t, any hand detected)
        self._load_motion()

    def _load_motion(self):
        torch = self.torch
        mot_dir = ROOT / "modalities" / "motion"
        self._mi = p4._load("jd_motion", mot_dir / "src" / "inference.py",
                            [mot_dir / "src"])
        ckpt = torch.load(mot_dir / "checkpoints" / "best_model_finetuned.pt",
                          map_location=self.device, weights_only=True)
        m = ckpt.get("config", {})
        self.mot_model = self._mi.MotionLSTM(
            hidden_size=m.get("hidden_size", 256),
            num_layers=m.get("num_layers", 3),
            dropout=m.get("dropout", 0.35)).to(self.device)
        self.mot_model.load_state_dict(ckpt["model_state_dict"])
        self.mot_model.eval()

    def push_frame(self, frame_bgr, t):
        super().push_frame(frame_bgr, t)
        res = self.holistic.last
        j25 = np.full((25, 3), np.nan, np.float32)
        ok = bool(res and res.pose_world_landmarks)
        if ok:
            wl = res.pose_world_landmarks.landmark
            for mp_i, ntu_i in MP_TO_NTU.items():
                lm = wl[mp_i]
                j25[ntu_i] = [-lm.x, -lm.y, lm.z]
            j25[0] = (j25[12] + j25[16]) / 2
            j25[1] = (j25[4] + j25[8]) / 2
            j25[2] = j25[1] * 0.5 + j25[3] * 0.5
        hands = bool(res and (res.left_hand_landmarks or res.right_hand_landmarks))
        self.hand_log.append((t, hands))
        while self.hand_log and t - self.hand_log[0][0] > p4.GES_SPAN:
            self.hand_log.pop(0)
        self.skel.append((t, j25, ok))
        while self.skel and t - self.skel[0][0] > MOT_SPAN:
            self.skel.pop(0)

    def _gesture(self, t):
        # The TCN always returns a confident-looking class, even for an input
        # with no hands (a face-only close-up came out as thumbs_down 0.8). A
        # gesture counts only if hands were really seen in enough recent frames.
        n_hand = sum(h for _, h in self.hand_log)
        if n_hand < HAND_MIN_FRAMES or n_hand < HAND_MIN_FRAC * len(self.hand_log):
            self._last_ges = None
            return None
        self._last_ges = super()._gesture(t)
        return self._last_ges

    def _emotion_super(self, t):
        self._last_emo = super()._emotion_super(t)
        return self._last_emo

    def motion(self, t):
        idx = [i for i, (ts, _, ok) in enumerate(self.skel)
               if ok and t - ts <= MOT_SPAN]
        if len(idx) < MOT_MIN_FRAMES:
            return None
        sel = np.array(idx)[uniform_indices(len(idx), MOT_WINDOW)]
        pos = np.stack([self._mi.normalize_skeleton(
            self.skel[i][1][self._mi.JOINT_SUBSET]).flatten() for i in sel])
        vel = np.vstack([np.zeros((1, 42), np.float32), np.diff(pos, axis=0)])
        seq = np.concatenate([pos, vel], axis=1)[None].astype(np.float32)
        with self.torch.no_grad():
            lg = self.mot_model(self.torch.from_numpy(seq).to(self.device))
            return self.torch.softmax(lg[0], 0).cpu().numpy()

    def full_step(self, t):
        """fusion step + the extra display-only motion cue + top-1 confidences."""
        r = self.step(t)
        mot = self.motion(t)
        ges, emo = self._last_ges, self._last_emo   # cached by step()
        cues = r["cues"]
        if emo is None and ges is None:      # nobody detected: no decision
            self.history.clear()
            self.candidate, self.candidate_count = None, 0
            self.active_intent = "F05"
            r["intent"] = r["action"] = None
        cues["motion"] = (MOTION_LABELS[int(np.argmax(mot))], float(mot.max())) if mot is not None else None
        cues["gesture"] = (GESTURE_LABELS[int(np.argmax(ges))], float(ges.max())) if ges is not None else None
        cues["emotion"] = (EMOTION_LABELS[int(np.argmax(emo[1]))], float(emo[1].max())) if emo is not None else None
        cues["emotion_super"] = (SUPER_LABELS[int(np.argmax(emo[0]))], float(emo[0].max())) if emo is not None else None
        r["observed"]["motion"] = mot is not None
        return r


INTENT_NAMES = {
    "F01": "Greeting / positive acknowledgment", "F02": "Emergency",
    "F03": "Request to follow / go to task", "F04": "Needs help / guidance",
    "F05": "No interaction intended", "F06": "Asking robot to move aside",
    "F07": "Frustration", "F08": "Break request",
    "F10": "Discouraged / disengaged",
}


def _fmt(c):
    return "--" if c is None else f"{c[0]} {c[1]:.2f}"


def _screen_size():
    try:
        import ctypes
        u = ctypes.windll.user32
        u.SetProcessDPIAware()
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def _text(img, txt, org, scale, col, thick=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_DUPLEX, scale, col, thick, cv2.LINE_AA)


def compose(frame, r, size, show_motion=True):
    """Letterbox the camera frame onto a size=(W,H) canvas and draw the UI at
    canvas resolution so text stays crisp in full screen."""
    W, H = size
    fh, fw = frame.shape[:2]
    sc = min(W / fw, H / fh)
    nw, nh = int(fw * sc), int(fh * sc)
    canvas = np.zeros((H, W, 3), np.uint8)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(frame, (nw, nh))
    if not r:
        return canvas

    u = H / 1080.0                           # UI scale
    c = r["cues"]
    emerg, fb = r["emergency"], r["fallback"]
    accent = (60, 60, 255) if emerg else ((0, 190, 255) if fb else (90, 220, 90))

    # cue card (top-left of the video)
    rows = [("EMOTION", c["emotion"]), ("GESTURE", c["gesture"]),
            ("MOTION", c["motion"] if show_motion else None),
            ("CONTEXT", (c["context_full"], None))]
    rows = [(n, v) for n, v in rows if v is not None]
    pad, rh = int(24 * u), int(66 * u)
    cw, ch = int(520 * u), pad * 2 + rh * len(rows)
    if rows:
        ov = canvas.copy()
        cv2.rectangle(ov, (x0 + 20, y0 + 20), (x0 + 20 + cw, y0 + 20 + ch), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.62, canvas, 0.38, 0, canvas)
    for i, (name, val) in enumerate(rows):
        y = y0 + 20 + pad + i * rh
        _text(canvas, name, (x0 + 20 + pad, y + int(14 * u)), 0.55 * u, (160, 160, 160))
        if val[1] is None:
            txt = val[0].replace("_", " ")
        else:
            txt = f"{val[0].replace('_', ' ')}  {int(val[1] * 100)}%"
        _text(canvas, txt, (x0 + 20 + pad, y + int(50 * u)), 1.0 * u, (255, 255, 255), 2)

    if r["intent"] is None:                  # no person cue -> blank decision
        return canvas

    # intent / action banner (bottom, full width)
    bh = int(190 * u)
    ov = canvas.copy()
    cv2.rectangle(ov, (0, H - bh), (W, H), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.75, canvas, 0.25, 0, canvas)
    cv2.rectangle(canvas, (0, H - bh), (int(14 * u), H), accent, -1)
    m = int(50 * u)
    _text(canvas, "DETECTED INTENT", (m, H - bh + int(46 * u)), 0.6 * u, (160, 160, 160))
    _text(canvas, r["intent"], (m, H - bh + int(115 * u)), 2.2 * u, accent, 3)
    _text(canvas, INTENT_NAMES[r["intent"]], (m, H - bh + int(165 * u)), 0.7 * u, (235, 235, 235))
    xa = m + int(560 * u)
    _text(canvas, "ROBOT ACTION", (xa, H - bh + int(46 * u)), 0.6 * u, (160, 160, 160))
    _text(canvas, f"{r['action']}  " + ("EMERGENCY STOP" if emerg else ""),
          (xa, H - bh + int(100 * u)), 1.5 * u, accent, 2)
    _text(canvas, r["action_text"], (xa, H - bh + int(155 * u)), 0.85 * u, (235, 235, 235))
    conf = f"{int(r['confidence'] * 100)}%"
    _text(canvas, conf, (W - int(170 * u), H - bh + int(135 * u)), 1.6 * u, (235, 235, 235), 2)
    _text(canvas, "confidence", (W - int(170 * u), H - bh + int(46 * u)), 0.6 * u, (160, 160, 160))
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Live laptop HRI demo (v4)")
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--json", default=None, help="append results as JSONL")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--tau-emergency", type=float, default=None)
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--windowed", action="store_true", help="don't start in full screen")
    ap.add_argument("--hide-motion", action="store_true",
                    help="omit the motion row (it never influences the intent)")
    ap.add_argument("--no-mirror", action="store_true", help="don't mirror the webcam preview")
    a = ap.parse_args()

    pipe = LivePipeline(tau=a.tau, tau_emergency=a.tau_emergency)
    read, close = open_source(a.source)
    jf = open(a.json, "a", encoding="utf-8") if a.json else None
    is_cam = str(a.source).isdigit()
    win = "HRI live (v4)"
    size = (1280, 720) if a.windowed else _screen_size()
    if not a.no_display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        if not a.windowed:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = not a.windowed
    t0 = time.time()
    next_step, last = STRIDE_SEC, None
    try:
        while True:
            frame = read()
            if frame is None:
                break
            t = time.time() - t0
            pipe.push_frame(frame, t)
            if t >= next_step:
                last = pipe.full_step(t)
                next_step += STRIDE_SEC
                c = last["cues"]
                print(f"[{t:6.1f}s] emo={_fmt(c['emotion'])} ges={_fmt(c['gesture'])} "
                      f"mot={_fmt(c['motion'])} ctx={c['context_full']} | "
                      f"{last['intent'] or '--'} -> {last['action'] or '--'} p={last['confidence']:.2f}"
                      f"{' EMERGENCY' if last['emergency'] else ''}", flush=True)
                if jf:
                    jf.write(json.dumps(last) + "\n")
            if not a.no_display:
                view = frame if (a.no_mirror or not is_cam) else cv2.flip(frame, 1)
                cv2.imshow(win, compose(view, last, size, not a.hide_motion))
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        win, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    size = _screen_size() if fullscreen else (1280, 720)
                    if not fullscreen:
                        cv2.resizeWindow(win, *size)
            if a.max_seconds and t > a.max_seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        close()
        if jf:
            jf.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
