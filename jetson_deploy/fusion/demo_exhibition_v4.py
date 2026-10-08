"""Exhibition demo (v4): live camera -> what the robot senses -> what it
understands -> how it responds. Same pipeline as live_laptop_v4.py, with a
presentation-grade full-screen interface.

    .venv\\Scripts\\python.exe jetson_deploy\\fusion\\demo_exhibition_v4.py
    ... --source 0 --title "My Project" --subtitle "Team / University"
    ... --source clip.mp4          # playback of a recording
Keys: q/Esc quit, f toggle full screen.

Decision policy on screen: an action is COMMITTED and held for HOLD_SEC (5 s).
While it is held the system keeps collecting the fused intent probabilities;
when the hold ends, the next action is chosen from their average over the whole
window (through the same confidence-gated policy as the streaming pipeline).
An F02 emergency is the one exception: it interrupts the hold immediately.
"""
import argparse
import json
import math
import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import policy_v4                                                    # noqa: E402
from live_laptop_v4 import (INTENT_NAMES, STRIDE_SEC, LivePipeline,  # noqa: E402
                            _screen_size, open_source)

HOLD_SEC, NO_PERSON_GRACE, FIRST_MIN_STEPS = 5.0, 2.0, 3
CW, CH = 1920, 1080                          # design canvas; resized to screen

BG_TOP, BG_BOT = (10, 14, 30), (22, 30, 58)
CARD, CARD_DIM = (26, 34, 64), (19, 25, 47)
TRACK = (44, 54, 92)
WHITE, SOFT, MUTED = (245, 247, 255), (170, 180, 210), (105, 116, 150)
RED = (255, 84, 96)
INTENT_COL = {
    "F01": (72, 214, 140), "F02": (255, 84, 96), "F03": (90, 160, 255),
    "F04": (70, 210, 230), "F05": (150, 160, 190), "F06": (255, 150, 80),
    "F07": (255, 190, 70), "F08": (190, 130, 255), "F10": (130, 140, 255),
}
CUE_COL = {"emotion": (255, 140, 170), "gesture": (110, 190, 255),
           "context": (120, 230, 190), "motion": (255, 200, 110)}

_FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        names = (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"])
        for n in names:
            try:
                _FONTS[key] = ImageFont.truetype(n, size)
                break
            except OSError:
                continue
        else:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


def mix(a, b, k):
    return tuple(int(a[i] * (1 - k) + b[i] * k) for i in range(3))


def make_background():
    g = np.linspace(0, 1, CH)[:, None, None]
    top, bot = np.array(BG_TOP), np.array(BG_BOT)
    arr = (top * (1 - g) + bot * g).astype(np.uint8)
    arr = np.repeat(arr, CW, axis=1)
    return Image.fromarray(arr, "RGB")


def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def bar(d, x, y, w, h, frac, col):
    rrect(d, (x, y, x + w, y + h), h // 2, fill=TRACK)
    fw = int(w * max(0.0, min(1.0, frac)))
    if fw >= h:
        rrect(d, (x, y, x + fw, y + h), h // 2, fill=col)


def pretty(s):
    return s.replace("_", " ").capitalize()


def clip_text(d, txt, fnt, maxw):
    if d.textlength(txt, font=fnt) <= maxw:
        return txt
    while txt and d.textlength(txt + "…", font=fnt) > maxw:
        txt = txt[:-1]
    return txt + "…"


_DEC_KEYS = ("intent", "action", "action_text", "confidence", "emergency", "fallback")


class Hold:
    """Commit an action, hold it HOLD_SEC, then pick the next one from the
    mean intent probabilities gathered during the hold."""

    def __init__(self):
        self.cur = None                      # committed decision (dict)
        self.t_commit = 0.0
        self.acc = np.zeros(len(policy_v4.INTENTS))
        self.n = 0
        self.last_person = -1e9
        self.ctx = "unknown"
        self.events = []                     # commits, for logging

    def _mean_decision(self):
        d = policy_v4.decide(self.acc / max(self.n, 1), context_label=self.ctx)
        return {"intent": d.intent, "action": d.action,
                "action_text": policy_v4.ACTIONS[d.action],
                "confidence": d.confidence, "emergency": d.emergency,
                "fallback": d.fallback}

    def _commit(self, dec, now, why):
        self.cur, self.t_commit = dec, now
        self.acc[:] = 0
        self.n = 0
        self.events.append({"t": round(now, 2), "why": why, **dec})

    def update(self, r, now):
        if r is None:
            return
        if r["intent"] is not None:
            self.last_person = now
            self.ctx = r["cues"]["context_full"]
            self.acc += np.asarray(r["probs"])
            self.n += 1
            if self.cur is None:
                if r["emergency"]:
                    self._commit({k: r[k] for k in _DEC_KEYS}, now, "emergency")
                elif self.n >= FIRST_MIN_STEPS:           # ~0.8 s of evidence
                    self._commit(self._mean_decision(), now, "first")
            elif r["emergency"] and not self.cur["emergency"]:
                self._commit({k: r[k] for k in _DEC_KEYS}, now, "emergency")
            elif now - self.t_commit >= HOLD_SEC:
                self._commit(self._mean_decision(), now, "window")
        elif self.cur is not None and now - self.last_person > NO_PERSON_GRACE:
            self.cur = None
            self.acc[:] = 0
            self.n = 0

    def remaining(self, now):
        return max(0.0, HOLD_SEC - (now - self.t_commit)) if self.cur else 0.0

    def preview(self):
        return self._mean_decision() if (self.cur and self.n >= 3) else None


class Ui:
    def __init__(self, title, subtitle):
        self.title, self.subtitle = title, subtitle
        self.live = True                     # False for a recorded clip
        self.bg = make_background()
        self.smooth = {}
        self.last_intent, self.t_change = None, 0.0
        self.vmask = None

    def _smooth(self, key, target):
        cur = self.smooth.get(key, 0.0)
        if cur == 0.0:
            cur = target
        self.smooth[key] = cur + (target - cur) * 0.25
        return self.smooth[key]

    def _video(self, img, frame, box):
        x, y, w, h = box
        fh, fw = frame.shape[:2]
        sc = max(w / fw, h / fh)             # cover-crop
        rs = cv2.resize(frame, (int(fw * sc) + 1, int(fh * sc) + 1))
        ox, oy = (rs.shape[1] - w) // 2, (rs.shape[0] - h) // 2
        crop = cv2.cvtColor(rs[oy:oy + h, ox:ox + w], cv2.COLOR_BGR2RGB)
        if self.vmask is None or self.vmask.size != (w, h):
            m = Image.new("L", (w, h), 0)
            ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), 28, fill=255)
            self.vmask = m
        img.paste(Image.fromarray(crop), (x, y), self.vmask)

    def render(self, frame, r, hold, now):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        NEUTRAL = (110, 125, 170)
        cur = hold.cur                                   # held action (5 s)
        live = r["intent"] if r else None                # live intent, not held
        live_emerg = bool(live and r["emergency"])
        emerg = bool(cur and cur["emergency"]) or live_emerg
        idle = live is None and cur is None
        acc = INTENT_COL.get(live, NEUTRAL) if live else NEUTRAL       # intent colour
        acc_act = INTENT_COL.get(cur["intent"], NEUTRAL) if cur else NEUTRAL
        pulse = 0.5 + 0.5 * math.sin(now * 6.0)

        if live != self.last_intent:
            self.last_intent, self.t_change = live, now
        flash = max(0.0, 1.0 - (now - self.t_change) / 0.7) if live else 0.0

        # header
        d.text((48, 22), self.title.upper(), font=font(38, True), fill=WHITE)
        d.text((50, 68), self.subtitle, font=font(23), fill=SOFT)
        tag = "● LIVE" if self.live else "RECORDED CLIP"
        tw = d.textlength(tag, font=font(24, True))
        tag_col = (RED if pulse > 0.5 else (170, 60, 70)) if self.live else SOFT
        d.text((CW - 48 - tw, 26), tag, font=font(24, True), fill=tag_col)
        if self.live:
            d.text((CW - 48, 64), "Q quit   F full screen", font=font(18), fill=MUTED, anchor="ra")
        x = 900
        for i, (lbl, col) in enumerate((("SENSE", (110, 190, 255)),
                                        ("UNDERSTAND", (255, 190, 70)),
                                        ("ACT", (72, 214, 140)))):
            d.text((x, 34), lbl, font=font(22, True), fill=col)
            x += int(d.textlength(lbl, font=font(22, True))) + 16
            if i < 2:
                d.text((x, 30), "›", font=font(30, True), fill=MUTED)
                x += 32

        # camera (large)
        vb = (30, 112, 1236, 927)
        bw = 6 + int(6 * pulse) if emerg else 4
        rrect(d, (vb[0] - bw, vb[1] - bw, vb[0] + vb[2] + bw, vb[1] + vb[3] + bw),
              32, outline=RED if emerg else acc, width=bw)
        self._video(img, frame, vb)
        d = ImageDraw.Draw(img)
        if idle:
            rrect(d, (vb[0] + 24, vb[1] + vb[3] - 84, vb[0] + 580, vb[1] + vb[3] - 24),
                  22, fill=(10, 14, 30))
            d.text((vb[0] + 48, vb[1] + vb[3] - 74), "Step in front of the camera",
                   font=font(30, True), fill=mix(SOFT, WHITE, pulse))
        if emerg:
            msg = "EMERGENCY DETECTED — ROBOT HALTING"
            mw = int(d.textlength(msg, font=font(30, True))) + 60
            bx = vb[0] + (vb[2] - mw) // 2
            rrect(d, (bx, vb[1] + 20, bx + mw, vb[1] + 76), 28, fill=RED)
            d.text((bx + mw // 2, vb[1] + 48), msg, font=font(30, True), fill=WHITE, anchor="mm")

        # sensing cards
        x0, cw = 1300, 590
        d.text((x0, 116), "WHAT THE ROBOT SEES", font=font(21, True), fill=MUTED)
        cues = r["cues"] if r else {"emotion": None, "gesture": None,
                                    "motion": None, "context_full": None}
        ctx = cues.get("context_full")
        rows = [("emotion", "EMOTION", cues["emotion"]),
                ("gesture", "GESTURE", cues["gesture"]),
                ("motion", "MOTION", cues["motion"]),
                ("context", "SCENE", (ctx, None) if ctx and ctx != "unknown" else None)]
        ch_, y = 94, 150
        for key, label, val in rows:
            has = val is not None
            col = CUE_COL[key]
            rrect(d, (x0, y, x0 + cw, y + ch_), 20, fill=CARD if has else CARD_DIM)
            rrect(d, (x0, y + 16, x0 + 8, y + ch_ - 16), 4, fill=col if has else TRACK)
            d.text((x0 + 30, y + 10), label, font=font(18, True), fill=col if has else MUTED)
            if has:
                d.text((x0 + 30, y + 34), pretty(val[0]), font=font(38, True), fill=WHITE)
                if val[1] is not None:
                    conf = self._smooth(key, val[1])
                    bar(d, x0 + 350, y + ch_ // 2 - 6, 140, 12, conf, col)
                    d.text((x0 + cw - 24, y + ch_ // 2 - 15), f"{int(conf * 100)}%",
                           font=font(22, True), fill=SOFT, anchor="ra")
            else:
                self.smooth[key] = 0.0
            y += ch_ + 12

        # understanding
        py = y + 6
        d.text((x0, py), "WHAT THE ROBOT UNDERSTANDS", font=font(21, True), fill=MUTED)
        py += 32
        ph = 138
        fill = mix(CARD, acc, 0.35 * flash + (0.25 * pulse if live_emerg else 0))
        rrect(d, (x0, py, x0 + cw, py + ph), 24, fill=fill, outline=acc if live else TRACK, width=3)
        if live:
            rrect(d, (x0 + 22, py + 22, x0 + 22 + 118, py + 22 + 94), 20, fill=acc)
            d.text((x0 + 22 + 59, py + 69), live, font=font(48, True), fill=(15, 20, 40), anchor="mm")
            lines = textwrap.wrap(INTENT_NAMES[live], 21)[:2]
            fs = 38 if len(lines) == 1 else 32
            ty = py + ph // 2 - (len(lines) * (fs + 4)) // 2
            for ln in lines:
                d.text((x0 + 164, ty), ln, font=font(fs, True), fill=WHITE)
                ty += fs + 6
        else:
            d.text((x0 + 30, py + ph // 2), "Waiting for a person…", font=font(36, True),
                   fill=mix(MUTED, SOFT, pulse), anchor="lm")

        # response (held action)
        ry = py + ph + 20
        d.text((x0, ry), "HOW THE ROBOT RESPONDS", font=font(21, True), fill=MUTED)
        ry += 32
        rh = vb[1] + vb[3] - ry
        rrect(d, (x0, ry, x0 + cw, ry + rh), 24, fill=CARD if cur else CARD_DIM,
              outline=acc_act if cur else TRACK, width=3 if emerg else 1)
        if cur:
            head = cur["action"] + ("   EMERGENCY STOP" if cur["emergency"] else "")
            d.text((x0 + 28, ry + 12), head, font=font(30, True), fill=RED if cur["emergency"] else acc_act)
            for i, ln in enumerate(textwrap.wrap(cur["action_text"], 36)[:2]):
                d.text((x0 + 28, ry + 50 + i * 36), ln, font=font(28), fill=WHITE)
            rem = hold.remaining(now)
            d.text((x0 + 28, ry + rh - 112), "ACTION HELD", font=font(17, True), fill=MUTED)
            d.text((x0 + cw - 28, ry + rh - 112),
                   f"next decision in {math.ceil(rem)} s" if rem > 0 else "deciding…",
                   font=font(17, True), fill=SOFT, anchor="ra")
            bar(d, x0 + 28, ry + rh - 86, cw - 56, 10, 1 - rem / HOLD_SEC, acc_act)
            pv = hold.preview()
            if pv is None:
                txt = "Observing…"
            elif pv["action"] == cur["action"]:
                txt = "Keep the current action"
            else:
                txt = f"{pv['action']} · {pv['action_text']}"
            d.text((x0 + 28, ry + rh - 66), "NEXT ACTION (FORMING)", font=font(17, True), fill=MUTED)
            d.text((x0 + 28, ry + rh - 40), clip_text(d, txt, font(22), cw - 56),
                   font=font(22), fill=SOFT)
        else:
            self.smooth["intent"] = 0.0

        if emerg:
            e = int(6 + 8 * pulse)
            d.rectangle((0, 0, CW - 1, CH - 1), outline=RED, width=e)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def fit_to_screen(canvas, size):
    W, H = size
    if (W, H) == (CW, CH):
        return canvas
    sc = min(W / CW, H / CH)
    nw, nh = int(CW * sc), int(CH * sc)
    out = np.zeros((H, W, 3), np.uint8)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(canvas, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def main():
    global HOLD_SEC
    ap = argparse.ArgumentParser(description="Exhibition HRI demo (v4)")
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--title", default="PerceptAI")
    ap.add_argument("--subtitle", default="Robots that understand people, not just signals")
    ap.add_argument("--hold", type=float, default=HOLD_SEC, help="seconds an action is held")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--json", default=None, help="log committed actions as JSONL")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--save-frame", default=None, help="write one UI frame (PNG) after 8 s, for testing")
    ap.add_argument("--record", default=None,
                    help="write the rendered UI to this .mp4 (one UI frame per source frame)")
    ap.add_argument("--no-display", action="store_true", help="no window (use with --record)")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--tau-emergency", type=float, default=None)
    a = ap.parse_args()
    HOLD_SEC = a.hold

    pipe = LivePipeline(tau=a.tau, tau_emergency=a.tau_emergency)
    ui, hold = Ui(a.title, a.subtitle), Hold()
    read, close = open_source(a.source)
    jf = open(a.json, "a", encoding="utf-8") if a.json else None
    is_cam = str(a.source).isdigit()
    ui.live = is_cam
    fps = 30.0
    if not is_cam:
        cap = cv2.VideoCapture(a.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    writer = None
    if a.record:
        writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), fps, (CW, CH))
    win = "HRI exhibition demo"
    fullscreen = not a.windowed
    size = _screen_size() if fullscreen else (1280, 720)
    if not a.no_display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(win, *size)

    t0 = time.time()
    next_step, last, saved, n_ev, n_frames = STRIDE_SEC, None, False, 0, 0
    try:
        while True:
            frame = read()
            if frame is None:
                break
            # recorded clips run on their own timeline, so the result doesn't
            # depend on how fast this machine processes them
            t = n_frames / fps if not is_cam else time.time() - t0
            n_frames += 1
            pipe.push_frame(frame, t)
            if t >= next_step:
                last = pipe.full_step(t)
                next_step += STRIDE_SEC
                hold.update(last, t)
            while n_ev < len(hold.events):
                ev = hold.events[n_ev]
                n_ev += 1
                print(f"[{ev['t']:6.1f}s] COMMIT ({ev['why']}) {ev['intent']} -> {ev['action']} "
                      f"p={ev['confidence']:.2f}", flush=True)
                if jf:
                    jf.write(json.dumps(ev) + "\n")
            view = frame if (a.no_mirror or not is_cam) else cv2.flip(frame, 1)
            canvas = ui.render(view, last, hold, t)
            if a.save_frame and not saved and t > 8:
                cv2.imwrite(a.save_frame, canvas)
                saved = True
            if writer is not None:
                writer.write(canvas)
            if a.no_display:
                if a.max_seconds and t > a.max_seconds:
                    break
                continue
            cv2.imshow(win, fit_to_screen(canvas, size))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
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
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
