"""Render the 'how PerceptAI decides' animated diagram (1920x1080, 30 fps).

    .venv\\Scripts\\python.exe pitch_video\\make_fusion_diagram.py
    ... --motion-awareness     # draw motion as a dashed 'awareness' link
"""
import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, DUR = 1920, 1080, 30, 9.5
BG_TOP, BG_BOT = (10, 14, 30), (22, 30, 58)
CARD, TRACK = (26, 34, 64), (44, 54, 92)
WHITE, SOFT, MUTED = (245, 247, 255), (170, 180, 210), (105, 116, 150)
TEAL, AMBER = (70, 210, 230), (255, 190, 70)

CUES = [  # key, label, value, confidence, colour
    ("emotion", "EMOTION", "Angry", 0.88, (255, 140, 170)),
    ("gesture", "GESTURE", "Thumbs up", 0.92, (110, 190, 255)),
    ("motion", "MOTION", "Standing", 0.90, (255, 200, 110)),
    ("context", "SCENE", "Kitchen", None, (120, 230, 190)),
]

_F = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _F:
        try:
            _F[k] = ImageFont.truetype("segoeuib.ttf" if bold else "segoeui.ttf", size)
        except OSError:
            _F[k] = ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    return _F[k]


def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def phase(t, start, dur):
    return ease((t - start) / dur)


def mix(a, b, k):
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


def bezier(p0, p1, p2, p3, s):
    u = 1 - s
    return (u**3 * p0[0] + 3 * u * u * s * p1[0] + 3 * u * s * s * p2[0] + s**3 * p3[0],
            u**3 * p0[1] + 3 * u * u * s * p1[1] + 3 * u * s * s * p2[1] + s**3 * p3[1])


def curve(a, b):
    mx = (a[0] + b[0]) / 2
    return a, (mx, a[1]), (mx, b[1]), b


def draw_curve(d, pts, col, width, upto=1.0, dashed=False):
    n = 60
    last = None
    for i in range(int(n * upto) + 1):
        p = bezier(*pts, i / n)
        if last is not None and (not dashed or (i // 3) % 2 == 0):
            d.line([last, p], fill=col, width=width)
        last = p


def background():
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    for y in range(H):
        d.line([(0, y), (W, y)], fill=mix(BG_TOP, BG_BOT, y / H))
    return img


def card(d, box, accent, label, value, conf, alpha):
    if alpha <= 0:
        return
    x0, y0, x1, y1 = box
    fill = mix(BG_TOP, CARD, alpha)
    d.rounded_rectangle(box, 22, fill=fill, outline=mix(BG_TOP, accent, alpha * 0.8), width=2)
    d.rounded_rectangle((x0, y0 + 18, x0 + 8, y1 - 18), 4, fill=mix(BG_TOP, accent, alpha))
    d.text((x0 + 32, y0 + 16), label, font=font(20, True), fill=mix(BG_TOP, accent, alpha))
    d.text((x0 + 32, y0 + 44), value, font=font(40, True), fill=mix(BG_TOP, WHITE, alpha))
    if conf is not None:
        bx, by, bw = x1 - 190, (y0 + y1) // 2 - 6, 110
        d.rounded_rectangle((bx, by, bx + bw, by + 12), 6, fill=mix(BG_TOP, TRACK, alpha))
        fw = int(bw * conf * alpha)
        if fw > 12:
            d.rounded_rectangle((bx, by, bx + fw, by + 12), 6, fill=mix(BG_TOP, accent, alpha))
        d.text((x1 - 24, by - 10), f"{int(conf * 100)}%", font=font(22, True),
               fill=mix(BG_TOP, SOFT, alpha), anchor="ra")


def frame(t, bg, motion_awareness):
    img = bg.copy()
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    g = ImageDraw.Draw(glow)
    d = ImageDraw.Draw(img)

    # header
    a = phase(t, 0.0, 0.6)
    d.text((80, 50), "HOW PERCEPTAI DECIDES", font=font(34, True), fill=mix(BG_TOP, WHITE, a))
    d.text((82, 96), "Example: a sarcastic thumbs-up", font=font(24), fill=mix(BG_TOP, SOFT, a))

    fus = (800, 400, 1120, 640)                      # fusion box
    fus_c = ((fus[0] + fus[2]) / 2, (fus[1] + fus[3]) / 2)
    intent_box = (1300, 290, 1840, 450)
    action_box = (1300, 600, 1840, 800)

    # cue cards + links
    for i, (key, label, value, conf, col) in enumerate(CUES):
        y0 = 210 + i * 180
        box = (80, y0, 600, y0 + 140)
        ca = phase(t, 0.4 + i * 0.35, 0.5)
        card(d, box, col, label, value, conf, ca)
        start = (600, y0 + 70)
        end = (fus[0], fus_c[1] - 75 + i * 50)
        pts = curve(start, end)
        dashed = motion_awareness and key == "motion"
        grow = phase(t, 2.2 + i * 0.12, 0.9)
        if grow > 0:
            draw_curve(d, pts, mix(BG_TOP, col, 0.55), 3, grow, dashed)
            if dashed and grow > 0.6:
                mid = bezier(*pts, 0.5)
                d.text((mid[0] - 40, mid[1] - 40), "awareness", font=font(18, True),
                       fill=mix(BG_TOP, col, 0.8), anchor="mm")
        # travelling pulses
        if 3.0 < t < 5.2 and not dashed:
            for k in range(3):
                s = ((t - 3.0) * 0.9 + k / 3 + i * 0.07) % 1.0
                p = bezier(*pts, s)
                r = 9
                g.ellipse((p[0] - r * 2, p[1] - r * 2, p[0] + r * 2, p[1] + r * 2), fill=col)
                d.ellipse((p[0] - r / 2, p[1] - r / 2, p[0] + r / 2, p[1] + r / 2), fill=WHITE)

    # fusion node
    fa = phase(t, 2.0, 0.7)
    if fa > 0:
        pulse = 0.5 + 0.5 * math.sin((t - 3.0) * 7) if 3.0 < t < 5.4 else 0.0
        lit = phase(t, 3.2, 1.8)
        edge = mix(TRACK, TEAL, max(lit, 0.35) * fa)
        if lit > 0:
            g.rounded_rectangle(fus, 40, fill=mix((0, 0, 0), TEAL, 0.5 * lit + 0.3 * pulse))
        d.rounded_rectangle(fus, 40, fill=mix(BG_TOP, (30, 48, 80), fa), outline=edge, width=4)
        d.text((fus_c[0], fus_c[1] - 30), "FUSION", font=font(46, True),
               fill=mix(BG_TOP, WHITE, fa), anchor="mm")
        d.text((fus_c[0], fus_c[1] + 22), "combines every cue", font=font(22),
               fill=mix(BG_TOP, SOFT, fa), anchor="mm")
        d.text((fus_c[0], fus_c[1] + 56), "on-device AI", font=font(20, True),
               fill=mix(BG_TOP, TEAL, fa * 0.9), anchor="mm")

    # fusion -> intent
    ia = phase(t, 5.2, 0.6)
    p1 = curve((fus[2], fus_c[1] - 40), (intent_box[0], (intent_box[1] + intent_box[3]) / 2))
    if phase(t, 4.9, 0.5) > 0:
        draw_curve(d, p1, mix(BG_TOP, AMBER, 0.7), 4, phase(t, 4.9, 0.5))
    if ia > 0:
        x0, y0, x1, y1 = intent_box
        g.rounded_rectangle(intent_box, 24, fill=mix((0, 0, 0), AMBER, 0.15 * ia))
        d.rounded_rectangle(intent_box, 24, fill=mix(BG_TOP, CARD, ia),
                            outline=mix(BG_TOP, AMBER, ia), width=3)
        d.text((x0 + 32, y0 + 18), "INTENT", font=font(20, True), fill=mix(BG_TOP, AMBER, ia))
        d.rounded_rectangle((x0 + 32, y0 + 56, x0 + 142, y0 + 132), 16, fill=mix(BG_TOP, AMBER, ia))
        d.text((x0 + 87, y0 + 94), "F07", font=font(38, True), fill=mix(BG_TOP, (15, 20, 40), ia),
               anchor="mm")
        d.text((x0 + 166, y0 + 70), "Frustration", font=font(40, True), fill=mix(BG_TOP, WHITE, ia))
        d.text((x1 - 28, y0 + 22), "97%", font=font(24, True), fill=mix(BG_TOP, SOFT, ia), anchor="ra")

    # intent -> action
    aa = phase(t, 6.6, 0.6)
    ix = (intent_box[0] + intent_box[2]) / 2
    grow = phase(t, 6.2, 0.45)
    if grow > 0:
        y_end = intent_box[3] + (action_box[1] - intent_box[3]) * grow
        d.line([(ix, intent_box[3]), (ix, y_end)], fill=mix(BG_TOP, AMBER, 0.7), width=4)
        if grow >= 1:
            d.polygon([(ix - 12, action_box[1] - 16), (ix + 12, action_box[1] - 16), (ix, action_box[1])],
                      fill=mix(BG_TOP, AMBER, 0.9))
    if aa > 0:
        x0, y0, x1, y1 = action_box
        g.rounded_rectangle(action_box, 24, fill=mix((0, 0, 0), AMBER, 0.15 * aa))
        d.rounded_rectangle(action_box, 24, fill=mix(BG_TOP, CARD, aa),
                            outline=mix(BG_TOP, AMBER, aa), width=3)
        d.text((x0 + 32, y0 + 18), "ROBOT ACTION", font=font(20, True), fill=mix(BG_TOP, AMBER, aa))
        d.text((x0 + 32, y0 + 54), "A08", font=font(40, True), fill=mix(BG_TOP, AMBER, aa))
        d.text((x0 + 32, y0 + 110), "Calm, de-escalating response", font=font(30, True),
               fill=mix(BG_TOP, WHITE, aa))

    # footer takeaway
    fa2 = phase(t, 7.6, 0.8)
    if fa2 > 0:
        d.text((W // 2, 960), "Same gesture, different meaning: PerceptAI reads the whole person.",
               font=font(30, True), fill=mix(BG_TOP, WHITE, fa2), anchor="mm")

    glow = glow.filter(ImageFilter.GaussianBlur(28))
    out = np.minimum(np.asarray(img, np.uint16) + np.asarray(glow, np.uint16), 255)
    return Image.fromarray(out.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-awareness", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--still", default=None, help="write one PNG at t=9 s instead of the video")
    a = ap.parse_args()
    bg = background()
    if a.still:
        frame(9.0, bg, a.motion_awareness).save(a.still)
        return
    out = a.out or str(Path(__file__).parent / "graphics" / (
        "fusion_diagram_motion_awareness.mp4" if a.motion_awareness else "fusion_diagram.mp4"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    for i in range(int(DUR * FPS)):
        proc.stdin.write(frame(i / FPS, bg, a.motion_awareness).tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
