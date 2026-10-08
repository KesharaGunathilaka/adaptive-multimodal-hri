"""Caption overlays (transparent 1920x1080 PNGs) for the live Jetson section."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
TEAL, RED, WHITE, SOFT = (70, 210, 230), (255, 84, 96), (245, 247, 255), (190, 200, 225)


def font(size, bold=False):
    return ImageFont.truetype("segoeuib.ttf" if bold else "segoeui.ttf", size)


def pill(lines, dot, y0):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fonts = [font(30, True)] + [font(26)] * (len(lines) - 1)
    width = max(d.textlength(s, font=f) for s, f in zip(lines, fonts))
    h = 28 + sum(f.size + 12 for f in fonts)
    x0 = 48
    d.rounded_rectangle((x0, y0, x0 + width + 96, y0 + h), 26, fill=(8, 12, 26, 215))
    d.ellipse((x0 + 28, y0 + 27, x0 + 44, y0 + 43), fill=dot)
    y = y0 + 14
    for s, f in zip(lines, fonts):
        d.text((x0 + 64, y), s, font=f, fill=WHITE if f is fonts[0] else SOFT)
        y += f.size + 12
    return img


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pill(["LIVE ON A JETSON EDGE DEVICE  ·  FINAL-YEAR EVALUATION"], TEAL, 40).save(out / "cap_live.png")
    pill(["Earlier prototype: its action panel refreshed every 10 s.",
          "PerceptAI now stops the robot instantly on an emergency."], RED, 130).save(out / "cap_note.png")


if __name__ == "__main__":
    main(sys.argv[1])
