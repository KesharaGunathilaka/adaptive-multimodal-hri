"""Build the looping exhibition video: corrected real-world clips, neural
voice-over, burned-in subtitles (+ .srt), ducked background music, and a
fade through black at the loop point.

    .venv\\Scripts\\python.exe pitch_video\\build_loop_video.py "D:\\Loop Video.mp4" "D:\\Loop Video_final.mp4" --tts <path-to-edge-tts.exe>
"""
import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FPS, W, H, SR = 30, 1920, 1080, 48000
VOICE = "en-US-AvaNeural"
SEC_START, SEC_END = 1348, 2740           # real-world section, original frames
CLIPS = [
    ("real_world/2_kitchen_wave_greeting.mp4",
     "Now, real recordings. A wave: a friendly greeting."),
    ("real_world/0_classroom_raise_hand_help.mp4",
     "A raised hand in class: he needs help, so it offers guidance."),
    ("real_world/3_classroom_sad_thumbs_down_help.mp4",
     "A sad thumbs-down at his own work: it offers support."),
    ("perceptai_output/spare_discouraged_validation_clip.mp4",
     "No gesture, just a discouraged face: it offers encouragement."),
    ("perceptai_output/01_genuine_thumbs_up.mp4",
     "A smile and a thumbs-up: genuine approval."),
    ("real_world/5_hospital_ai_thumbs_up_frustration.mp4",
     "The same thumbs-up with disgust, in a hospital,"),
    ("real_world/optional_museum_ai_thumbs_up_frustration.mp4",
     "or with anger, in a museum: frustration. Same gesture, different meaning."),
    ("real_world/6_kitchen_emergency.mp4",
     "And fear, both hands up: emergency. The robot stops instantly."),
]
# (start seconds, subtitle text) outside the real-world section
LINES_BEFORE = [
    (0.8, "Robots already work alongside us: in hospitals, on factory floors, in our homes and offices."),
    (8.0, "They can see us. But can they understand us?"),
    (13.0, "Today's robots detect signals. They don't understand people."),
    (17.0, "Meet PerceptAI: robots that understand people, not just signals."),
    (21.6, "With one ordinary camera, it reads a person's emotion, gesture, movement, and the scene around them."),
    (28.3, "Take a sarcastic thumbs-up. Our fusion model weighs every cue together, and reads frustration, not approval."),
    (35.2, "It all runs on a compact edge device inside the robot. No cloud needed."),
    (40.4, "A raised hand? The robot knows he needs help."),
]
LINES_AFTER = [
    (91.6, "We don't build robots. We sell the brain."),
    (95.0, "Licensed per robot, shipped as an edge kit, and always learning. From Sri Lanka, to the world."),
    (101.4, "In hospitals, it notices a patient who needs care. In classrooms, the student who needs help."),
    (108.4, "Robots detect signals. PerceptAI understands people."),
]


def spoken(text):
    return text.replace("PerceptAI", "Percept A.I.")


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def duration(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout
    return float(out.strip())


def decode_audio(path, channels):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac",
                          str(channels), "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, channels).copy()


def build_base(src, dst, frames, total):
    norm = ("fps=30,scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p")
    inputs = ["-i", src]
    for c, _ in CLIPS:
        inputs += ["-i", str(HERE / c)]
    fc = [f"[0:v]{norm},trim=start_frame=0:end_frame={SEC_START},setpts=PTS-STARTPTS[a]",
          f"[0:v]{norm},trim=start_frame={SEC_END},setpts=PTS-STARTPTS[c]"]
    for i, f in enumerate(frames, 1):
        fc.append(f"[{i}:v]{norm},tpad=stop_mode=clone:stop_duration=3,"
                  f"trim=end_frame={f},setpts=PTS-STARTPTS[b{i}]")
    fc.append("[a]" + "".join(f"[b{i}]" for i in range(1, len(frames) + 1)) +
              f"[c]concat=n={len(frames) + 2}:v=1:a=0,"
              f"fade=in:st=0:d=0.5,fade=out:st={total - 0.8:.3f}:d=0.8[v]")
    run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(fc),
         "-map", "[v]", "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-t", f"{total:.3f}", dst])


def subtitle_chunks(text, start, dur, max_chars=58):
    import re
    phrases = [p.strip() for p in re.split(r"(?<=[:.?,])\s+", text) if p.strip()]
    parts, cur = [], ""
    for p in phrases:
        sentence_end = cur.endswith((".", "?")) and len(cur) >= 25
        if cur and (len(cur) + 1 + len(p) > max_chars or sentence_end):
            parts.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    parts.append(cur)
    out_parts = []
    for p in parts:
        if len(p) > max_chars:
            n = -(-len(p) // max_chars)
            out_parts += textwrap.wrap(p, -(-len(p) // n) + 4)
        else:
            out_parts.append(p)
    parts = out_parts
    total_chars = sum(len(p) for p in parts)
    t, out = start, []
    for p in parts:
        d = dur * len(p) / total_chars
        out.append((t, t + d, p))
        t += d
    return out


def render_sub(text, font, cx):
    img = Image.new("RGBA", (W, 120), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    tw = d.textlength(text, font=font)
    x0 = cx - tw / 2 - 28
    d.rounded_rectangle((x0, 20, x0 + tw + 56, 100), 18, fill=(8, 12, 26, 200))
    d.text((cx, 60), text, font=font, fill=(245, 247, 255, 255), anchor="mm")
    return np.asarray(img)


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--tts", required=True, help="path to the edge-tts executable")
    a = ap.parse_args()

    total = duration(a.src)
    work = HERE / "_build_loop"
    work.mkdir(exist_ok=True)

    # real-world section timing (same frame budget as the original section)
    native = [round(duration(HERE / c) * FPS) for c, _ in CLIPS]
    slot = SEC_END - SEC_START
    frames = [n + (slot - sum(native)) // len(native) for n in native]
    frames[-1] += slot - sum(frames)
    lines = list(LINES_BEFORE)
    t = SEC_START / FPS
    for (c, text), f in zip(CLIPS, frames):
        lines.append((t + 0.9, text))
        t += f / FPS
    lines += LINES_AFTER

    # voice-over
    voice = np.zeros(int(total * SR) + SR, np.float32)
    subs, last_end = [], 0.0
    for i, (start, text) in enumerate(lines):
        mp3 = work / f"line_{i:02d}.mp3"
        if not mp3.exists():
            run([a.tts, "--voice", VOICE, "--rate", "+4%", "--text", spoken(text),
                 "--write-media", str(mp3)], capture_output=True)
        clip = decode_audio(mp3, 1)[:, 0]
        nz = np.nonzero(np.abs(clip) > 0.01)[0]
        clip = clip[max(nz[0] - 800, 0): nz[-1] + 2400]            # trim silence
        start = max(start, last_end + 0.25)
        s = int(start * SR)
        voice[s:s + len(clip)] += clip
        d = len(clip) / SR
        last_end = start + d
        if last_end > total - 0.5:
            print(f"WARNING: line {i} ends at {last_end:.2f}s, past the video end")
        subs += subtitle_chunks(text, start, d)
        print(f"  {start:6.2f}-{last_end:6.2f}  {text}")
    voice = voice[:int(total * SR)]
    voice *= 0.89 / np.abs(voice).max()

    # music, ducked under the voice
    music_wav = work / "music.wav"
    run([sys.executable, str(HERE / "make_music.py"), "--seconds", f"{total:.3f}", "--out", str(music_wav)])
    music = decode_audio(music_wav, 2)[:len(voice)]
    env = np.convolve(np.abs(voice), np.ones(int(0.3 * SR)) / (0.3 * SR), mode="same")
    duck = 1.0 - 0.55 * np.clip(env / (env.max() * 0.25), 0, 1)
    k = int(0.25 * SR)
    duck = np.convolve(duck, np.ones(k) / k, mode="same")
    music_level = 10 ** (-24 / 20) / (np.sqrt(np.mean(music ** 2)) + 1e-9) * 0.35
    mix = voice[:, None] + music * music_level * duck[:, None]
    fade = np.clip(np.minimum(np.arange(len(mix)) / (0.5 * SR), (len(mix) - np.arange(len(mix))) / (0.8 * SR)), 0, 1)
    mix *= fade[:, None]
    mix *= 0.95 / np.abs(mix).max()
    mix_wav = work / "mix.wav"
    import wave
    with wave.open(str(mix_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())

    # subtitles sidecar
    srt = Path(a.dst).with_suffix(".srt")
    with open(srt, "w", encoding="utf-8") as f:
        for i, (s, e, text) in enumerate(subs, 1):
            f.write(f"{i}\n{srt_time(s)} --> {srt_time(e)}\n{text}\n\n")

    # video: base, then burn subtitles frame by frame
    base = work / "base.mp4"
    build_base(a.src, str(base), frames, total)
    font = ImageFont.truetype("seguisb.ttf", 42)
    cache = {}
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(base), "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(mix_wav),
                            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.dst],
                           stdin=subprocess.PIPE)
    n = 0
    y0 = H - 130
    while True:
        buf = dec.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        t = n / FPS
        cur = next((txt for s, e, txt in subs if s <= t < e), None)
        if cur:
            # inside the PerceptAI app clips, centre under the camera view
            cx = 648 if SEC_START / FPS <= t < SEC_END / FPS else W // 2
            key = (cur, cx)
            if key not in cache:
                cache[key] = render_sub(cur, font, cx)
            frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
            ov = cache[key]
            alpha = ov[:, :, 3:4].astype(np.float32) / 255
            region = frame[y0:y0 + 120].astype(np.float32)
            frame[y0:y0 + 120] = (region * (1 - alpha) + ov[:, :, :3] * alpha).astype(np.uint8)
            buf = frame.tobytes()
        enc.stdin.write(buf)
        n += 1
    enc.stdin.close()
    enc.wait()
    dec.wait()
    print("wrote", a.dst, f"({duration(a.dst):.2f} s) and", srt)


if __name__ == "__main__":
    main()
