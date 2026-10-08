"""Rebuild the final pitch video: replace the 0:40.97-1:30.13 real-world section
with current PerceptAI renders, caption the live Jetson section, add quiet music.
Total length and every other frame of the original stay identical.

    .venv\\Scripts\\python.exe pitch_video\\build_final_v2.py D:\\FInal.mp4 D:\\FInal_v2.mp4
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FPS = 30
SEC_START, SEC_END = 1229, 2704          # replaced section, in original frames
LIVE = (109.033, 127.6)                   # live Jetson footage
NOTE = (123.9, 127.6)                     # F02 shown with the lagging action panel

CLIPS = [                                 # new order for the real-world section
    "real_world/2_kitchen_wave_greeting.mp4",
    "real_world/0_classroom_raise_hand_help.mp4",
    "real_world/3_classroom_sad_thumbs_down_help.mp4",
    "perceptai_output/spare_discouraged_validation_clip.mp4",
    "perceptai_output/01_genuine_thumbs_up.mp4",
    "real_world/5_hospital_ai_thumbs_up_frustration.mp4",
    "real_world/optional_museum_ai_thumbs_up_frustration.mp4",
    "real_world/6_kitchen_emergency.mp4",
]


def duration(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout
    return float(out.strip())


def main(src, dst):
    total = duration(src)
    work = HERE / "_build"
    work.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(HERE / "make_music.py"), "--seconds", f"{total:.3f}",
                    "--out", str(work / "music.wav")], check=True)
    subprocess.run([sys.executable, str(HERE / "make_captions.py"), str(work)], check=True)

    paths = [HERE / c for c in CLIPS]
    native = [round(duration(p) * FPS) for p in paths]
    slot = SEC_END - SEC_START
    extra = slot - sum(native)
    assert extra >= 0, f"clips are {-extra} frames too long for the section"
    frames = [n + extra // len(native) for n in native]
    frames[-1] += slot - sum(frames)

    t = 0.0
    print("real-world section timeline:")
    for c, f in zip(CLIPS, frames):
        s = SEC_START / FPS + t
        print(f"  {int(s // 60)}:{s % 60:05.2f}  {Path(c).stem}  ({f / FPS:.1f} s)")
        t += f / FPS

    inputs = ["-i", src]
    for p in paths:
        inputs += ["-i", str(p)]
    k = len(paths) + 1
    inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{total:.3f}", "-i", str(work / "cap_live.png"),
               "-loop", "1", "-framerate", str(FPS), "-t", f"{total:.3f}", "-i", str(work / "cap_note.png"),
               "-i", str(work / "music.wav")]

    norm = "fps=30,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p"
    fc = [f"[0:v]{norm},trim=start_frame=0:end_frame={SEC_START},setpts=PTS-STARTPTS[a]",
          f"[0:v]{norm},trim=start_frame={SEC_END},setpts=PTS-STARTPTS[c]"]
    for i, f in enumerate(frames, 1):
        fc.append(f"[{i}:v]{norm},tpad=stop_mode=clone:stop_duration=3,"
                  f"trim=end_frame={f},setpts=PTS-STARTPTS[b{i}]")
    fc.append("[a]" + "".join(f"[b{i}]" for i in range(1, len(frames) + 1)) +
              f"[c]concat=n={len(frames) + 2}:v=1:a=0[v0]")
    fc.append(f"[{k}:v]format=rgba,fade=in:st={LIVE[0]}:d=0.5:alpha=1,"
              f"fade=out:st={LIVE[1] - 0.4}:d=0.4:alpha=1[cl]")
    fc.append(f"[{k + 1}:v]format=rgba,fade=in:st={NOTE[0]}:d=0.4:alpha=1,"
              f"fade=out:st={NOTE[1] - 0.4}:d=0.4:alpha=1[cn]")
    fc.append(f"[v0][cl]overlay=0:0:enable='between(t,{LIVE[0]},{LIVE[1]})'[v1]")
    fc.append(f"[v1][cn]overlay=0:0:enable='between(t,{NOTE[0]},{NOTE[1]})',format=yuv420p[v]")
    fc.append(f"[{k + 2}:a]loudnorm=I=-30:TP=-6:LRA=7,atrim=0:{total:.3f}[m]")

    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(fc),
           "-map", "[v]", "-map", "[m]", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{total:.3f}", dst]
    subprocess.run(cmd, check=True)
    print("wrote", dst, f"({duration(dst):.2f} s, original {total:.2f} s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
