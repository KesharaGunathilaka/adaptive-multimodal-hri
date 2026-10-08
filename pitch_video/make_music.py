"""Synthesize a quiet ambient background track (royalty-free, generated here).

    .venv\\Scripts\\python.exe pitch_video\\make_music.py --seconds 138.6 --out music.wav
"""
import argparse
import wave

import numpy as np

SR = 44100
CHORD_SEC = 6.0
# (bass, pad notes) as MIDI numbers: Am7 - Fmaj7 - Cmaj7 - G6
CHORDS = [(45, [57, 60, 64, 67]), (41, [53, 57, 60, 64]),
          (48, [60, 64, 67, 71]), (43, [55, 59, 62, 64])]


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def reverb(x, seconds=2.8, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 7, n))
    ir /= np.sqrt((ir ** 2).sum())
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[:len(x)]
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    n = int(a.seconds * SR)
    t = np.arange(n) / SR
    pad = np.zeros(n)
    bass = np.zeros(n)
    arp_l = np.zeros(n)
    arp_r = np.zeros(n)
    rng = np.random.default_rng(1)

    n_chords = int(np.ceil(a.seconds / CHORD_SEC)) + 1
    xf = 1.5                                   # chord crossfade, seconds
    for k in range(n_chords):
        root, notes = CHORDS[k % len(CHORDS)]
        s0 = k * CHORD_SEC - xf / 2
        s1 = s0 + CHORD_SEC + xf
        i0, i1 = max(0, int(s0 * SR)), min(n, int(s1 * SR))
        if i0 >= i1:
            continue
        tt = t[i0:i1] - s0
        dur = s1 - s0
        env = np.clip(np.minimum(tt / xf, (dur - tt) / xf), 0, 1)
        env = 0.5 - 0.5 * np.cos(np.pi * env)
        for m in notes:
            f = hz(m)
            for cents in (-5, 0, 5):
                fd = f * 2 ** (cents / 1200)
                ph = rng.uniform(0, 2 * np.pi)
                pad[i0:i1] += env * (np.sin(2 * np.pi * fd * tt + ph)
                                     + 0.15 * np.sin(4 * np.pi * fd * tt + ph))
        bass[i0:i1] += env * np.sin(2 * np.pi * hz(root) * tt)

        # soft arpeggio: one pluck every 0.75 s from the chord, an octave up
        for j in range(int(CHORD_SEC / 0.75)):
            start = k * CHORD_SEC + j * 0.75
            p0 = int(start * SR)
            if p0 >= n:
                break
            m = notes[[0, 2, 1, 3, 2, 1, 3, 0][j % 8]] + 12
            L = min(int(1.6 * SR), n - p0)
            tp = np.arange(L) / SR
            pluck = np.sin(2 * np.pi * hz(m) * tp) * np.exp(-tp * 3.2) * np.minimum(tp / 0.01, 1)
            pan = 0.35 if j % 2 else 0.65
            arp_l[p0:p0 + L] += pluck * (1 - pan)
            arp_r[p0:p0 + L] += pluck * pan

    pad /= np.abs(pad).max()
    bass /= np.abs(bass).max()
    arp_peak = max(np.abs(arp_l).max(), np.abs(arp_r).max(), 1e-9)
    arp_l /= arp_peak
    arp_r /= arp_peak

    # gentle swell in the middle, fade in/out
    swell = 0.85 + 0.15 * np.sin(np.pi * np.clip(t / a.seconds, 0, 1))
    fade = np.clip(np.minimum(t / 3.0, (a.seconds - t) / 5.0), 0, 1)
    base = (0.55 * pad + 0.30 * bass) * swell * fade
    left = base + 0.22 * arp_l * fade
    right = base + 0.22 * arp_r * fade
    left = 0.75 * left + 0.45 * reverb(left, seed=2)
    right = 0.75 * right + 0.45 * reverb(right, seed=3)
    peak = max(np.abs(left).max(), np.abs(right).max())
    stereo = np.stack([left, right], 1) / peak * 0.5

    with wave.open(a.out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((stereo * 32767).astype(np.int16).tobytes())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
