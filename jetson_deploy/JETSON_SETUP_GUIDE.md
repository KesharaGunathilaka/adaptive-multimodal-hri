# Jetson Orin Nano — First-Time Setup & Live Inference Guide

> This copy lives inside `jetson_deploy/` itself so it travels with the
> folder over USB. Start with **`CLAUDE.md`** in this same folder (rename
> `HANDOVER_CLAUDE.md` to `CLAUDE.md` if it hasn't been already) for the
> bigger picture and what to do with what you find here — this file is just
> the step-by-step mechanics. Log results in `JETSON_TEST_LOG.md`, also in
> this folder.

Target: get all 4 deployed models (emotion, gesture, motion, context) running
live on the Jetson Orin Nano with the RealSense D455, starting from a board
that has JetPack booted but nothing project-related installed yet.

Everything you need is already packaged in this `jetson_deploy/` folder (see
`README.md` for the short version — this doc is the long version, written
for someone who hasn't touched a Jetson before). You'll work directly at the
Jetson's own monitor/keyboard and move files over by USB drive, per your
setup.

---

## Step 0 — Identify what you're working with

On the Jetson, open a terminal and run:

```bash
cat /etc/nv_tegra_release
sudo apt show nvidia-jetpack 2>/dev/null | grep -i version
python3 --version
nvcc --version 2>/dev/null || echo "nvcc not on PATH (normal, CUDA is still installed)"
df -h /
free -h
```

This tells you three things that change the exact commands later:

| JetPack | L4T | Ubuntu | Python | CUDA |
|---|---|---|---|---|
| 5.x | R35 | 20.04 | 3.8 | 11.4 |
| 6.x | R36 | 22.04 | 3.10 | 12.2 / 12.6 |

**Tell me the output of the first three commands and I'll pin down exact
version numbers for Step 3 below** — the two JetPack generations need
different PyTorch wheel sources. In the meantime, everything else in this
guide (RealSense, mediapipe, project copy, running the models) is the same
either way.

Also check free disk: the project copy is ~2.5 GB (see Step 2), and a full
Python environment (PyTorch + transformers + mediapipe + ultralytics +
librealsense) will add another 5–8 GB. If `/` shows less than ~20 GB free,
say so — swap/storage may need attention first (common on the base 64 GB
Orin Nano eMMC/SD image).

---

## Step 1 — System prep

Update packages and switch to max performance mode (you want this for any
real-time inference benchmarking):

```bash
sudo apt update && sudo apt upgrade -y

# List power modes, then set to the highest-performance one (usually mode 0 = MAXN)
sudo nvpmodel -q
sudo nvpmodel -m 0
sudo jetson_clocks

# Optional but very useful: a `htop`-like GPU/CPU/thermal monitor
sudo pip3 install -U jetson-stats
sudo reboot
```

After reboot, run `jtop` any time you want to watch GPU/CPU utilization and
temperature while a model is running live — genuinely useful for judging
whether a model is fast enough in practice, not just accurate.

---

## Step 2 — Copy the project onto the Jetson (USB drive)

On the **Windows PC**, first trim ~570 MB of dead weight — leftover partial
downloads in the HF cache that aren't referenced by anything (verified: no
symlinks are used in this cache, everything is a plain file, so this is safe
to delete):

```powershell
cd "d:\Documents\Project\12. Adaptive Human Robot Interaction\adaptive-multimodal-hri\jetson_deploy"
Get-ChildItem -Recurse -Filter *.incomplete | Remove-Item
```

Then copy the whole `jetson_deploy/` folder (now ~2.5 GB) to your USB drive.
No symlink or archive tricks needed — every file in `hf_cache/` is a real
file on this Windows checkout, so a plain folder copy is safe and Linux will
read it back fine.

- If the drive is FAT32: fine here (largest single file is the SmolVLM2
  `model.safetensors` at 1.9 GB, under FAT32's 4 GB per-file limit) — but
  exFAT is safer in general if you reuse this drive for anything bigger later.
- On the Jetson, plug in the drive, then:

```bash
lsblk                      # find the drive, e.g. sda1
udisksctl mount -b /dev/sda1   # or just use the file manager GUI — it auto-mounts

mkdir -p ~/hri
cp -r /media/$USER/<drive-name>/jetson_deploy ~/hri/
cd ~/hri/jetson_deploy
du -sh .                   # sanity check it all came across (~2.5 GB)
```

---

## Step 3 — Python virtual environment + PyTorch

**Do not `pip install torch` from plain PyPI on Jetson** — the generic wheel
is x86_64 and either fails to install or installs a CPU-only build with no
CUDA support. Jetson needs ARM64 wheels built against the board's specific
JetPack/CUDA/cuDNN combo.

```bash
sudo apt install -y python3-venv python3-pip
python3 -m venv ~/hri/venv
source ~/hri/venv/bin/activate
pip install --upgrade pip
```

**If JetPack 6.x (L4T R36, CUDA 12.x)** — use the community-run Jetson AI Lab
pip index, which hosts prebuilt ARM64+CUDA wheels matched to JetPack 6
(this is the current recommended path, far simpler than the old manual
`.whl` download from NVIDIA forums):

```bash
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
```

**If JetPack 5.x (L4T R35, CUDA 11.4)** — use the jp5 index instead:

```bash
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp5/cu114
```

Verify CUDA is actually visible before doing anything else:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If this prints `False` for `cuda.is_available()`, stop here and fix it before
continuing — nothing downstream will be usably fast on CPU-only ARM.

> **Heads-up for JetPack 5 specifically**: the context modality needs a fairly
> recent `transformers` for SmolVLM2, which in turn wants a fairly recent
> PyTorch. If you're on JetPack 5 and the `transformers`/`open_clip_torch`
> install in Step 5 complains about a torch version requirement, tell me the
> exact error — we may need to pin an older `transformers` for that one
> modality rather than fighting the JetPack 5 torch ceiling. Emotion, gesture,
> and motion don't have this constraint (plain torch/torchvision only).

---

## Step 4 — RealSense SDK (D455) + pyrealsense2

This is the fiddliest part of the whole setup — Intel does not publish a
prebuilt `pyrealsense2` wheel for Linux ARM64 on PyPI, so `pip install
pyrealsense2` will fail here (it's fine on your Windows/x86 dev machine,
which is why it's just in `requirements.txt` unqualified). You build
`librealsense` from source instead, which also builds the Python bindings.
This is Intel's own documented method for Jetson boards.

```bash
sudo apt install -y git cmake build-essential libssl-dev libusb-1.0-0-dev \
    libudev-dev pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev

cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.55.1          # or the latest tagged release — avoid building master

# udev rules so the D455 doesn't need sudo to access
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

mkdir build && cd build
source ~/hri/venv/bin/activate   # make sure the venv's python is what gets bound
cmake .. \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_GRAPHICAL_EXAMPLES=ON

make -j$(nproc)     # this takes a while on Orin Nano — 20-40 min is normal
sudo make install
```

`-DFORCE_RSUSB_BACKEND=ON` is the key flag: it makes librealsense talk to the
camera via libusb directly instead of the Linux kernel's native UVC/HID
metadata patches, which historically required patching and rebuilding the
Jetson kernel. RSUSB avoids all of that — recommended by Intel for Jetson
specifically.

Point Python at the built bindings (the build installs them under
`/usr/local/lib` by default, which usually isn't on your venv's import path):

```bash
echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib' >> ~/hri/venv/bin/activate
source ~/hri/venv/bin/activate
python3 -c "import pyrealsense2 as rs; print(rs.__version__)"
```

Verify the camera itself with the GUI viewer (plug the D455 into a USB3 port
— the blue one; USB2 ports will connect but severely limit bandwidth/fps):

```bash
realsense-viewer
```

You should see live RGB + depth streams. If `realsense-viewer` shows the
device but streams are choppy or fail to start, it's almost always a USB3
port/cable issue on Jetson (short, USB3-certified cable, USB3 port
specifically) — not a software bug.

---

## Step 5 — Remaining Python dependencies

Install order matters here: `ultralytics` (used by an earlier, now-removed
context sub-model but still listed in the shared `requirements.txt`) and
`mediapipe` both pull their own torch/opencv constraints and can silently
downgrade or reinstall the JetPack-matched torch you just built if you're not
careful. Install torch first (done above), then everything else, and check
torch's CUDA availability again afterward.

```bash
cd ~/hri/jetson_deploy
source ~/hri/venv/bin/activate

pip install opencv-python pillow numpy tqdm scikit-learn pandas tabulate
pip install mediapipe==0.10.35
pip install ultralytics --no-deps   # --no-deps: keep it from touching your JetPack torch
pip install transformers open_clip_torch num2words

# Re-verify torch wasn't silently replaced
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Notes:
- `mediapipe==0.10.35` does publish `manylinux2014_aarch64` wheels on PyPI, so
  this should install cleanly on Jetson without a source build. If it fails
  to find a matching wheel, tell me the exact error — there are known-good
  community-built aarch64 mediapipe wheels as a fallback.
- Skip `pyrealsense2` in `requirements.txt` — you already have it from the
  source build in Step 4; installing over it from pip would fail anyway
  (no aarch64 wheel) or silently no-op.
- `optuna` (hyperparameter tuning) isn't needed for inference-only testing —
  skip it unless you plan to retrain on-device.

---

## Step 6 — Point HuggingFace at the bundled offline cache

The context modality (CLIP scene classifier + SmolVLM2) needs model weights
from `jetson_deploy/hf_cache/`. Set these every session (or add to
`~/.bashrc`) so it never tries to hit the network:

```bash
export HF_HOME=~/hri/jetson_deploy/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

```bash
echo 'export HF_HOME=~/hri/jetson_deploy/hf_cache' >> ~/.bashrc
echo 'export HF_HUB_OFFLINE=1' >> ~/.bashrc
echo 'export TRANSFORMERS_OFFLINE=1' >> ~/.bashrc
```

---

## Step 7 — Sanity checks before touching the camera

Run each of these from `~/hri/jetson_deploy` with the venv active. Each
should complete with no errors (they just instantiate the model and print
the resolved checkpoint path — don't need a camera yet):

```bash
cd ~/hri/jetson_deploy
python3 -c "
import sys; sys.path.insert(0, '.')
from modalities.emotion.src.models import build_model
print('emotion OK')
"

python3 -c "
import sys; sys.path.insert(0, 'modalities/gesture')
from src.engine import GestureEngine
GestureEngine()
print('gesture OK')
"

python3 -c "
import sys; sys.path.insert(0, 'modalities/motion/src')
from inference import MotionInference
MotionInference('modalities/motion/checkpoints/best_model_finetuned.pt')
print('motion OK')
"

python3 -c "
import sys; sys.path.insert(0, '.')
from modalities.context.scene_classification.src.classifier import create_scene_classifier
create_scene_classifier('clip')
print('context/CLIP OK (loaded from offline cache, no network call)')
"
```

If the last one tries to hit the network instead of using the cache, double
check the three `HF_*` env vars are actually exported in *this* shell
(`echo $HF_HOME`), not just written to `.bashrc` in a shell you haven't
reloaded.

---

## Step 8 — Run each model live

All commands run from `jetson_deploy/` with the venv active.

```bash
# Emotion — RealSense (falls back to webcam if RealSense isn't found)
python3 modalities/emotion/inference/realtime_realsense.py

# Gesture — RealSense
python3 modalities/gesture/inference/realtime_realsense.py

# Motion — plain webcam capture (cv2.VideoCapture(0) — the D455's RGB
# sensor also enumerates as a regular UVC webcam, so this works with it too)
python3 modalities/motion/inference/realtime.py

# Context — CLIP scene + SmolVLM2, plain webcam capture
python3 modalities/context/inference/realtime.py

# Context — scene only (lighter, skips the ~1s/frame VLM call)
python3 modalities/context/scene_classification/inference/realtime.py
```

**Check the printed checkpoint/model path at startup for each script before
trusting results** — this project got bitten once by a folder copy silently
resetting a default checkpoint to the weak baseline (see
`jetson_deploy/modalities/emotion/README.md`). It's one line to glance at,
cheap insurance.

Known accuracy/failure modes per model (so a live demo that looks
occasionally wrong isn't a surprise) are documented in each modality's
`README.md` in `jetson_deploy/modalities/<name>/README.md` — worth a skim
before your first live test, especially:
- Motion: `stepping_back` fails specifically in kitchen-like framing.
- Gesture: static/seated `point` reads as idle; `wave` recall is weak.
- Emotion: solid across the board (92.5% test-subject accuracy).
- Context/scene: strongest of the four (~99% on captured video).

---

## Step 9 — Video-file mode (no camera needed, good for a first smoke test)

If you want to confirm everything works before wrestling with the camera,
every modality also has a `video.py` that runs on a file instead:

```bash
python3 modalities/emotion/inference/video.py --video path/to/clip.mp4
python3 modalities/gesture/inference/video.py --input path/to/clip.mp4 --save
python3 modalities/motion/inference/video.py --video path/to/clip.mp4 --save
python3 modalities/context/inference/video.py --input path/to/clip_or_folder --save
```

You can copy a couple of clips from the main repo's `videos/` folder onto the
same USB drive for this. `--save` writes an annotated mp4 to an `outputs/`
folder next to the script — easiest way to eyeball results without a live
camera in the loop.

---

## Troubleshooting quick-reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is `False` | Wrong wheel source (generic PyPI torch instead of Jetson AI Lab index) | Re-do Step 3 with the correct jp5/jp6 index URL |
| `pip install pyrealsense2` fails | No aarch64 Linux wheel on PyPI — expected | Build from source (Step 4), don't pip install it |
| `realsense-viewer` sees device but no stream / very low fps | USB2 port or cable | Use a USB3 port + short USB3-certified cable |
| `import pyrealsense2` → `ModuleNotFoundError` even after building | Bindings installed to `/usr/local/lib`, not on venv's path | `export PYTHONPATH=$PYTHONPATH:/usr/local/lib` (Step 4) |
| Context modality tries to download from HF despite cache present | `HF_HOME`/`HF_HUB_OFFLINE` not exported in current shell | `echo $HF_HOME` to check; re-`source ~/.bashrc` or re-export |
| `ultralytics` install downgrades torch | It pulled its own torch dependency | Reinstall with `pip install ultralytics --no-deps`, verify torch again |
| Board feels sluggish / thermal throttling during live inference | Not in max-performance power mode | `sudo nvpmodel -m 0 && sudo jetson_clocks`; watch with `jtop` |
| `mediapipe` install fails, no matching wheel | Uncommon on 0.10.35, but Jetson pip resolver quirks happen | Report the exact error — fallback is a community-built aarch64 wheel |

---

## Still open — record these in `JETSON_TEST_LOG.md`

1. Output of the Step 0 version-check commands (JetPack/L4T/Python version) —
   locks in the exact torch install command.
2. Whether `mediapipe==0.10.35` installs cleanly from PyPI on your board (it
   should, but Jetson pip resolvers surprise people).
3. Whether JetPack 5 vs 6 changes the `transformers`/SmolVLM2 install (only
   relevant if you're on JetPack 5 — see the callout in Step 3).

Once the board is running, the real target of this exercise (per
`CLAUDE.md` in this folder) is benchmarking actual inference latency of each
of the 4 models on-device, not just "it runs" — record numbers in
`JETSON_TEST_LOG.md` so they make it back into the main repo's
`docs/WORKLOG.md` next sync.
