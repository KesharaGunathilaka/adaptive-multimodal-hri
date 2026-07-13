# Motion Recognition (skeleton LSTM)

Body-motion classification for the adaptive HRI system. MediaPipe Pose
**world landmarks** are mapped onto a 14-joint NTU-style skeleton
(hip-centered, shoulder-width normalized, X/Y axis-corrected), stacked with
per-frame velocities into 84-dim features, and a 3-layer **LSTM** classifies
30-frame windows (~1 s at 30 fps) into **4 classes**:

`sitting · standing · walking · stepping_back`

Trained on **NTU RGB+D** skeletons, then **fine-tuned on the project's
real-world intent dataset** (`videos/struct`) to close the domain gap.
Targeted at **Jetson Orin Nano**.

## Results

| Checkpoint | Data | Val / clip accuracy |
|---|---|---|
| `best_model.pt` | NTU only (4-class) | 96.7% val (window-level) |
| `best_model_6class.pt` | NTU only (6-class, +running/slumped) | 95.9% val |
| **`best_model_finetuned.pt` (deployed)** | NTU + struct train scenarios | **82.6%** clip-level on held-out scenarios (baseline: 27.4%) |

Per-class (fine-tuned, held-out test clips): sitting F1 0.90, standing 0.96,
walking 0.77, stepping_back 0.48. Known weakness: slow walking **toward** the
camera reads as standing (translation is normalized away — see
`reports/TRAINING_NOTES.txt` for the plan to expose z-velocity to fusion).

## Folder structure

```
motion/
├── README.md
├── requirements.txt
├── src/                    # importable library (no CLI)
│   ├── model.py            # MotionLSTM architecture
│   └── inference.py        # MotionInference: sliding-window realtime engine
├── scripts/                # runnable pipeline stages
│   ├── build_dataset.py        # NTU skeletons -> windowed .npy training data
│   ├── build_hri_dataset.py    # videos/struct clips -> fine-tune windows
│   ├── train.py                # train from scratch on NTU windows
│   ├── tune.py                 # optuna hyper-parameter search
│   ├── finetune.py             # fine-tune best_model.pt on struct windows
│   ├── batch_process_videos.py # run inference/video.py over a folder
│   └── explore_skeleton.py     # NTU skeleton EDA helper
├── inference/
│   ├── realtime.py         # live webcam demo (skeleton overlay + prediction)
│   └── video.py            # video file -> annotated mp4 (--save -> outputs/)
├── checkpoints/            # .pt weights (git-ignored)
├── logs/                   # training logs, optuna study, best hparams
├── reports/                # analysis report (html/pdf), charts, training notes
└── outputs/                # annotated inference outputs (git-ignored)
```

## Pipeline (run from this folder)

```bash
# 1. NTU base training
python scripts/build_dataset.py            # needs NTU_DIR env var or default path
python scripts/train.py
python scripts/tune.py                     # optional; writes logs/best_hparams.json

# 2. Real-world fine-tune on videos/struct
python scripts/build_hri_dataset.py --split train
python scripts/build_hri_dataset.py --split val
python scripts/finetune.py
```

## Inference

```bash
python inference/video.py --video ../../videos/struct/raw/clips/classroom/S01_F04/S01_F04_c001.mp4 --save --headless
python inference/realtime.py               # webcam
python scripts/batch_process_videos.py --input-folder <dir> --save
```

## Main script integration (how to call)

```python
import sys, os
sys.path.insert(0, "modalities/motion/src")
from inference import MotionInference

engine = MotionInference("modalities/motion/checkpoints/best_model_finetuned.pt")
result = engine.update(joints_25)      # (25,3) NTU-layout world joints per frame
if result is not None and result.label != "buffering":
    print(result.label, result.confidence)   # + result.probs for fusion
```

`MotionInference` keeps its own 30-frame sliding buffer; call
`engine.reset()` when the tracked person changes. The MediaPipe → NTU joint
mapping and axis correction live in `inference/video.py` (`MP_TO_NTU`,
negate X/Y of world landmarks).

## Notes

- Frame-rate handling: struct clips are recorded at mixed fps; fine-tune
  windows are resampled onto a uniform 30 Hz grid (`build_hri_dataset.py`)
  so window semantics match NTU training.
- Scenario S19 ("Move backward (run)") is excluded from fine-tuning — it has
  no clean mapping onto the 4-class taxonomy.
- Constants (joint subset, normalization, window size) are intentionally
  duplicated between `src/inference.py` and the dataset builders — they must
  match exactly; change them together.
- History note: this modality was developed in a separate repo
  (`hrimultimodal/hri_motion_model`); its local `.git` was corrupt and has
  been moved out of the tree (see repo root docs). The files are now tracked
  by the main repo.
