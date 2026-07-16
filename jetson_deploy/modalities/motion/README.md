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

| Checkpoint | Data | Accuracy |
|---|---|---|
| `best_model.pt` | NTU only (4-class) | 96.7% val (window-level, synthetic NTU benchmark) |
| `best_model_6class.pt` | NTU only (6-class, +running/slumped) | 95.9% val (synthetic) |
| **`best_model_finetuned.pt` (deployed)** | NTU + real intent-dataset clips | **86.8%** clip-level on real held-out video (85.1% on unseen subjects specifically) |

Full-dataset check (all 1,061 clips, every subject): sitting 99% F1,
standing 78-82% F1 (100% recall, 64-70% precision — over-triggers),
walking 83-89% F1, **stepping_back the weak class — 41-58% recall
depending on split**.

**Known weakness, now confirmed on the full dataset: `stepping_back` fails
specifically in the kitchen environment, for every subject (train and test
alike) — this is an environment/recording issue, not a subject-generalization
gap.** All three kitchen stepping_back scenarios fail almost completely
regardless of who's in the clip:

| Scenario | Env | n clips | Motion accuracy | What it predicts instead |
|---|---|---|---|---|
| `S06_F08` | classroom | 55 | **100%** | — |
| `S19_F02` | kitchen | 46 | **0%** | "standing", always |
| `S23_F08` | kitchen | 19 | **5%** | "walking" mostly |
| `S26_F02` | kitchen | 32 | **0%** | "standing", nearly always |

Since it fails for train-subject clips just as badly as test-subject clips
in the same scenarios, this isn't the model failing to generalize to new
people — something about the kitchen recording setup (camera distance/angle,
counter occlusion, or simply less pronounced backward steps in that space)
is defeating the feature pipeline. The model's features are hip-centered and
shoulder-normalized (global translation deliberately removed by design), so
a backward step with limited visible limb swing loses most of its signal —
consistent with `S08_F06` (walking *toward* the camera, classroom) also
scoring only 26-36%. **Recommended next step: pull up the raw `S19_F02` /
`S26_F02` clips and check camera distance/framing before retraining** — if
it's a framing issue, more training data won't fix it; if it's a genuine
data gap (kitchen stepping-back under-represented in fine-tuning), it will.
See `reports/TRAINING_NOTES.txt` for the plan to expose z-velocity to
fusion, which would give the classifier back the translation signal it
currently discards by design.

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
