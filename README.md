# adaptive-multimodal-hri

Adaptive human–robot interaction framework integrating multimodal emotion, gesture,
motion, and contextual cues for real-time policy learning and dynamic robot behavior.
Four independent perception models feed a downstream fusion layer; each is
documented and benchmarked in its own folder under `modalities/`.

## Modalities

| Modality | Model | Real-video accuracy (test-subject, held-out) | Status |
|---|---|---|---|
| [Context/Scene](modalities/context/README.md) | Zero-shot CLIP (ViT-B/32) | **98.8% acc / 99.3% macro-F1** | Fusion-ready, no known issues |
| [Emotion](modalities/emotion/README.md) | MobileNetV2, RAF-DB + real-world fine-tune | **92.5% acc / 90.1% macro-F1** | Fusion-ready |
| [Gesture](modalities/gesture/README.md) | TCN over MediaPipe Holistic landmarks | **84.1% acc / 82.9% macro-F1** | Fusion-ready; 2 gesture classes unverified on held-out subjects |
| [Motion](modalities/motion/README.md) | LSTM over MediaPipe Pose landmarks | **76.8% acc / 73.2% macro-F1** | Fusion-ready; known kitchen stepping-back gap |

All numbers above are on subjects **never used in training or checkpoint
selection** (P03/P05/P07/P08/P09 in `data/annotations/clips.csv`) — see each
modality's README for the full breakdown, known failure modes, and how the
number was produced. Numbers evaluated on the same subset without this care
(train or validation subjects mixed in) read noticeably higher for
emotion/motion and should not be trusted as generalization estimates.

## Repository layout

```
adaptive-multimodal-hri/
├── modalities/          # emotion, gesture, motion, context — each self-contained
│                         # (config.py, src/, scripts/, inference/, checkpoints/, reports/)
├── data/                 # git-ignored: the real intent-video dataset (annotations + raw clips)
├── fusion/               # multimodal fusion models (rule-based baseline; more planned)
├── docs/                 # design specs (e.g. gesture v2 guide) and dataset documentation
└── videos/               # miscellaneous captured footage, older struct/test copies
```

## Setup

```bash
.venv\Scripts\activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Each modality's README has its own run instructions; all of them assume the
repo's shared `.venv` (CUDA-enabled PyTorch).
