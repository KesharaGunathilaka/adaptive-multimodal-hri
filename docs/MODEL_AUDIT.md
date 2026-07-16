# MODEL_AUDIT — Phase 0 findings (2026-07-16, WIN-3060)

Audit of the 4 unimodal perception models per HANDOVER_CLAUDE.md §5.1. Canonical deployed
checkpoints are hashed in `docs/checkpoint_manifest.sha256`; `jetson_deploy/` copies verified
byte-identical. Accuracies are the held-out real-video numbers from each modality's reports.

## 1. Emotion — MobileNetV2 (per-frame, face crop)

| | |
|---|---|
| Framework | PyTorch (torchvision MobileNetV2 backbone) |
| Deployed checkpoint | `modalities/emotion/checkpoints/finetuned_MobileNetV2.pth` |
| Input | 224×224 RGB face crop, ImageNet norm (mean [0.485,0.456,0.406], std [0.229,0.224,0.225]); per-frame |
| Output | 7-class softmax, **RAF-DB order**: `Surprise, Fear, Disgust, Happy, Sad, Anger, Neutral` |
| Accuracy | 92.5% acc / 90.1% macro-F1 (real-world clips) |
| Inference | `modalities/emotion/inference/video.py --video <clip>`; `config.py` `EMOTION_LABELS` |
| Notes | Alternates exist (EfficientNet_B0 etc.) — **not** deployed. Thin Fear/Disgust training samples. Requires a face detector upstream (see inference script). |

⚠ Contract: handover §5.2 lists emotion as `neutral, happy, sad, angry, disgust, fear, surprise` —
actual head order differs. **The fusion feature table will store probabilities in the model's
native RAF-DB order**; any remapping happens once, documented, at extraction time.

## 2. Gesture — TCN over MediaPipe keypoint sequences (per-window)

| | |
|---|---|
| Framework | PyTorch (custom TCN, 683,272 params) + MediaPipe Holistic for keypoints |
| Deployed checkpoint | `modalities/gesture/checkpoints/best_TCN.pth` + `model_config.json` |
| Input | window of **32 frames** × 185-dim per-frame features: pose 33×(x,y,vis) + each hand 21×(x,y) wrist-relative/scaled + 2 presence flags |
| Output | 8-class softmax: `idle, wave, point, thumbs_up, thumbs_down, beckoning, raise_hand, both_hands_up` |
| Accuracy | val 93.2%/92.8%; real-world 84.1% acc / 82.9% macro-F1; conf_threshold 0.6 in live use |
| Inference | `modalities/gesture/inference/video.py --input <clip_or_folder>`; engine in `src/engine.py` |
| Notes | Weak: static `point`, weak `wave`. Class 0 is `idle` = the table's `none` — a real observation, distinct from [MISSING]. |

⚠ Contract: handover table order (`raise_hand, wave, point, thumbs_up, thumbs_down, both_hands_up, beckoning, neutral(none)`) differs from the actual head order above. Native order wins.

## 3. Motion — skeleton-sequence classifier (per-window)

| | |
|---|---|
| Framework | PyTorch |
| Deployed checkpoint | `modalities/motion/checkpoints/best_model_finetuned.pt` |
| Input | sliding window of **30 frames** × 84-dim normalized skeleton features (14 joints × 3, normalized per `src/inference.py`) |
| Output | **4-class** softmax: `sitting, standing, walking, stepping_back` |
| Accuracy | 76.8% acc / 73.2% macro-F1 (weakest model; kitchen `stepping_back` is the known failure) |
| Inference | `modalities/motion/inference/video.py --video <clip>`; `MotionInference` in `src/inference.py` |

⚠ Contract: handover §5.2 said 5 classes incl. `run`; V3 table §2.5 says "(6)". **Actual head = 4,
no `run`.** V3 rows describing running (#23, #34, #53) can at best produce `walking`/`stepping_back`.
⚠ **No direction output** (toward robot/exit/object) — but V3 uses direction as the key
disambiguator (F01 vs F09, F03 vs F06, #58/#59). Open decision — see DATASET_STATUS.md.

## 4. Context / Scene — CLIP ViT-B-32 zero-shot (per-sampled-frame)

| | |
|---|---|
| Framework | open_clip (ViT-B-32), zero-shot prompts; no local checkpoint (HF cache in `jetson_deploy/hf_cache/`) |
| Input | RGB frame, CLIP preprocess; sampled every N frames (SCENE_EVERY=5 in deploy config) |
| Output | 5-class prob over `classroom, kitchen, hospital, cloth_store, museum` (softmax over prompt similarities, + abstain prompts) |
| Accuracy | 98.8% acc / 99.3% macro-F1 (replaced legacy 3-class CNN whose kitchen domain gap was 82.2%) |
| Inference | `modalities/context/scene_classification/inference/realtime.py`; classifier in `scene_classification/src/zero_shot.py`, labels in `scene_classification/config.py` `SCENE_LABELS` |
| Notes | Separate SmolVLM2-500M caption path exists (`modalities/context/src/`) — NOT part of the fusion cue; fusion consumes the scene classifier only. Legacy `best_EfficientNet_B0.pth` kept but not deployed. |

⚠ Contract: table says `Shop`; model label is `cloth_store`. Only classroom/kitchen occur in
recorded data. For fusion, context cue = 5-dim CLIP softmax (or collapse to 2-dim
classroom/kitchen + other — decide at extraction, log in DECISIONS.md).

## 5. Fusion input contract (v1 — derived from actual heads)

One row per (W=32, S=8) window step on the shared 15 fps clip timeline:

| Cue | Dim | Source granularity → aggregation |
|---|---|---|
| emotion | 7 | per-frame → mean softmax over last 8 frames of window |
| gesture | 8 | native window 32 → one vector per step |
| motion | 4 | native window 30 → pad/align to the 32-frame window, one vector per step |
| context | 5 | per-sampled-frame → recompute each 30 frames, hold between |

Total 24-dim (+4 missing flags). Note clips are **15 fps** (not 30): W=32 ≈ 2.1 s, S=8 ≈ 0.53 s
of real time. Window-size sweep (handover §8.3) should confirm W given 15 fps sources vs 30 fps
deployment — flag this in the sweep writeup.

## 6. ONNX export risk (trial due Week 1)

- Emotion/motion/gesture nets: plain PyTorch — low risk.
- Gesture/motion **feature extraction** (MediaPipe, skeleton normalization) happens outside the
  net — ONNX covers only the classifier; MediaPipe stays as-is on Jetson.
- Context CLIP: open_clip → ONNX is doable but heavier; alternative is keeping CLIP in PyTorch on
  Jetson (it already runs from the HF cache). Decide after trial export.
