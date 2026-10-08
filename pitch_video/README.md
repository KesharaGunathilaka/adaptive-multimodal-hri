# PerceptAI pitch video: real system output

Everything in `perceptai_output/` is genuine PerceptAI output. Each video is
the demo app (`jetson_deploy/fusion/demo_exhibition_v4.py`) running on one
recorded clip, rendered frame by frame at the clip's own speed. Nothing was
edited, and every decision was logged in `logs/`.

| File | Use in the 2-min video | Length | Source clip | Split | Result |
|---|---|---|---|---|---|
| `01_genuine_thumbs_up` | Real-output shot 1 | 5.8 s | S32_F01_c036 (kitchen) | test | F01 → A01 greeting |
| `02_sarcastic_thumbs_up` | Real-output shot 2 | 4.2 s | S59_F07_c014 (kitchen) | test | F07 → A08 de-escalate |
| `03_emergency_no_gesture` | Real-output shot 3 | 6.0 s | S54_F02_c026 (kitchen) | test | F02 → A02 emergency stop |
| `spare_emergency_classroom` | Backup for shot 3, or the live-demo slide | 6.3 s | S23_F02_c034 (classroom) | test | F02 → A14 halt + notify |
| `spare_greeting_wave` | Backup / use-case montage | 5.8 s | S18_F01_c033 (classroom) | test | F01 → A01 greeting |
| `spare_discouraged_validation_clip` | Optional | 5.6 s | S20_F10_c004 (classroom) | validation | F10 → A12 encourage |

"Test" clips were never used in training or model selection. The validation
clip was used for model selection, so don't call it "unseen".

## How to include them
- Put the caption **"Real PerceptAI output · recorded test clips"** on the three
  main shots, bottom-left, for the whole shot.
- The first ~0.8 s of each clip shows "Waiting for a person…" while evidence
  builds (the emergency clip starts at 0.3 s). Trim it, or keep it to show
  the system "thinking".
- The clips are short. To fill a longer slot, freeze the last frame for 1–2 s.
  Do not slow the clip down or loop it.
- `source_clips/` holds the original camera footage, if you want to show a
  before/after split.
- Recorded clips use a 2 s action hold (the live demo uses 5 s), because each
  clip is only 4–6 s long.

## Regenerate a clip
```
.venv\Scripts\python.exe jetson_deploy\fusion\demo_exhibition_v4.py --source <clip.mp4> --hold 2 --no-display --record out.mp4
```
