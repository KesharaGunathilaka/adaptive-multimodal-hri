# Real-world dataset evaluation (videos/dataset/hri-multimodal-intent-v1.0.0)

1270 clips, model TCN `best_TCN.pth`. offline = whole-clip resample; engine = GestureEngine simulation (EMA + conf gate + 300 ms debounce, video time).

## All trained-class clips (1009 clips)

| Mode | Accuracy | Macro-F1 (6 trained classes) |
|---|---|---|
| offline | 61.1% | 60.4% |
| engine | 61.1% | 63.1% |

### Per-class (engine mode)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| idle | 0.16 | 1.00 | 0.27 | 56 |
| wave | 1.00 | 0.46 | 0.63 | 117 |
| point | 0.86 | 0.21 | 0.33 | 184 |
| thumbs_up | 1.00 | 0.69 | 0.82 | 127 |
| thumbs_down | 0.88 | 0.58 | 0.70 | 279 |
| beckoning | 0.99 | 0.79 | 0.88 | 122 |
| raise_hand | 0.65 | 1.00 | 0.79 | 124 |

### Confusion (engine mode)

| true \ pred | idle | wave | point | thumbs_up | thumbs_down | beckoning | raise_hand | both_hands_up |
|---|---|---|---|---|---|---|---|---|
| **idle** | 56 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **wave** | 24 | 54 | 0 | 0 | 0 | 0 | 39 | 0 |
| **point** | 133 | 0 | 38 | 0 | 6 | 0 | 7 | 0 |
| **thumbs_up** | 24 | 0 | 0 | 88 | 13 | 1 | 1 | 0 |
| **thumbs_down** | 93 | 0 | 6 | 0 | 161 | 0 | 19 | 0 |
| **beckoning** | 23 | 0 | 0 | 0 | 2 | 96 | 1 | 0 |
| **raise_hand** | 0 | 0 | 0 | 0 | 0 | 0 | 124 | 0 |
| **both_hands_up** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Truly unseen scenarios only (no training/val overlap) (763 clips)

| Mode | Accuracy | Macro-F1 (6 trained classes) |
|---|---|---|
| offline | 51.9% | 38.0% |
| engine | 52.0% | 39.6% |

### Per-class (engine mode)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| idle | 0.17 | 1.00 | 0.29 | 56 |
| wave | 1.00 | 0.46 | 0.63 | 117 |
| point | 0.86 | 0.21 | 0.33 | 184 |
| thumbs_up | 1.00 | 0.69 | 0.82 | 127 |
| thumbs_down | 0.89 | 0.58 | 0.70 | 279 |
| beckoning | 0.00 | 0.00 | 0.00 | 0 |
| raise_hand | 0.00 | 0.00 | 0.00 | 0 |

### Confusion (engine mode)

| true \ pred | idle | wave | point | thumbs_up | thumbs_down | beckoning | raise_hand | both_hands_up |
|---|---|---|---|---|---|---|---|---|
| **idle** | 56 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **wave** | 24 | 54 | 0 | 0 | 0 | 0 | 39 | 0 |
| **point** | 133 | 0 | 38 | 0 | 6 | 0 | 7 | 0 |
| **thumbs_up** | 24 | 0 | 0 | 88 | 13 | 1 | 1 | 0 |
| **thumbs_down** | 93 | 0 | 6 | 0 | 161 | 0 | 19 | 0 |
| **beckoning** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **raise_hand** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **both_hands_up** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Untrained classes (point, both_hands_up) — what the model does

- **both_hands_up** (261 clips) → engine predicts: idle 147/261, raise_hand 103/261, thumbs_down 11/261

## Per-scenario breakdown (engine mode)

| Scenario | Context | Intended | n | Offline acc | Engine acc | Top engine pred | Overlap |
|---|---|---|---|---|---|---|---|
| S01_F04 | classroom | raise_hand | 70 | 100% | 100% | raise_hand | TRAINED (subject+clips in train) |
| S02_F01 | classroom | wave | 61 | 52% | 75% | wave | unseen |
| S03_F05 | classroom | point | 55 | 0% | 0% | idle | unseen |
| S04_F04 | classroom | thumbs_down | 61 | 70% | 66% | thumbs_down | unseen |
| S05_F02 | classroom | both_hands_up | 50 | 0% | 0% | idle | unseen |
| S06_F08 | classroom | thumbs_down | 63 | 92% | 86% | thumbs_down | unseen |
| S07_F03 | classroom | beckoning | 59 | 100% | 100% | beckoning | TRAINED (subject+clips in train) |
| S08_F06 | classroom | idle | 56 | 98% | 100% | idle | unseen |
| S09_F02 | classroom | both_hands_up | 50 | 0% | 0% | idle | unseen |
| S11_F05 | classroom | raise_hand | 54 | 100% | 100% | raise_hand | val/test subject (never trained) |
| S12_F01 | classroom | thumbs_up | 54 | 93% | 91% | thumbs_up | unseen |
| S18_F01 | kitchen | thumbs_up | 73 | 53% | 53% | thumbs_up | unseen |
| S19_F02 | kitchen | both_hands_up | 54 | 0% | 0% | raise_hand | unseen |
| S20_F03 | kitchen | beckoning | 63 | 59% | 59% | beckoning | val/test subject (never trained) |
| S21_F04 | kitchen | thumbs_down | 58 | 62% | 52% | thumbs_down | unseen |
| S22_F05 | kitchen | point | 60 | 0% | 0% | idle | unseen |
| S23_F08 | kitchen | thumbs_down | 30 | 27% | 20% | idle | unseen |
| S24_F07 | kitchen | both_hands_up | 59 | 0% | 0% | raise_hand | unseen |
| S25_F09 | kitchen | wave | 56 | 5% | 14% | raise_hand | unseen |
| S26_F02 | kitchen | both_hands_up | 48 | 0% | 0% | idle | unseen |
| S27_F06 | kitchen | point | 11 | 0% | 0% | idle | unseen |
| S28_F10 | kitchen | thumbs_down | 67 | 51% | 46% | idle | unseen |
| S29_F03 | kitchen | point | 58 | 66% | 66% | point | unseen |