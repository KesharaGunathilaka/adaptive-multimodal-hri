# Real-world dataset evaluation (videos/dataset/hri-multimodal-intent-v1.0.0)

1270 clips, model TCN `best_TCN.pth`. offline = whole-clip resample; engine = GestureEngine simulation (EMA + conf gate + 300 ms debounce, video time).

## All trained-class clips (1270 clips)

| Mode | Accuracy | Macro-F1 (6 trained classes) |
|---|---|---|
| offline | 66.5% | 63.9% |
| engine | 66.8% | 65.4% |

### Per-class (engine mode)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| idle | 0.14 | 0.84 | 0.24 | 56 |
| wave | 0.72 | 0.46 | 0.56 | 117 |
| point | 0.92 | 0.26 | 0.40 | 184 |
| thumbs_up | 0.91 | 0.57 | 0.70 | 127 |
| thumbs_down | 0.93 | 0.71 | 0.80 | 279 |
| beckoning | 0.88 | 0.88 | 0.88 | 122 |
| raise_hand | 0.65 | 1.00 | 0.78 | 124 |
| both_hands_up | 0.98 | 0.77 | 0.86 | 261 |

### Confusion (engine mode)

| true \ pred | idle | wave | point | thumbs_up | thumbs_down | beckoning | raise_hand | both_hands_up |
|---|---|---|---|---|---|---|---|---|
| **idle** | 47 | 0 | 1 | 6 | 0 | 2 | 0 | 0 |
| **wave** | 19 | 54 | 0 | 0 | 5 | 0 | 39 | 0 |
| **point** | 105 | 10 | 47 | 0 | 7 | 0 | 15 | 0 |
| **thumbs_up** | 36 | 5 | 0 | 72 | 2 | 12 | 0 | 0 |
| **thumbs_down** | 59 | 4 | 3 | 0 | 197 | 0 | 12 | 4 |
| **beckoning** | 12 | 1 | 0 | 0 | 0 | 107 | 2 | 0 |
| **raise_hand** | 0 | 0 | 0 | 0 | 0 | 0 | 124 | 0 |
| **both_hands_up** | 59 | 1 | 0 | 1 | 0 | 0 | 0 | 200 |

## Truly unseen scenarios only (no training/val overlap) (926 clips)

| Mode | Accuracy | Macro-F1 (6 trained classes) |
|---|---|---|
| offline | 57.8% | 44.3% |
| engine | 58.2% | 44.7% |

### Per-class (engine mode)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| idle | 0.15 | 0.84 | 0.26 | 56 |
| wave | 0.74 | 0.46 | 0.57 | 117 |
| point | 0.92 | 0.26 | 0.40 | 184 |
| thumbs_up | 0.92 | 0.57 | 0.70 | 127 |
| thumbs_down | 0.93 | 0.71 | 0.80 | 279 |
| beckoning | 0.00 | 0.00 | 0.00 | 0 |
| raise_hand | 0.00 | 0.00 | 0.00 | 0 |
| both_hands_up | 0.97 | 0.75 | 0.84 | 163 |

### Confusion (engine mode)

| true \ pred | idle | wave | point | thumbs_up | thumbs_down | beckoning | raise_hand | both_hands_up |
|---|---|---|---|---|---|---|---|---|
| **idle** | 47 | 0 | 1 | 6 | 0 | 2 | 0 | 0 |
| **wave** | 19 | 54 | 0 | 0 | 5 | 0 | 39 | 0 |
| **point** | 105 | 10 | 47 | 0 | 7 | 0 | 15 | 0 |
| **thumbs_up** | 36 | 5 | 0 | 72 | 2 | 12 | 0 | 0 |
| **thumbs_down** | 59 | 4 | 3 | 0 | 197 | 0 | 12 | 4 |
| **beckoning** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **raise_hand** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **both_hands_up** | 41 | 0 | 0 | 0 | 0 | 0 | 0 | 122 |

## Per-scenario breakdown (engine mode)

| Scenario | Context | Intended | n | Offline acc | Engine acc | Top engine pred | Overlap |
|---|---|---|---|---|---|---|---|
| S01_F04 | classroom | raise_hand | 70 | 100% | 100% | raise_hand | TRAINED (subject+clips in train) |
| S02_F01 | classroom | wave | 61 | 66% | 72% | wave | unseen |
| S03_F05 | classroom | point | 55 | 0% | 0% | idle | unseen |
| S04_F04 | classroom | thumbs_down | 61 | 87% | 87% | thumbs_down | unseen |
| S05_F02 | classroom | both_hands_up | 50 | 100% | 100% | both_hands_up | TRAINED (copied to custom train) |
| S06_F08 | classroom | thumbs_down | 63 | 100% | 100% | thumbs_down | unseen |
| S07_F03 | classroom | beckoning | 59 | 100% | 100% | beckoning | TRAINED (subject+clips in train) |
| S08_F06 | classroom | idle | 56 | 82% | 84% | idle | unseen |
| S09_F02 | classroom | both_hands_up | 50 | 100% | 100% | both_hands_up | unseen |
| S11_F05 | classroom | raise_hand | 54 | 100% | 100% | raise_hand | val/test subject (never trained) |
| S12_F01 | classroom | thumbs_up | 54 | 61% | 56% | thumbs_up | unseen |
| S18_F01 | kitchen | thumbs_up | 73 | 58% | 58% | thumbs_up | unseen |
| S19_F02 | kitchen | both_hands_up | 54 | 56% | 59% | both_hands_up | unseen |
| S20_F03 | kitchen | beckoning | 63 | 76% | 76% | beckoning | val/test subject (never trained) |
| S21_F04 | kitchen | thumbs_down | 58 | 62% | 60% | thumbs_down | unseen |
| S22_F05 | kitchen | point | 60 | 10% | 10% | idle | unseen |
| S23_F08 | kitchen | thumbs_down | 30 | 43% | 47% | thumbs_down | unseen |
| S24_F07 | kitchen | both_hands_up | 59 | 68% | 68% | both_hands_up | unseen |
| S25_F09 | kitchen | wave | 56 | 18% | 18% | raise_hand | unseen |
| S26_F02 | kitchen | both_hands_up | 48 | 58% | 58% | both_hands_up | val/test subject (copied to custom) |
| S27_F06 | kitchen | point | 11 | 0% | 0% | idle | unseen |
| S28_F10 | kitchen | thumbs_down | 67 | 48% | 48% | thumbs_down | unseen |
| S29_F03 | kitchen | point | 58 | 71% | 71% | point | unseen |