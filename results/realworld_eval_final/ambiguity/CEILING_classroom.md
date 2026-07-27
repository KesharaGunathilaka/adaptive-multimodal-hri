# Table-imposed intent ceiling — classroom

Generated 2026-07-27 03:37 · weights = clips on disk (view: realsense_480p)

A fusion model reading only the cues left in a masking condition cannot exceed these numbers, because the V3 table maps the same cue tuple to different intents on some rows. Compare every masking result against the matching ceiling before calling a model weak.

## Recorded `test` rows — 10 rows, 160 clips

| Cue(s) masked | Ceiling | Clips unreachable |
|---|---|---|
| none | 0.9 | 16 |
| context | 0.9 | 16 |
| emotion | 0.8 | 32 |
| gesture | 0.9 | 16 |
| motion | 0.9 | 16 |
| context+emotion | 0.8 | 32 |
| context+gesture | 0.9 | 16 |
| context+motion | 0.9 | 16 |
| emotion+gesture | 0.4 | 96 |
| emotion+motion | 0.8 | 32 |
| gesture+motion | 0.6 | 64 |

**Ambiguous under `none`**

- rows [22, 30] → {'F01': 16, 'F09': 16} (context=classroom, emotion=[missing], gesture=wave, motion=walk) — 16 clips unreachable

**Ambiguous under `context`**

- rows [22, 30] → {'F01': 16, 'F09': 16} (emotion=[missing], gesture=wave, motion=walk) — 16 clips unreachable

**Ambiguous under `emotion`**

- rows [28, 29] → {'F07': 16, 'F08': 16} (context=classroom, gesture=thumbs up, motion=sit) — 16 clips unreachable
- rows [22, 30] → {'F01': 16, 'F09': 16} (context=classroom, gesture=wave, motion=walk) — 16 clips unreachable

**Ambiguous under `gesture`**

- rows [22, 30] → {'F01': 16, 'F09': 16} (context=classroom, emotion=[missing], motion=walk) — 16 clips unreachable

**Ambiguous under `motion`**

- rows [22, 30] → {'F01': 16, 'F09': 16} (context=classroom, emotion=[missing], gesture=wave) — 16 clips unreachable

**Ambiguous under `context+emotion`**

- rows [28, 29] → {'F07': 16, 'F08': 16} (gesture=thumbs up, motion=sit) — 16 clips unreachable
- rows [22, 30] → {'F01': 16, 'F09': 16} (gesture=wave, motion=walk) — 16 clips unreachable

**Ambiguous under `context+gesture`**

- rows [22, 30] → {'F01': 16, 'F09': 16} (emotion=[missing], motion=walk) — 16 clips unreachable

**Ambiguous under `context+motion`**

- rows [22, 30] → {'F01': 16, 'F09': 16} (emotion=[missing], gesture=wave) — 16 clips unreachable

**Ambiguous under `emotion+gesture`**

- rows [22, 23, 24, 27, 30] → {'F01': 16, 'F02': 16, 'F03': 16, 'F06': 16, 'F09': 16} (context=classroom, motion=walk) — 64 clips unreachable
- rows [26, 28, 29] → {'F05': 16, 'F07': 16, 'F08': 16} (context=classroom, motion=sit) — 32 clips unreachable

**Ambiguous under `emotion+motion`**

- rows [28, 29] → {'F07': 16, 'F08': 16} (context=classroom, gesture=thumbs up) — 16 clips unreachable
- rows [22, 30] → {'F01': 16, 'F09': 16} (context=classroom, gesture=wave) — 16 clips unreachable

**Ambiguous under `gesture+motion`**

- rows [22, 25, 30] → {'F01': 16, 'F04': 16, 'F09': 16} (context=classroom, emotion=[missing]) — 32 clips unreachable
- rows [27, 28] → {'F06': 16, 'F07': 16} (context=classroom, emotion=angry) — 16 clips unreachable
- rows [24, 26] → {'F03': 16, 'F05': 16} (context=classroom, emotion=neutral) — 16 clips unreachable

## Recorded `train` rows — 19 rows, 708 clips

| Cue(s) masked | Ceiling | Clips unreachable |
|---|---|---|
| none | 0.9774 | 16 |
| context | 0.9774 | 16 |
| emotion | 0.7528 | 175 |
| gesture | 0.6907 | 219 |
| motion | 0.9774 | 16 |
| context+emotion | 0.7528 | 175 |
| context+gesture | 0.6907 | 219 |
| context+motion | 0.9774 | 16 |
| emotion+gesture | 0.4068 | 420 |
| emotion+motion | 0.6342 | 259 |
| gesture+motion | 0.5918 | 289 |

**Ambiguous under `none`**

- rows [1, 18] → {'F01': 42, 'F09': 16} (context=classroom, emotion=happy, gesture=wave, motion=walk) — 16 clips unreachable

**Ambiguous under `context`**

- rows [1, 18] → {'F01': 42, 'F09': 16} (emotion=happy, gesture=wave, motion=walk) — 16 clips unreachable

**Ambiguous under `emotion`**

- rows [7, 11] → {'F04': 70, 'F05': 54} (context=classroom, gesture=raise hand, motion=sit) — 54 clips unreachable
- rows [3, 14] → {'F02': 40, 'F07': 48} (context=classroom, gesture=both hands up, motion=stand) — 40 clips unreachable
- rows [8, 15, 17] → {'F04': 61, 'F07': 16, 'F08': 17} (context=classroom, gesture=thumbs down, motion=sit) — 33 clips unreachable
- rows [1, 18, 19] → {'F01': 42, 'F09': 32} (context=classroom, gesture=wave, motion=walk) — 32 clips unreachable
- rows [10, 20] → {'F05': 42, 'F10': 16} (context=classroom, gesture=none, motion=sit) — 16 clips unreachable

**Ambiguous under `gesture`**

- rows [5, 7, 10] → {'F03': 59, 'F04': 70, 'F05': 42} (context=classroom, emotion=neutral, motion=sit) — 101 clips unreachable
- rows [2, 11] → {'F01': 54, 'F05': 54} (context=classroom, emotion=happy, motion=sit) — 54 clips unreachable
- rows [13, 14] → {'F06': 16, 'F07': 48} (context=classroom, emotion=angry, motion=stand) — 16 clips unreachable
- rows [1, 18] → {'F01': 42, 'F09': 16} (context=classroom, emotion=happy, motion=walk) — 16 clips unreachable
- rows [12, 19] → {'F06': 54, 'F09': 16} (context=classroom, emotion=neutral, motion=walk) — 16 clips unreachable
- rows [8, 20] → {'F04': 61, 'F10': 16} (context=classroom, emotion=sad, motion=sit) — 16 clips unreachable

**Ambiguous under `motion`**

- rows [1, 18] → {'F01': 42, 'F09': 16} (context=classroom, emotion=happy, gesture=wave) — 16 clips unreachable

**Ambiguous under `context+emotion`**

- rows [7, 11] → {'F04': 70, 'F05': 54} (gesture=raise hand, motion=sit) — 54 clips unreachable
- rows [3, 14] → {'F02': 40, 'F07': 48} (gesture=both hands up, motion=stand) — 40 clips unreachable
- rows [8, 15, 17] → {'F04': 61, 'F07': 16, 'F08': 17} (gesture=thumbs down, motion=sit) — 33 clips unreachable
- rows [1, 18, 19] → {'F01': 42, 'F09': 32} (gesture=wave, motion=walk) — 32 clips unreachable
- rows [10, 20] → {'F05': 42, 'F10': 16} (gesture=none, motion=sit) — 16 clips unreachable

**Ambiguous under `context+gesture`**

- rows [5, 7, 10] → {'F03': 59, 'F04': 70, 'F05': 42} (emotion=neutral, motion=sit) — 101 clips unreachable
- rows [2, 11] → {'F01': 54, 'F05': 54} (emotion=happy, motion=sit) — 54 clips unreachable
- rows [13, 14] → {'F06': 16, 'F07': 48} (emotion=angry, motion=stand) — 16 clips unreachable
- rows [1, 18] → {'F01': 42, 'F09': 16} (emotion=happy, motion=walk) — 16 clips unreachable
- rows [12, 19] → {'F06': 54, 'F09': 16} (emotion=neutral, motion=walk) — 16 clips unreachable
- rows [8, 20] → {'F04': 61, 'F10': 16} (emotion=sad, motion=sit) — 16 clips unreachable

**Ambiguous under `context+motion`**

- rows [1, 18] → {'F01': 42, 'F09': 16} (emotion=happy, gesture=wave) — 16 clips unreachable

**Ambiguous under `emotion+gesture`**

- rows [2, 5, 7, 8, 10, 11, 15, 17, 20] → {'F01': 54, 'F03': 59, 'F04': 131, 'F05': 96, 'F07': 16, 'F08': 17, 'F10': 16} (context=classroom, motion=sit) — 258 clips unreachable
- rows [1, 12, 18, 19] → {'F01': 42, 'F06': 54, 'F09': 32} (context=classroom, motion=walk) — 74 clips unreachable
- rows [3, 13, 14, 21] → {'F02': 40, 'F06': 16, 'F07': 48, 'F10': 16} (context=classroom, motion=stand) — 72 clips unreachable
- rows [4, 16] → {'F02': 16, 'F08': 55} (context=classroom, motion=step back) — 16 clips unreachable

**Ambiguous under `emotion+motion`**

- rows [8, 15, 16, 17] → {'F04': 61, 'F07': 16, 'F08': 72} (context=classroom, gesture=thumbs down) — 77 clips unreachable
- rows [7, 11] → {'F04': 70, 'F05': 54} (context=classroom, gesture=raise hand) — 54 clips unreachable
- rows [3, 4, 14] → {'F02': 56, 'F07': 48} (context=classroom, gesture=both hands up) — 48 clips unreachable
- rows [1, 13, 18, 19] → {'F01': 42, 'F06': 16, 'F09': 32} (context=classroom, gesture=wave) — 48 clips unreachable
- rows [10, 20, 21] → {'F05': 42, 'F10': 32} (context=classroom, gesture=none) — 32 clips unreachable

**Ambiguous under `gesture+motion`**

- rows [5, 7, 10, 12, 19] → {'F03': 59, 'F04': 70, 'F05': 42, 'F06': 54, 'F09': 16} (context=classroom, emotion=neutral) — 171 clips unreachable
- rows [1, 2, 11, 18] → {'F01': 96, 'F05': 54, 'F09': 16} (context=classroom, emotion=happy) — 70 clips unreachable
- rows [8, 20, 21] → {'F04': 61, 'F10': 32} (context=classroom, emotion=sad) — 32 clips unreachable
- rows [13, 14, 15] → {'F06': 16, 'F07': 64} (context=classroom, emotion=angry) — 16 clips unreachable
