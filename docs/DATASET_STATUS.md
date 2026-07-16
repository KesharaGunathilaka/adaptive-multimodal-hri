# DATASET_STATUS — recorded data vs Final_Dataset V3 (2026-07-16, WIN-3060; updated same day after user decisions)

The Final_Dataset.docx (V3, 62 scenarios) is the label authority, but **only the old-table (V2)
train scenarios are recorded**. This file maps what exists on disk to V3 rows and lists every gap.

## 1. What is on disk

| Location | Clips | Status |
|---|---|---|
| `data/raw/clips/` | **1,061** (22 scenarios) | **Canonical (user-curated).** The team manually removed low-quality clips that didn't depict the intended cues — 209 removed, including *all* of S27_F06. `data/annotations/clips.csv` updated 2026-07-16 to match (rows filtered, fps/resolution/duration corrected from OpenCV probe, `split_subject` merged in). `data/labels.csv` = per-scenario V3-corrected labels. `data/inventory.csv` = per-clip probe results (all 1,061 readable). |
| `videos/struct/raw/clips/` | 1,270 | Full **uncurated archive** (pre-curation superset). Keep for provenance; do not train from it. Its `annotations/splits.csv` is the source of the subject split. |

Clip properties (probed, not the old annotation values): **770 clips 640×480 @ 15 fps** (RealSense),
**~290 clips at 24/30 fps** in mixed resolutions incl. portrait phone video (1280×720, 1024×576,
1080×1920) — all kitchen scenarios. ~4–5 s each. **Feature extraction must window by TIME using
per-clip probed fps, not by raw frame count.**

Subject split after curation (actor-disjoint, from struct `splits.csv`): P01/P02/P06 → train
(886 clips), P04 → val (93), P03/P05/P07/P08/P09 → test (82).

## 2. Recorded scenario → V3 row mapping

Folder numbers = old-table (V2) row numbers. V3 renumbered; mapping by cue tuple + V3 change log:

| Recorded | Cues (context, emotion, gesture, motion) | V3 row | V3 intent | Note |
|---|---|---|---|---|
| S01_F04 | class, neutral, raise hand, sit | #7 | F04 | |
| S02_F01 | class, happy, wave, walk | #1 | F01 | direction: toward robot |
| S03_F05 | class, neutral, point, sit | #10 | F05 | |
| S04_F04 | class, sad, thumbs down, sit | #8 | F04 | |
| S05_F02 | class, angry, both hands up, stand | #14 | **F07** | ⚠ RELABELED F02→F07 in V3 |
| S06_F08 | class, disgust, thumbs down, step back | #16 | F08 | |
| S07_F03 | class, neutral, beckoning, sit | #5 | F03 | |
| S08_F06 | class, neutral, [MISSING], walk | #12 | F06 | gesture missing by design |
| S09_F02 | class, surprise, both hands up, stand | #3 | F02 | |
| S11_F05 | class, happy, raise hand, sit | #11 | F05 | |
| S12_F01 | class, happy, thumbs up, sit | #2 | F01 | |
| S18_F01 | kitchen, happy, thumbs up, stand | #32 | F01 | |
| S19_F02 | kitchen, fear, both hands up, step back | #34 | F02 | |
| S20_F03 | kitchen, neutral, beckoning, walk | #36 | F03 | |
| S21_F04 | kitchen, sad, thumbs down, sit | #38 | F04 | ⚠ collides with S28 |
| S22_F05 | kitchen, neutral, point, stand | #40 | F05 | |
| S23_F08 | kitchen, disgust, thumbs down, step back | #46 | F08 | only 30 clips |
| S24_F07 | kitchen, angry, both hands up, stand | #44 | F07 | |
| S25_F09 | kitchen, happy, wave, walk | #48 | F09 | direction: toward exit |
| S26_F02 | kitchen, surprise, both hands up, step back | #35 | F02 | |
| S27_F06 | kitchen, disgust, point, step back | #42 | F06 | only 11 clips |
| S28_F10 | kitchen, sad, thumbs down, sit | #51 | F10 | ⚠ V3 sets gesture=none; recordings directed WITH thumbs down |
| S29_F03 | kitchen, neutral, point, walk | #37 | F03 | |

## 3. Label issues — RESOLVED 2026-07-16 (user decisions)

1. **S21_F04 vs S28_F10 collision → S21 trains, S28 reserved.** `labels.csv` marks S28
   `status=recombination_pool`: its 53 clips are excluded from supervised training and serve as
   cue-vector inputs for synthetic recombination. After feature extraction, S28 clips whose
   *gesture-model output* is `idle` may be promoted to F10 training (they then match V3 #51
   exactly); clips with clear `thumbs_down` stay in the pool.
2. **S05_F02 relabeled to F07** in `labels.csv` (V3 row #14). Folder names unchanged.
3. **Direction cue: skipped for fusion v1** (user decision). Expect F01/F09 (S02 vs S25 are the
   only wave+walk rows and sit in different contexts, which softens this) and F03/F06 confusion;
   revisit after error analysis.
4. Fear appears only via S19 (kitchen). V3 train rows #4 (classroom fear) etc. are unrecorded.

## 4. Coverage vs V3 (what recombination must fill)

- Recorded: 23 of 41 V3 **train** rows.
- **Unrecorded train rows (18):** #4, #6, #9, #13, #15, #17, #18, #19, #20, #21 (classroom);
  #33, #39, #41, #43, #45, #47, #49, #50 (kitchen). All missing-cue train rows (#6, #9, #12*,
  #43, #49) except #12=S08 are unrecorded → cue-recombination + modality-dropout must cover them.
- **Test rows (21): none recorded.** Plan per handover: test set will be recorded later; until
  then, all evaluation is on held-out real subjects/clips from the 23 recorded scenarios, and
  T01–T05 subsets only become fully measurable once test recordings exist.
- 10 intents: all appear in recorded data ≥1× except — F10 only via disputed S28; F02 has 3
  scenarios; classroom lacks recorded F07 (S05 relabel supplies it), F09, F10.

## 5. Direction gap

V3 added a Direction column as the stated disambiguator (greet vs farewell, summon vs give-way,
rows #58/#59 double-masked), but **no perception model outputs direction**.
**Decision 2026-07-16: skip direction for fusion v1** and measure the resulting F01/F09 and
F03/F06 confusion in error analysis. If it dominates the error budget, v2 option remains:
derive a toward-robot/away/lateral/static feature from skeleton trajectory or person-bbox scale
change (no retraining needed) and add it as a 5th fusion token.
