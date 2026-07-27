# Are the four unimodal models fit to feed fusion? — complete classroom set

**Date:** 2026-07-27 · **Host:** WIN-3060 · **Data:** `data/final`, classroom,
RealSense 480p view, 868 clips over 29 recorded V3 rows (all 31 classroom rows
except #6 and #9, which are derived — see `docs/methodology/04_missing_cues.md`).

Evidence: `unimodal/UNIMODAL_classroom.md`, `ambiguity/CEILING_classroom.md`,
`fusion_transfer/TRANSFER_classroom.md`.
Scripts: `15_final_extract.py` → `16_final_unimodal_eval.py`,
`17_cue_ambiguity.py`, `19_final_fusion_eval.py`.

---

## 1. Verdict in one table

Accuracy on **`raw_take_20260725` clips only** — the 2026-07-25 recordings, the
only classroom clips no perception model has ever trained on. The other column is
kept beside it because it is the number that would otherwise be quoted, and it is
not a generalisation estimate.

| Modality | Training-overlap clips | **Held-out clips** | macro-F1 (held-out) | Fit for fusion? |
|---|---|---|---|---|
| **Context** (CLIP zero-shot) | 1.000 | **1.000** | 1.000 | ✅ Yes, unchanged |
| **Gesture** (TCN) | 0.882 | **0.766** | 0.689 | ⚠️ Usable, two fixable confusions |
| **Emotion** (MobileNetV2) | 0.969 | **0.643** | 0.494 | ❌ No — Fear/Disgust unusable |
| **Motion** (LSTM) | 0.920 | **0.637** | 0.446 | ❌ No — systematic class bias |

Context is genuinely solved (868/868, zero false kitchen/hospital predictions).
The 0.969 emotion and 0.920 motion figures come from clips those models were
fine-tuned on; the honest numbers are 0.643 and 0.637.

## 2. What the deployed fusion head does on this data

`jetson_deploy/fusion/fusion_attn.pt`, frozen, no retraining:

| Rows | Fusion v1 accuracy | Table ceiling |
|---|---|---|
| train-design (19 rows) | 0.887 | 0.977 |
| **test-design (10 rows)** | **0.388** | 0.900 |
| held-out clips only | 0.478 | — |

Fusion v1 reported **0.939** on `data/old`. It scores **0.388** on the classroom
test rows. **This is not evidence that fusion is broken** — fusion v1 was trained
on 23 of 41 V3 rows, none of them classroom test rows, so 10 of these 10 rows are
cue combinations it has never seen. It is evidence that **fusion must be retrained
on `data/final`**, which is now possible for the first time.

The per-row table makes the split unmistakable: every `curated_clip` row (what
fusion v1 trained on) scores ≥ 0.778 and most score 1.000; **every failure is a
2026-07-25 row.**

Two results worth keeping for the thesis:

* Under cue masking on train-design rows, fusion v1 sits *at* the table ceiling —
  gesture masked 0.678 vs ceiling 0.691, emotion+gesture masked 0.380 vs ceiling
  0.407. The masking machinery works; the Phase-2 robustness claim survives.
* Row #1 (F01) scores **1.000** and row #18 (F09) scores **0.000**. Identical cue
  tuple, opposite label — the model picks one and always answers it. That is the
  direction gap made visible in a single line, and it is the cleanest possible
  motivation figure for adding a direction cue.

## 3. Root causes, in order of how much they cost

### 3.1 Fusion has not seen 18 of the 29 recorded classroom rows (dominant)

Fully fixable, no new recording, no unimodal retraining. Retrain the fusion head
on the `data/final` feature table. Everything below matters *after* this.

### 3.2 Emotion errors cascade deterministically into wrong intents

Not random noise — each confusion maps one intent onto another specific intent:

| Emotion error | Recall | Rows hit | Intent consequence |
|---|---|---|---|
| **Fear → Neutral / Surprise** | 0.25 | #23 (test), #4 | **F02 emergency missed.** Row #23 scores 0.000 on emotion and 0.188 on fusion. Safety-critical. |
| **Disgust → Sad** (19/88) | 0.67 | #17, #29 (test) | F08 → **F04**, because (sad, thumbs_down, sit) *is* row #8 = F04. Row #17 fusion 0.235, row #29 0.312. |
| **Sad → Happy** (12/109) | 0.74 | #20, #21, #31 (test) | F10 → F01/F05. All three F10 rows land at 0.44–0.50. |

Anger (0.964), Happy (0.958), Neutral (0.960) and Surprise (0.975) are fine — the
model is not uniformly weak, it has three bad classes and they are the three the
classroom test set leans on.

Note the test-set design amplifies this: rows **#28 (F07) and #29 (F08) are
separable only by Anger vs Disgust**, and #23 (F02) only by Fear.

### 3.3 Motion has a systematic bias on the new framing

Not noise either — a consistent direction of error:

| True | Predicted | Rows |
|---|---|---|
| `stepping_back` | `standing` / `walking` | #4 → 0.000, #25 → 0.062 |
| `standing` | `walking` | #13 → 0.000, #21 → 0.188, #31 → 0.188 |
| `sitting` (arms overhead) | `standing` | #26 → 0.000 |

Meanwhile `walking` is near-perfect on the same clips (#18, #23, #24, #30 all
1.000). The model over-predicts motion. `stepping_back` collapsing is the same
weakness the 2026-07-16 audit already flagged for kitchen — it is now confirmed on
classroom and it is worse on the new framing, which points at camera distance /
full-body vs seated framing rather than at the class itself.

### 3.4 Gesture: two named confusions, otherwise strong

11 of 17 held-out scenarios score 1.000. The failures are concentrated:

* **wave ↔ idle ↔ raise_hand** — #25 raise_hand → wave ×15 (0.062), #30 wave →
  idle ×9 (0.438), #18/#19 wave → idle ×6 (0.625). An amplitude/duration problem.
* **thumbs_up/thumbs_down → point** — #17 ×5, #15 ×3, #29 ×3. Hand shape at 480p
  and distance.

### 3.5 Direction: an irreducible 0.90 cap until a 5th cue exists

Rows #22 (F01) and #30 (F09) are the identical tuple (classroom, emotion
`[MISSING]`, wave, walk). No amount of model improvement separates them.
Classroom test ceiling = 0.900; train ceiling = 0.977 (rows #1 vs #18).

## 4. The designed-missing rows did not come out as designed ⚠️

| Cue | Rows | Intended | Measured observation rate |
|---|---|---|---|
| emotion | #22, #25, #30 | face occluded / sensor occluded | **0.997** |
| gesture | #12, #23 | hands occupied / motion blur | **1.000** |

The emotion model finds a face in 99.7 % of windows on rows whose scenario text
says the face is hidden behind a book, and the gesture model always fires because
it only needs a pose, not hands. **So the "missing modality" test rows are not
actually missing at the sensor.** Consequences:

1. T03 on classroom currently measures *simulated* masking (flag-driven), not real
   occlusion. That is still a valid experiment but it is a weaker claim, and the
   thesis must not describe it as real sensor failure.
2. There is a **train/deploy mismatch**: training masks these rows from the table,
   but on the Jetson nothing sets `obs=0` for them — the detector succeeds — so the
   deployed system will feed a confident, meaningless cue into fusion. Confidence-
   thresholded masking (§5, item 5) closes this.
3. Cheapest real fix: re-shoot rows #22/#25/#30 with the face genuinely occluded
   (16 takes each, ~20 minutes) and #23 with real motion blur. This is the one
   place where new recording buys something no augmentation can.

## 4b. It is not the camera — all three views agree (added 2026-07-27, 1,440 clips)

Every 2026-07-25 take was filmed simultaneously by three cameras, so this compares
the *same performances* at 640×480@15, 1920×1080@30 and 3840×2160@60. Held-out
clips only, clip-level accuracy / macro-F1:

| Modality | RealSense 480p | phone 1080p | phone 4K |
|---|---|---|---|
| emotion | 0.643 / 0.494 | 0.683 / 0.527 | 0.674 / 0.481 |
| gesture | 0.766 / 0.689 | 0.718 / 0.750 | 0.746 / 0.747 |
| motion | 0.637 / 0.446 | 0.526 / 0.396 | 0.617 / 0.421 |
| context | 1.000 / 1.000 | 0.996 / 0.499 | 1.000 / 1.000 |

**A 36× increase in pixels buys nothing.** Emotion stays in 0.64–0.68 and motion in
0.53–0.64 across all three. So the weakness is *not* sensor resolution, and a
better camera on the Jetson would not fix it — which is exactly what makes
fine-tuning (plan items 2 and 3) the right lever rather than a hardware change.

Two secondary readings:

* Emotion is mildly better on the phones (+0.04 accuracy), consistent with the
  `data/old` view study; the face crop benefits from resolution while nothing else
  does. Not enough to justify changing the deployment camera.
* **Motion is *worse* on the higher-resolution phones** (0.526 at 1080p vs 0.637 at
  480p). Resolution cannot hurt a skeleton classifier, so this is the phone's
  different angle and distance — direct support for the framing hypothesis in §3.3
  and further evidence that motion needs fine-tuning on the new geometry, not a
  better sensor.

## 5. Recommended plan, highest value first

| # | Action | Cost | Expected gain |
|---|---|---|---|
| 1 | **Retrain fusion on `data/final`** (all 29 classroom rows + derived #6/#9) | ~1 h, no GPU decode | The 0.388 → high-0.8s. This is the whole gap for 18 rows. |
| 2 | **Fine-tune emotion on the 2026-07-25 classroom takes**, class-weighted toward Fear / Disgust / Sad | ~2 h | Directly unblocks F02 (safety), F08 and F10. Highest-leverage model fix. |
| 3 | **Fine-tune motion** on the new framing; oversample `stepping_back` and `standing` | ~2 h | Fixes rows #4, #13, #25, #26, #31. |
| 4 | **Collision-aware modality dropout** — never mask a cue whose removal makes a row's intent unrecoverable (computed by `17_cue_ambiguity.py`) | ~1 h | Removes label noise the current p=0.3 dropout injects, e.g. gesture-dropout on #20/#21 colliding with #9. |
| 5 | **Confidence-thresholded masking** at runtime, not just detector failure | ~1 h + τ sweep | Closes the train/deploy mismatch in §4. |
| 6 | **Direction cue** as a 5th token (approach / recede / lateral / static) from `pose_img` bbox scale + drift — already cached, no re-decode | ~3 h | Test ceiling 0.90 → 1.00; fixes #18 vs #1 and #22 vs #30. |
| 7 | Re-shoot #22/#25/#30/#23 with genuine occlusion | ~30 min recording | Turns T03 into a real-occlusion result. |
| 8 | Gesture fine-tune for wave/raise_hand/idle amplitude | ~2 h | Rows #25, #30, #18, #19. |

Do 1 before anything else: it is the largest single gain and it changes the error
analysis that items 2–8 should be prioritised against. Handover §8.4 is explicit
that a unimodal model should only be retrained once fusion error analysis pins the
blame on it — after item 1, that analysis is finally meaningful.

## 6. Open items

* `dilanka/` (40 phone clips) is one actor's session spanning several scenarios
  and is excluded everywhere until it is split into per-V3-row folders.
* `data/final/annotations/subjects_pending.csv` — 316 takes need a `person_id`
  before an actor-disjoint validation split is possible. 1,061 migrated clips
  resolved automatically from `data/old`.
* ~~Phone-view extraction~~ **done** (572 clips, 211.8 min, 0 failures) — see §4b.
  All 1,440 classroom clips now have a per-frame cache.
* Kitchen is 10 of 31 rows recorded — this assessment is classroom-only.
