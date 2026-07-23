# Stage 1 — Data & labels

**Goal of this stage:** turn a pile of recorded video clips into a clean,
machine-readable, leakage-free dataset that the rest of the pipeline can trust.

Everything downstream (feature extraction, training, evaluation) reads the
outputs of this stage and never touches the raw videos again for label
decisions. So mistakes here poison everything — which is why this stage got the
most careful auditing.

---

## 1.1 The design: what the dataset is *supposed* to contain

The dataset is defined by a design table, `docs/Final_Dataset.docx` (**"V3"**,
the third revision). It is not a list of videos — it is a **specification** of
scenarios. Each of its **62 scenarios** is one row describing a situation:

> *"Student raises hand with neutral face while sitting at desk"* →
> context = classroom, emotion = neutral, gesture = raise_hand, motion = sitting,
> intent = **F04 (help request)**, action = A05, tests = —.

Key properties of the V3 table:
- **62 scenarios** = 31 classroom + 31 kitchen, split **41 train / 21 test** (2:1).
- **10 intents** (F01–F10), each appearing ≥2 train + ≥1 test in *both*
  environments.
- **15 robot actions** (A01–A15); the intent→action mapping is in the table.
- A **labeling rubric** (§2.6 of the table) that says exactly how cues combine
  into an intent — e.g. *"thumbs down: sad+static → F04; angry → F07; disgust →
  F08; happy → F01 (playful)."* This rubric is the ground truth the fusion model
  must learn, and the reference the rule-based baseline mirrors.

**Why a table first?** Because the whole thesis rests on cue *combinations*, the
combinations have to be designed deliberately (e.g. deliberately include the
same gesture under different emotions) rather than hoped-for in random footage.
The table guarantees coverage of the G1/G2/G3 cases.

### The 10 intents

| Code | Name | Code | Name |
|---|---|---|---|
| F01 | Greeting / positive ack | F06 | Requests passage / space |
| F02 | Emergency / danger | F07 | Frustration / agitation |
| F03 | Task assistance request | F08 | Break / relief request |
| F04 | Help request | F09 | Farewell |
| F05 | Engaged / busy (no interaction) | F10 | Discouraged / giving up |

---

## 1.2 What was actually recorded (the gap)

The table describes 62 scenarios, but **only 22 are recorded so far** — these
correspond to the *train* rows of the earlier **V2** table (the team recorded
against V2, then the table was revised to V3). So the first job of this stage was
to reconcile "what's on disk" against "what V3 says."

- Recorded footage: **~1,061 clips** after curation, 640×480 and 720p/portrait
  phone video, ~4–5 s each, 4 actors, folders named `S<row>_F<intent>`.
- Full mapping recorded-scenario → V3 row is in `docs/DATASET_STATUS.md §2`.
- The 21 V3 *test* scenarios and 18 unrecorded V3 *train* scenarios do not exist
  as video yet. Stage 5 fills the unrecorded *train* rows synthetically; the
  *test* rows must be recorded later for the full T01–T05 evaluation.

**Why not just record them?** The dataset is frozen (deadline + actor
availability). The handover's explicit instruction was to fill unrecorded rows
synthetically from existing cue data, and to record the test set separately later.

---

## 1.3 Curation: `data/` vs `videos/struct/`

There are two copies of the footage on disk, and it matters which is canonical:

| Folder | Clips | Role |
|---|---|---|
| `data/raw/clips/` | 1,061 | **Canonical (curated).** The team manually deleted clips that didn't clearly show the intended cue (bad lighting, wrong expression, off-frame), and dropped one whole scenario (S27) that was unusable. |
| `videos/struct/raw/clips/` | 1,270 | Uncurated archive — the full original set, kept only for provenance. **Never train from it.** |

**Decision (2026-07-16, user):** `data/` is the single source of truth. This is
recorded in `DECISIONS.md`. The rule for anyone/any agent: *never "restore"
missing clips from `struct` — they were removed on purpose.*

---

## 1.4 The annotation files  (what the code reads)

Stage 1 produces three machine-readable files under `data/`, all built by
`scripts/00_inventory.py` and `scripts/01_update_annotations.py`:

### `data/annotations/clips.csv` — one row per clip
`clip_id, scenario_id, person_id, filepath, duration_s, fps, resolution,
frame_count, sha256, split_subject`. This is the curated clip list. Two
corrections were applied here:
- Filtered from 1,270 → 1,061 rows to match the curated videos on disk.
- **fps/resolution/duration rewritten from an actual OpenCV probe**, because the
  original annotations were wrong for **104 kitchen clips**: they claimed 15 fps
  but were really 24/30 fps phone video, some in portrait (1080×1920). This bug
  is why all later windowing is **time-based, not frame-based** (Stage 3).

### `data/labels.csv` — one row per scenario (the label authority)
`scenario_id, v3_row, split, status, context, emotion, gesture, motion, missing,
intent, intent_recorded, action, n_clips, note`. This carries the **V3-corrected
labels**, not the folder names. Two label corrections live here:
- **S05:** folder says F02, but V3 relabeled it **F07** (angry + both-hands-up +
  standing in a classroom = controlled frustration, not danger). `labels.csv`
  carries F07.
- **S28 vs S21 collision:** both are *kitchen, sad, thumbs-down, sitting* but the
  table wants S21=F04 and S28=F10. Identical cues, different labels = unlearnable.
  Resolution: **S21 is the training scenario (F04); S28 is marked
  `status=recombination_pool`** — its clips are excluded from supervised training
  and instead serve as *source material* for synthetic recombination (Stage 5).

### `data/inventory.csv` — one row per clip, integrity check
Output of `scripts/00_inventory.py`, which opens every clip with OpenCV and reads
the first and last frame. Result: **all 1,061 clips readable, 0 corrupt.** (This
matters because a 2026-07-13 incident had left kitchen clips 0-byte after a bad
copy; the check confirms that's resolved.)

---

## 1.5 The split protocol (why the results are trustworthy)

A fusion model can look great and still be useless if it was tested on data too
similar to training. Two rules prevent that:

1. **Scenario split is fixed by the table** (train rows vs test rows) — never
   moved. (For now all 22 recorded scenarios are V3-train, so the current
   test/val split is *within* them, by actor — see below.)
2. **Actor-disjoint validation/test.** All clips of one person are entirely in
   one split. We never let the model train on person P01 and test on P01 —
   otherwise it could memorize a face/body instead of learning the cue logic.

Concretely, from the subject split in the annotations:

| Split | Actors | Clips |
|---|---|---|
| train | P01, P02, P06 | 886 |
| val | P04 | 93 |
| test | P03, P05, P07, P08, P09 | 82 |

So every accuracy number you'll see later is measured on **people the model
never trained on**. The test set is 100% real and never augmented.

**Honest limitation:** because only 22 scenarios are recorded, "test" here means
*unseen actors on seen situations*. It proves the model generalizes across
people; it does not yet prove generalization across *situations* (that needs the
V3 test scenarios recorded). We state this everywhere.

---

## 1.6 What Stage 1 produced (summary)

```
data/
├── raw/clips/…                 1,061 curated videos (canonical)
├── annotations/clips.csv       per-clip, probed metadata + subject split
├── labels.csv                  per-scenario V3-corrected labels + status
└── inventory.csv               integrity probe (all readable)
```
Plus the reconciliation docs `DATASET_STATUS.md` and the decisions in
`DECISIONS.md`. Scripts: `scripts/00_inventory.py`, `scripts/01_update_annotations.py`.

**In one sentence:** we took 1,061 hand-curated clips, gave them correct probed
metadata and V3-correct labels, resolved two label conflicts, and split them so
no actor appears on two sides — producing a small but clean and honestly-split
dataset that the rest of the pipeline reads instead of the videos.

---

### Open points you might want to change
- The S28→`recombination_pool` decision is reversible: if you later verify S28's
  videos show no real thumbs-down, we can promote it back to F10 training.
- The current val actor is a single person (P04). If you want a more robust val
  estimate we could rotate actors (k-fold by subject) — costs more compute, tiny
  here.
- Once test scenarios are recorded, this doc gets a new section and the
  "honest limitation" above is lifted.
