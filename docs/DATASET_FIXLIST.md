# DATASET FIXLIST — `data/final_merged` + `docs/final_dataset_merged.docx`

Status as of **2026-08-01**. Re-check any time with:

```
.venv\Scripts\python scripts\21_validate_table.py    # the .docx against itself
.venv\Scripts\python scripts\22_verify_csvs.py       # the CSVs against the .docx
.venv\Scripts\python scripts\20_merged_annotations.py  # rebuild the CSVs
```

Current: **1 error, 13 warnings** from the table validator; **all 62 rows match**
between the .docx and the CSVs (41,848 field comparisons).

---

## ✅ DONE — P1, fixed in the document 2026-08-01

Backup of the pre-fix document: `docs/final_dataset_merged.docx.bak`.

- **Test-tag column** — `#18` → `T01`, `#32` → `T01, T04`, `#52` cleared (it is a
  train row). Neither #18 nor #32 got `T05`: that tag means "naive per-cue rules
  fail", and the `wave→greet` rule now *succeeds* on #18 while #32's aligned
  thumbs-up was never a rule failure.
- **Row #63 justification** — rewritten. It records what the row actually is (see
  "resolved" below) and the actor overlap with #38.
- **Motion legend** — `Motion (6)` → `Motion (4)`; four values are listed.
- **Row #30** deleted outright (it was already an empty row); **row #63** moved from
  between #56 and #57 to the end, so numbering reads 1→63 in order.
- **Word run-splitting** repaired in 9 rows (`F0 1`, `T est`, `T rain`, `n eutral`,
  `s tand`, `s t ep back`).
- **Stale notes removed** — "Optional trim if budget-tight" from #19/#21/#47, and
  #50's "VERIFY against recordings" (resolved, see below).
- **12 broken cross-references corrected**: #47 cited itself (→#46); #39 named #57
  as its context-masked twin (→#56); #9 named #54 as a gesture-masked test (→#53);
  #53 named #44 (→#43); #55 named #42 (→#41); #21 and #62 listed #52 among
  gesture-none rows (→#50); #26 listed #45 among both-hands-up rows (→#44); and
  #2/#13/#28/#32 named #60 as a conflict row when the kitchen conflict is #59.

Because the document is now correct, `merged_common.TABLE_OVERRIDES` is empty — the
parser reads the .docx literally, one source of truth.

**F09 numbering gap** — left as-is by decision; codes stay `F01–F08, F10`.

---

## ⚠ OPEN — 1 error

### E1. Same cues, same intent, different action (#1 vs #18)

```
classroom | happy | wave | walk  ->  F01
    #1  (train)  ->  A01   Positive acknowledgment; prompt/prepare next task
    #18 (test)   ->  A09   Wave back; do not follow
```

Deleting F09 removed the direction ambiguity from the **intent** level, but it
reappeared at the **action** level. F01 now maps to three actions (A01/A09/A10) and
nothing in the cue vector distinguishes them — the deciding factor is walking
direction, which no model outputs.

Consequence: if the policy layer is `intent → action`, F01 is under-determined by
design and A09/A10 are unreachable. Options: (a) accept it and always emit A01 for
F01, documenting the loss; (b) add a direction feature (`pose_img` is already
cached for exactly this, per DECISIONS 2026-07-27); (c) split F01 back into
greeting and farewell at the *action* layer only.

---

## ⚠ OPEN — recordings

### R1. Two rows have no deployment-view footage
`#40` (`kitchen/S40_F05`, train) and `#56` (`kitchen/S56_F04`, test) have **0**
RealSense clips — only phone. Every headline number is reported on the RealSense
640×480 view, so these two rows cannot appear in it.

### R2. Row #58 has no RealSense footage of its own
Its 12 RealSense clips are the copies derived from **#49**, a *train* row. Its own
19 clips are phone-only. A short RealSense re-record removes the contamination
flagged by `derived_rows.csv.crosses_split`.

### R3. Six kitchen rows have only 5–6 RealSense clips
`S52_F01` (6), `S53_F02` (5), `S54_F02` (6), `S55_F03` (6), `S59_F07` (6),
`S60_F08` (6) — four are test rows. RealSense is 1,338 of 2,870 clips (47%) and
lopsided: classroom #1–#17 are 100% RealSense, most kitchen rows are a minority.

### R4. 726 of 1,016 "phone_1080p" clips are not 1080p
All kitchen: 633 at 1280×720, 59 at 1024×576, 30 portrait, a few odd sizes. Looks
like a re-encode pass. If the originals survive, re-copy them.

### R5. 38 clips are shorter than the 4 s aggregation window
Shortest 1.67 s. Mostly `S36_F03` (6), `S40_F05` (5), `S52_F01` (5).

### R6. 35 duplicate files still on disk
Already excluded in `clips.csv` with `dup_of` pointers; deleting them makes disk
and CSV agree. List in `annotations/INTEGRITY.md`.

### R7. The `dilanka` subject was dissolved
36 of that actor's 60 clips were folded 4-at-a-time into the nine classroom **test**
folders and 24 dropped; the subject label survives only through the SHA-256
provenance map in `clips.csv`.

---

## ⚠ OPEN — label vs footage, needs your call

### L1. Row #31's motion (**unresolved — I did not change it**)
Motion column says `step back`, but the row's own scenario text says *"stands
turned away"* and its own justification says *"(turned away, **standing**)"*. The
frames sampled show people standing still. If it should be `stand`, note that #31
then becomes cue-identical to train row #21 (classroom/sad/idle/stand → F10) and
#31 is a **test** row.

### L2. Rows #10, #40, #41 label a described point as `idle`
All three are F05. Their scenario text describes pointing ("occasionally pointing at
the notebook / at ingredients", "self-directed point") but the gesture column says
`idle`, on the design argument that a task-embedded point is not a communicative
signal. The gesture model has no such notion — it classifies hand shape, so it will
output `point`. That is a systematic label-vs-perception mismatch on three rows.
Either re-word the scenarios (if the actors do not actually point) or accept that
these rows train the fusion head against a cue value perception will not produce.

### L3. "run" is not in the motion vocabulary
`#23`, `#34` and `#53` describe running; the vocabulary is sit/stand/walk/step back,
so it is silently normalised (walk, step back, walk). State the normalisation in the
document — for an emergency row the difference between running and walking is
exactly the cue a reader would expect to carry the alarm.

---

## ⚠ OPEN — metadata you supply

`person_id` is known for **1,097 of 2,870** clips (only the migrated ones).
Actor-disjoint validation is impossible until the rest are filled in. Template:
`data/final_merged/annotations/takes.csv` — one row per synchronised take, so ~781
answers rather than 2,870.

---

## ⚠ Known, accepted (warnings, no action needed)

- Row numbering skips 30, so 62 rows end at #63. Deliberate.
- The gesture legend says `neutral (none)` while the rows use `idle`; the context
  legend lists 5 environments of which 2 are used. Both are cosmetic wording
  choices, but aligning the gesture legend with the model's actual class name
  (`idle`) would remove a reader trap.
- `#57` and `#63` repeat a training cue tuple (a direct consequence of unmasking
  them). Test rows presenting an unseen combination: **19 of 22**.
- `#27`'s scenario says "stands up … stepping toward it" while motion is `walk` —
  the standing is a transition, not the sustained motion. Fine as-is.

---

## ✅ Verified correct — no action

- **#50's "VERIFY against recordings"** is resolved: the actors are seated with
  hands in lap and perform **no** thumbs down, so `gesture = idle` is right. (#51
  idle, #38 thumbs down and #57 task-focused also match their entries.)
- **The old S21/S28 collision is fixed.** Row #63 *is* the old `S28_F10`, correctly
  relabelled F04, and #51 was re-recorded gesture-free on 2026-07-28 so F10 keeps
  its own footage. `data/old/labels.csv`'s `recombination_pool` status is obsolete.
- **All 2,905 files readable** — 0 corrupt, 0 zero-byte.
- **62 live rows ↔ 62 folders, 1:1**; every folder's F-tag and context match its
  row; re-verified against `data/old/labels.csv` on all 23 migrated scenarios.
- **Zero cue collisions** — no two rows share an observable tuple with different
  intents. This is what deleting F09 bought.
- **CSVs match the .docx exactly** — 41,848 field comparisons, all 62 rows.

---

# ADDENDUM — independent audit 2026-08-03 (WIN-3060)

Re-verified the whole of `data/final_merged` after `clips.csv` / `scenarios.csv` /
`splits.csv` were rebuilt on Aug 1–2 (later than the body of this file).

## ✅ Resolved since 2026-08-01

- **`person_id` is now 100 % complete** — 2,869/2,869 in both `clips.csv` and
  `splits.csv`, with **zero disagreements** between them. (Was 1,097/2,870.)
  Actor-disjoint validation is now *possible* — see N1.
- **Disk ↔ CSV is exact**: 2,904 video files on disk, 2,904 rows, 0 orphans, 0
  missing. (Note the tree is mixed `.mp4` + `.MOV`; a `*.mp4`-only glob under-counts
  by 516 and looks alarming — it is not.)
- `splits.csv` exists: train 1,866 / test 1,003 usable clips.
- 35 duplicates correctly excluded (`usable=False`), splits.csv carries only usable.

## ⚠ NEW — open

### ✅ N1. Validation split — FIXED 2026-08-03
`splits.csv` has only `train` / `test`. Every training script therefore invents its
own val split ad hoc (e.g. the 2026-07-28 gap study used a random 20 % of *takes*).
Different scripts → different val → numbers that are not comparable across runs.
Needed: a third `val` value, carved from **train**, **actor-disjoint** and **grouped
by take** (the 3 simultaneous views of one take must never straddle a split).

### N2. Train and test share actors — by design, but must be stated
6 of 10 actors appear on both sides (P01 258/552, P02 249/521, P04 219/271,
P06 207/428, P07 34/22, P09 36/6 as test/train). This is *correct* for the V3
design — the split is by **scenario**, so the test set measures **compositional
generalisation only, not subject generalisation**. The V3 document itself suggests
reserving one actor as test-only to report a subject-independence score. P03 (48),
P05 (13), P08 (4) and P10 (1) are train-only and too small to serve as that actor.

### N3. Three test rows cannot measure generalisation
Validator `[C4]`: rows **40/41/57** are identical in cues, intent *and* action
(kitchen, neutral, idle, stand → F05/A06) and **38/63** likewise (kitchen, sad,
thumbs down, sit → F04/A05) — each group containing both a train and a test row.
Those test rows are memorisation checks, not generalisation tests. 19 of 22 test
rows present a genuinely unseen tuple; say "19 of 22" in the thesis, not "22".

### N4. Row #58 is 56 % train footage ⛔ (contaminates the test set)
`derived_rows.csv` marks it `crosses_split=True`: 24 of #58's 43 clips **are** row
#49's frames (train) re-used with emotion+gesture masked — and the mask *changes*
the intent (F01 → F06) rather than preserving it. Any headline test number computed
over #58 is partly measuring training frames. Until it is re-recorded, exclude #58
from headline test metrics or report with and without it.

### ✅ N5. `takes.csv` staleness — FIXED 2026-08-03
1,737 rows vs 1,736 distinct takes in `clips.csv`. The extra is `S28_F07` take 19,
whose `example_clip` (`S28_F07_c052`) exists nowhere in `clips.csv` — an orphan left
by de-duplication. Its `person_id` is also blank, and takes.csv is only 956/1,737
populated while clips/splits are 100 %. Regenerate it from `clips.csv`.

### ✅ N6. `21_validate_table.py` crash — FIXED 2026-08-03
`UnicodeEncodeError: 'charmap' codec can't encode '\u2192'` — the validator dies
*while printing its own error list*, so its output is unreadable unless run as
`PYTHONIOENCODING=utf-8 python scripts/21_validate_table.py`. One-line fix:
`sys.stdout.reconfigure(encoding="utf-8")`, or use `->` instead of `→`.

### N7. Short-clip count is larger than R5 states
**127 clips (4.4 %) are shorter than the 4 s aggregation window**, min 1.67 s —
not 38. (R5's figure is the RealSense-only subset, where it is 8.) With 4 s
aggregation now the primary path, this needs a stated policy: pad, use the clip's
full length, or exclude. Worst rows: S40_F05 (10), S36_F03 (9), S52_F01 (9),
S63_F04 (8), S45_F07 (8), S46_F08 (8).

### N8. `view` label is unreliable for phone_1080p (extends R4)
Only **289 of 1,015** `phone_1080p` clips are actually 1920×1080. The rest: 633 at
1280×720, 59 at 1024×576, 16 portrait 1080×1920, 14 portrait 576×1024, and singles
at 1120×576 / 784×576 / 800×576 / 832×464. Any per-view comparison silently mixes
four resolutions and two orientations. Add a true `resolution_class` column, or
rename the view.

### N9. RealSense — the deployment view — is thin on test
2 rows have **zero** RealSense (#40 train, #56 test); 6 rows have only 5–6, four of
them test rows (#53, #54, #55, #59, #60). RealSense is 275 of 1,003 test clips
(27 %). Reporting headline numbers on RealSense only leaves several test rows
statistically meaningless; reporting on all views mixes the resolution problem in N8.


---

## ✅ Fixes applied 2026-08-03

**N1 — validation split.** `scripts/23_build_splits.py` now writes a second column
`split` (train/val/test) alongside `split_design`. `test` is untouched; `val` is
carved from train **by actor**: `VAL_ACTORS = (P04, P03)` — 319 clips, 17.1% of
train, covering 30 of 40 train rows, emptying none. P04 was also `data/old`'s val
actor. Result: **train 1,547 / val 319 / test 1,003**. The script now self-checks
and prints: val is actor-disjoint from train ✅, no (scenario, actor) group
straddles splits ✅, 40/40 train rows retained ✅.

*Actor- rather than row-disjoint val is deliberate*: it satisfies the handover and
keeps every train row's cue tuple in training. Holding out rows would mirror the
test condition but shrink the tuple coverage the 2026-07-28 gap decomposition
identified as the binding constraint.

**Training code should read `split`, not `split_design`.**

**N5 — takes.csv.** New `scripts/24_regen_takes.py` reproduces script 20's takes
aggregation exactly, but touches nothing else, so it is safe to run whenever
`clips.csv` is amended on its own. Dropped the orphan `S28_F07` take 19; person_id
went **956 → 1,736 of 1,736**. `22_verify_csvs.py` now reports *"ALL 62 rows match
exactly — every table agrees with the .docx"* with no residual inconsistency.

**N6 — validator crash.** `21_validate_table.py` reconfigures stdout/stderr to
UTF-8 at import. It now runs to completion on a stock Windows console and reports
its 1 error + 13 warnings instead of dying while printing them.


## ✅ Decisions applied 2026-08-03 (N4, N7, N9)

All three are now **columns in `splits.csv`**, so downstream scripts filter
consistently instead of each re-deciding:

| Decision | Column | Effect |
|---|---|---|
| **N4** — exclude #58's borrowed frames from headline test | `headline_eval` (bool) | test 1,003 → **979 headline clips**; the 24 excluded are row #58's copies of #49's train footage. #58's own 19 clips remain. |
| **N9** — report all views | `resolution_class`, `orientation` | 480p 1,339 · 720p 633 · 4k 516 · 1080p 305 · 576p 76; landscape 2,833 · portrait 36. `view` is kept but is no longer the thing to group by. |
| **N7** — short clips use full length | `agg_span_s` | `min(duration_s, 4.0)`; 127 clips fall below 4 s (min 1.67 s). No padding. |

**Why N8 was judged not to threaten the results:** ASSESSMENT §4b measured a **36×
pixel increase (480p → 4K) changing accuracy by <0.04**, and motion was actually
*worse* at 1080p (framing, not resolution). Portrait is 36 clips (1.3 %), only 5 in
test. What was genuinely wrong was the **metadata**, now fixed. Remaining caveat:
`view` is confounded with context (RealSense 961 classroom/377 kitchen vs
phone_1080p 286/729), so a per-view table is never a clean resolution study.
