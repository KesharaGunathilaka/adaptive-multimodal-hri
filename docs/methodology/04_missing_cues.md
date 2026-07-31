# Stage 4 — Missing cues: how absence travels through the pipeline

**The question this stage answers:** the dataset table marks some scenarios as
having a cue `[MISSING]`. How does that reach the fusion model, given that the
feature extractor always runs all four perception models on every video?

**The short answer:** *missingness is never extracted — it is a flag.* The
extractor's job is to report what it saw; deciding that a cue should be ignored
happens one layer up. You therefore **cannot "miss a cue" during recording**, and
you never need to record a special video to represent a missing cue.

---

## 4.1 Three independent channels, often confused

A cue can be absent for three completely different reasons, and the pipeline keeps
them in three separate places. Conflating them is the most common way to
mis-measure robustness.

| # | Channel | Where it lives | Set by | Meaning |
|---|---|---|---|---|
| 1 | **Designed-missing** | `missing_v3` in `scenarios_v3.csv` (from Final_Dataset.docx) | The dataset table | *This scenario is about a sensor being unavailable.* A property of the row, known before any video is shot. |
| 2 | **Runtime-missing** | `emo_obs` / `ges_obs` / `mot_obs` / `ctx_obs` in the feature table | The extractor, per window | *The model produced nothing here* — no face detected, no pose detected. A measurement, not a label. |
| 3 | **Synthetic-missing** | modality dropout in `fusion/model/datasets.py` | Training augmentation | *Pretend this cue was unavailable.* Train-only, never in eval. |

Channel 2 is the only one the extractor produces. Channels 1 and 3 are metadata
and augmentation respectively — they never touch a video.

## 4.2 What the extractor actually does with a designed-missing row

Nothing special, and that is deliberate (handover §7.1: *"still run the model if
the video allows, but set the missing flag"*).

```
video ──► all four models always run ──► 4 probability vectors + 4 obs flags
                                             │
                    scenarios_v3.csv ────────┤  joined by v3_row
                                             ▼
                              feature table row  (probs, obs, missing_designed)
                                             │
                                             ▼
                       fusion input: obs=0  ⇒  that modality's token is dropped
```

The masking decision is made at the **fusion input**, from
`missing_designed OR NOT observed`. In `fusion/model/model.py` this is
`missing_mode='exclude'`: the masked modality's token is removed from the
attention key-padding mask, so the transformer marginalises over it rather than
reading a zero vector. (Phase 2 established that this beats a learned `[MISSING]`
token — see `results/fusion_v1/RESULTS.md`.)

**Why running the model anyway is the right call, not laziness:** it gives a free
check on the recording. If V3 #22 says the face is occluded by a held book, then
the emotion model *should* mostly fail to find a face on those clips. Section 4 of
`results/realworld_eval_final/unimodal/UNIMODAL_classroom.md` reports the
observation rate on exactly those rows. A designed-missing row whose cue is still
observed 95 % of the time means **the recording did not realise the design**, and
we would rather find that out from a number than discover it in the final results.

## 4.3 At deployment there is no table

On the Jetson nothing knows the scenario, so channel 1 does not exist and channel
2 carries the whole load. `jetson_deploy/fusion/pipeline.py` sets `obs[k] = 1.0`
only when cue *k* returned a vector this step — i.e. detector-driven. That is the
same input format the fusion head was trained on, which is why designed-missing
rows must be represented as `obs=0` and not as some special class: **training
missingness and deployment missingness have to be the same signal.**

One gap worth noting: the runtime currently masks only on *detector failure*, not
on *low confidence*. A face that is detected but yields a flat 7-way softmax is
still passed through as observed. Confidence-thresholded masking is an available
improvement (handover §8.1 anticipates it) and needs a τ tuned on val.

## 4.4 Derived rows — V3 #6 from #5, and V3 #9 from #8

Two classroom rows were deliberately not filmed, because they describe **the same
human behaviour with a sensor turned off**:

| Row | Cues | Missing | Intent | Recorded? |
|---|---|---|---|---|
| #5 | classroom, neutral, beckoning, sit | — | F03 / A04 | ✅ 59 clips |
| #6 | classroom, neutral, beckoning, sit | context | F03 / A04 | ❌ derived from #5 |
| #8 | classroom, sad, thumbs down, sit | — | F04 / A05 | ✅ 61 clips |
| #9 | classroom, sad, *(hands below desk)*, sit | gesture | F04 / A05 | ❌ derived from #8 |

**This is the correct approach, and here is the precise reason.** Between #5 and
#6 the pixels are not what differs — the *room-identity sensor being offline* is
what differs, and that is a state of the perception stack, not of the scene. There
is no video you could shoot of "the context sensor is offline". Masking the
context token on #5's clips is not an approximation of #6; it **is** #6. The same
holds for #9 with the gesture channel.

The spec is explicit and machine-readable in
`data/final/annotations/derived_rows.csv`, so the derivation is reproducible
rather than folklore.

### The four rules that keep this honest

1. **Train rows only.** #6 and #9 are both `split_design=train`. A *test* row is
   never derived: it would measure the fusion head's response to synthetic
   masking instead of to a real occlusion. This costs nothing here — the classroom
   test rows that carry missing cues (**#22, #23, #24, #25, #30**) are all
   genuinely recorded, so T03 stays a real measurement.
2. **Splits follow the source clip.** A derived sample carries its parent's
   `clip_id` lineage and subject, so the same pixels can never land in train as
   #5 and in val as #6. Split by subject, never randomly.
3. **Never scored.** Derived samples are training material. They appear in no
   reported accuracy, and are flagged `synthetic_missing=True` for that purpose.
4. **Check the masked label is still unique** — the subtle one, next section.

### The trap in row #9

Mask gesture on `(classroom, sad, sit)` and you get an input that matches **two**
V3 rows with **different intents**:

| Row | Gesture | Intent |
|---|---|---|
| #9 | *masked* | **F04** (needs help) |
| #20 | `none` → model class `idle` | **F10** (disengaged / upset) |

They are separable *only* by `gesture_obs`: #9 has the token dropped, #20 has it
present and predicting `idle`. The fusion model can learn that, but two
consequences follow:

- **Modality dropout must not mask gesture on rows #20/#21.** If it does, those
  samples become byte-identical to #9's inputs with a conflicting label — the
  augmentation would be injecting label noise. Dropout should be
  *collision-aware*: for each row, only mask cues whose removal leaves the intent
  recoverable, which `scripts/17_cue_ambiguity.py` computes from the table.
- `#6` has no such problem — masking context on `(neutral, beckoning, sit)`
  collides with nothing. Context is the safest cue to mask in this dataset,
  which is consistent with the ceiling table below.

## 4.5 The table's own ceiling (`scripts/17_cue_ambiguity.py`)

Fusion cannot beat the label table. Where two rows share a cue tuple and disagree
on intent, the best possible policy is to answer the majority intent. Weighted by
clips actually on disk (classroom, RealSense view):

| Cue(s) masked | train ceiling | test ceiling |
|---|---|---|
| none | 0.977 | **0.900** |
| context | 0.977 | 0.900 |
| motion | 0.977 | 0.900 |
| emotion | 0.753 | 0.800 |
| gesture | 0.691 | 0.900 |
| emotion + gesture | 0.407 | 0.400 |
| gesture + motion | 0.592 | 0.600 |

Two things to read off this table:

1. **Even with all four cues, the classroom test set caps at 0.90.** Rows #22
   (F01 greeting) and #30 (F09 farewell) are the *identical* tuple — classroom,
   emotion `[MISSING]`, wave, walk — and differ only in whether the student is
   walking *toward the robot* or *toward the door*. **Direction** is the
   disambiguator, and no model outputs it (`DATASET_STATUS.md` §5 deferred it for
   v1). The same pair exists in train as #1 vs #18. That is 16 of 160 classroom
   test clips that are a coin flip by construction.
2. **The known "emotion masked → −28 points" result is mostly the table, not the
   model.** The ceiling under emotion masking is 0.753 on train rows. A fusion
   model at ~0.67 there is close to optimal. Report every masking number against
   its ceiling.
