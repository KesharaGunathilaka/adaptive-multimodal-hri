# T01-T05 scenario test report — deployed fusion (self-attn, `full`) vs rules

Generated 2026-08-06 03:15 · `data/final_merged` · headline test clips · model = self-attention, `full` recombination recipe (same as `RECOMBINATION.md`/`FUSION_CONFUSION.md`), 3 seeds, majority vote.

## T01 — full-cue fusion accuracy / T05 — rule-baseline comparison

> **CORRECTED 2026-08-08 — the original claim in this section did not hold.** It read
> "Fusion beats rules by 0.0071 acc — the G1/T05 claim holds", based on a 3-seed mean of
> 0.7191 ± 0.0164. The margin was less than half its own standard deviation, and re-running
> with 10 seeds shows the 3-seed figure was a small-sample artifact: seeds (0,1,2) happen to
> include seed 0 (0.7416), the best of 10 and the *only* seed that beats rules significantly.
> See `SIGNIFICANCE.md` for the full test.

| | Clip acc | Macro-F1 |
|---|---|---|
| Fusion (self-attention, `full`), mean ± std over **10 seeds** | 0.7133 ± 0.0160 | 0.6146 ± 0.0212 |
| Fusion, 10-seed majority vote | 0.7130 | 0.6140 |
| Rule-based baseline | 0.7120 | **0.6414** |
| *(superseded)* Fusion, 3 seeds (0,1,2) | *0.7191 ± 0.0164* | *0.6241* |

**The G1/T05 claim does NOT hold on overall accuracy.** McNemar's exact test on the
10-seed ensemble vs rules: p=1.0000 (n10=73, n01=72) — statistically indistinguishable.
Bootstrap over clips gives a mean difference of +0.0009 with 95% CI [−0.0235, +0.0255],
containing zero. Only **1 of 10 seeds** beats rules significantly; **4 of 10 favour rules**.
The fusion accuracy 95% CI over seeds, [0.7019, 0.7247], contains rules' 0.7120.

Note also that **rules beat fusion on macro-F1** (0.6414 vs 0.6146) by a wider margin than
fusion's accuracy edge — and macro-F1 is the metric that matters for the rare, safety-relevant
intents. This is expected: V3 intent labels are `rule_intent()`'s own output, so rules+oracle
= 1.000 by construction (`GAP_DECOMPOSITION_MERGED.md`) and overall accuracy is an arena that
structurally favours the rule baseline. T05 should be argued on the axes where the two systems
genuinely diverge — missing-cue robustness (T03a below) and F02 recall (`F02_RECALL_FIX.md`) —
not on this table.

(T02/T03/T04's per-row tables below use a 3-seed MAJORITY VOTE ensemble — a smoother, related
but distinct construction, needed to get one clean prediction per clip rather than 3 separate
accuracy numbers. Those per-row numbers inherit the same 3-seed limitation flagged above and
should be read as directional.)

## T02 — cue-conflict resolution (same gesture, different meaning)

Per-scenario-row accuracy for gesture families where the SAME gesture maps to different intents depending on emotion/context/motion — exactly the case a naive single-cue rule ("gesture=X always means Y") gets wrong and joint reasoning is required.

| Gesture | Row | Context | Emotion | Motion | Expected | n | Fusion acc | Rule acc |
|---|---|---|---|---|---|---|---|---|
| [missing] | #23 | classroom | fear | walk | F02 | 44 | 0.5227 | 0.8636 |
| [missing] | #53 | kitchen | fear | walk | F02 | 39 | 0.3846 | 0.5385 |
| [missing] | #58 | kitchen | [missing] | walk | F06 | 19 | 0.9474 | 0.9474 |
| beckoning | #24 | classroom | neutral | walk | F03 | 53 | 0.9057 | 0.6038 |
| beckoning | #56 | kitchen | sad | stand | F04 | 34 | 0.5588 | 0.5294 |
| both hands up | #26 | classroom | neutral | sit | F05 | 52 | 0.9615 | 0.8846 |
| idle | #31 | classroom | sad | step back | F10 | 52 | 0.8462 | 0.7885 |
| idle | #54 | kitchen | fear | stand | F02 | 38 | 0.6579 | 0.7632 |
| idle | #57 | kitchen | neutral | stand | F05 | 28 | 0.5714 | 0.5357 |
| idle | #62 | kitchen | sad | step back | F10 | 42 | 0.8095 | 0.881 |
| point | #27 | classroom | angry | walk | F06 | 53 | 0.0377 | 0.6415 |
| point | #55 | kitchen | happy | walk | F03 | 42 | 0.381 | 0.5714 |
| raise hand | #25 | classroom | [missing] | step back | F04 | 52 | 0.3077 | 0.1923 |
| thumbs down | #60 | kitchen | disgust | sit | F08 | 41 | 0.8049 | 0.7317 |
| thumbs down | #63 | kitchen | sad | sit | F04 | 46 | 0.8696 | 0.8696 |
| thumbs up | #28 | classroom | angry | sit | F07 | 55 | 0.7455 | 0.7273 |
| thumbs up | #29 | classroom | disgust | sit | F08 | 49 | 0.7551 | 0.6939 |
| thumbs up | #32 | kitchen | happy | stand | F01 | 54 | 0.8148 | 0.8333 |
| thumbs up | #59 | kitchen | angry | stand | F07 | 42 | 0.9286 | 0.9286 |
| wave | #18 | classroom | happy | walk | F01 | 48 | 0.875 | 0.5625 |
| wave | #22 | classroom | [missing] | walk | F01 | 52 | 1.0 | 0.6731 |
| wave | #61 | kitchen | sad | walk | F01 | 44 | 1.0 | 1.0 |

## T03a — missing-cue robustness (simulated masking sweep)

Each modality (then each pair) force-masked at eval time; compares fusion's learned missing-cue handling against rules' fixed-default fallback (`emotion->Neutral, gesture->idle, motion->standing`). **Context masking does not affect rules** — the rule system always uses the clip's true context (a fixed-installation assumption), unlike fusion which must infer it like any other cue (`scripts/realworld_eval/merged_gap.py`'s `rule_predict` docstring).

| Masked | Fusion acc | Rule acc |
|---|---|---|
| none | 0.7191 | 0.712 |
| emotion | 0.4147 | 0.4239 |
| gesture | 0.4637 | 0.2819 |
| motion | 0.6874 | 0.6425 |
| context | 0.7058 | 0.712 |
| emotion+gesture | 0.2421 | 0.1328 |
| emotion+motion | 0.4045 | 0.3892 |
| emotion+context | 0.3963 | 0.4239 |
| gesture+motion | 0.3953 | 0.2298 |
| gesture+context | 0.4545 | 0.2819 |
| motion+context | 0.6813 | 0.6425 |

**No formal table-imposed ceiling was recomputed for `final_merged` in this pass** (`scripts/17_cue_ambiguity.py` computed one for the old table only — `07_evaluation.md` §7.5's warning to always read a masking number against its ceiling still applies; treat the raw numbers above as directional until that's rebuilt).

## T03b — REAL scenario-designed missing-cue rows (not simulated)

Rows #9/12/22/23/25/43/49/53/58 have a cue marked `[missing]` in the V3 table itself (the scenario was filmed so that cue genuinely isn't recoverable) — a stronger test than the simulated sweep above.

| Designed-missing? | n clips | Fusion acc | Rule acc |
|---|---|---|---|
| no (full cues) | 773.0 | 0.7426 | 0.7439 |
| yes (designed [missing]) | 206.0 | 0.6019 | 0.5922 |

Observation rate on those rows — does the detector actually fail, or find something anyway (the §7.5 honesty check)?

| Row | Designed-missing cue | Still observed | n clips |
|---|---|---|---|
| #22 | emotion | 100.0% | 52 |
| #23 | gesture | 100.0% | 44 |
| #25 | emotion | 100.0% | 52 |
| #53 | gesture | 100.0% | 39 |
| #58 | emotion | 100.0% | 19 |
| #58 | gesture | 100.0% | 19 |

## T04 — context generalisation

Of the V3-table (emotion, gesture, motion) tuples recorded in BOTH contexts, **1 flip intent** when context changes and **16 stay invariant** — both are valid rubric behaviours; the question is whether fusion tracks context correctly in each case.

| Emotion | Gesture | Motion | Context | Row | Intent | n | Fusion acc | Rule acc |
|---|---|---|---|---|---|---|---|---|
| [missing] | wave | walk | classroom | #22 | F01 | 52 | 1.0 | 0.6731 |
| [missing] | wave | walk | kitchen | #49 | F01 | 0 | nan | nan |
| angry | both hands up | stand | classroom | #14 | F07 | 0 | nan | nan |
| angry | both hands up | stand | kitchen | #44 | F07 | 0 | nan | nan |
| disgust | thumbs down | sit | classroom | #17 | F08 | 0 | nan | nan |
| disgust | thumbs down | sit | kitchen | #60 | F08 | 41 | 0.8049 | 0.7317 |
| disgust | thumbs down | step back | classroom | #16 | F08 | 0 | nan | nan |
| disgust | thumbs down | step back | kitchen | #46 | F08 | 0 | nan | nan |
| fear | [missing] | walk | classroom | #23 | F02 | 44 | 0.5227 | 0.8636 |
| fear | [missing] | walk | kitchen | #53 | F02 | 39 | 0.3846 | 0.5385 |
| fear | both hands up | step back | classroom | #4 | F02 | 0 | nan | nan |
| fear | both hands up | step back | kitchen | #34 | F02 | 0 | nan | nan |
| happy | raise hand | sit | classroom | #11 | F05 | 0 | nan | nan |
| happy | raise hand | sit | kitchen | #52 | F01 | 0 | nan | nan |
| happy | wave | walk | classroom | #1 | F01 | 0 | nan | nan |
| happy | wave | walk | classroom | #18 | F01 | 48 | 0.875 | 0.5625 |
| happy | wave | walk | kitchen | #48 | F01 | 0 | nan | nan |
| neutral | [missing] | walk | classroom | #12 | F06 | 0 | nan | nan |
| neutral | [missing] | walk | kitchen | #43 | F06 | 0 | nan | nan |
| neutral | beckoning | walk | classroom | #24 | F03 | 53 | 0.9057 | 0.6038 |
| neutral | beckoning | walk | kitchen | #36 | F03 | 0 | nan | nan |
| sad | idle | sit | classroom | #20 | F10 | 0 | nan | nan |
| sad | idle | sit | kitchen | #50 | F10 | 0 | nan | nan |
| sad | idle | stand | classroom | #21 | F10 | 0 | nan | nan |
| sad | idle | stand | kitchen | #51 | F10 | 0 | nan | nan |
| sad | idle | step back | classroom | #31 | F10 | 52 | 0.8462 | 0.7885 |
| sad | idle | step back | kitchen | #62 | F10 | 42 | 0.8095 | 0.881 |
| sad | thumbs down | sit | classroom | #8 | F04 | 0 | nan | nan |
| sad | thumbs down | sit | kitchen | #38 | F04 | 0 | nan | nan |
| sad | thumbs down | sit | kitchen | #63 | F04 | 46 | 0.8696 | 0.8696 |