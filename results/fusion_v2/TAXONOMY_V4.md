# Phase 2 — perception-aware taxonomy v4 and gate

Keyed on **gesture** (most reliable), **emotion superclass** (`alarm={Fear,Surprise}`, `hostile={Anger,Disgust}`, `positive`, `low`, `neutral`) and **context**. **Motion is excluded entirely** — 0.435 accurate, and its measured value to intent is −0.013.

## Intents and actions (all actionably distinct — plan P4)

| intent | robot action |
|---|---|
| F01 | A01 greet/acknowledge |
| F02 | A02 emergency response |
| F03 | A04 go to task |
| F04 | A05 offer help |
| F05 | A06 do not interrupt |
| F06 | A11 yield/move aside |
| F07 | A08 de-escalate |
| F08 | A07 offer break |
| F10 | A12 encourage/re-engage |

## Gate

| split | n | v4 ceiling (measured cues) | v4 rules (measured cues) | v3 rules (reference) |
|---|---|---|---|---|
| val | 283 | **0.9505** | **0.8269** | 0.6996 |
| test | 773 | **0.9237** | **0.7646** | 0.7439 |

**Gate decided on val: ceiling 0.9505 (need ≥0.90), rules 0.8269 (need ≥0.80) → PASS**

Test is shown for information only; the design was not iterated against it.

## v4 class balance and rule recall (val)

| intent | n | rules recall |
|---|---|---|
| F01 | 35 | 0.657 |
| F02 | 30 | 1.000 |
| F03 | 6 | 1.000 |
| F04 | 31 | 0.774 |
| F05 | 23 | 0.522 |
| F06 | 38 | 1.000 |
| F07 | 20 | 0.950 |
| F08 | 53 | 0.943 |
| F10 | 47 | 0.681 |
