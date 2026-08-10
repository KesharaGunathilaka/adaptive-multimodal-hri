# T04 — context generalization, measured as a counterfactual swap

Generated 2026-08-08 05:44 · `data/final_merged` · headline test n=979 · 5-seed majority vote · recombination balance `uniform`.

Each test clip keeps its real emotion/gesture/motion cues; only the **context** cue is replaced by a real context vector from a val-split clip of the other room. The rubric's answer for the swapped tuple is the target.

## The structural problem with T04 on this rubric

Context changes the intent for **24 of 224 (emotion × gesture × motion) tuples (10.7%)** — and every one of them involves the gesture `raise_hand`. For all other gestures the rubric is context-invariant, so 89% of any T04 measurement is an invariance test that a model can pass by ignoring context entirely.

In the headline test split the flip subset is **n=52** clips (rows {25: 52}) against **n=927** invariant clips. `raise_hand` is also the gesture model's weakest class, so the only part of T04 with discriminative power rests on the least reliable cue.

## Results

| Measure | Fusion | Rules |
|---|---|---|
| Accuracy, original context (all clips) | 0.7099 | 0.7120 |
| **Invariant** (n=927): prediction unchanged after swap | 0.9946 | 1.0000 |
| **Invariant**: still correct after swap | 0.7303 | 0.7411 |
| **Flip** (n=52): correct BEFORE swap | 0.3077 | 0.1923 |
| **Flip**: prediction changed at all | 0.3077 | 0.1923 |
| **Flip**: followed the rubric to the other room's intent | 0.8269 | 0.4038 |
| **Flip: correct in BOTH rooms** (the honest metric) | **0.3077** | **0.1923** |

## Reading this

- **Invariance** is the safe half and both systems should score high; rules score 1.0 on 'unchanged' by construction for every non-`raise_hand` clip, since context enters `rule_intent` nowhere else. A high fusion score here means fusion has correctly learned NOT to over-use context.
- **Do not quote `flip_followed` on its own.** Row #25's classroom answer is F04 and its kitchen answer is F01 — and F01 is the majority intent overall. A model biased toward F01 therefore scores well on 'followed the rubric' *without tracking context at all*: it was simply already predicting F01 while still in the classroom, where that was wrong. The tell is that `flip_changed_at_all` equals `flip_correct_before` exactly — the only predictions that moved are the ones that had been right beforehand.
- **`flip_correct_both` is the metric that means something**: right in the original room AND right after the swap. That requires actually reading context, and it cannot be won by class bias.
- **Flip** is the only part that tests G3's 'meaning flips with environment' claim, and it is both small (n=52) and concentrated on one row in one direction. Treat it as directional evidence, not a headline number.
- The honest conclusion for the thesis is that **T04 cannot strongly support or refute context generalization on this label rubric**. Strengthening it needs a rubric where context matters for more than one gesture — a dataset-design change, not a model change.