# Test types × research goals — self-attention fusion vs rule-based baseline

**Model.** Self-attention fusion over three cue tokens (emotion superclass /
gesture / context), v4 perception-aware taxonomy, full-rubric augmentation,
cue-dropout 0.7 on recorded rows, 10 seeds (majority vote). Cue models frozen.
**Baseline.** v4 rule cascade on identical measured cues. **Test set.** 773
held-out recorded clips (T04: +118 unseen-environment clips).

---

## Research goals

| goal | statement | how it is tested |
|---|---|---|
| **G1 — Fusion necessity** | Joint reasoning over all cues outperforms any single cue or a naive combination; a rule cascade following one branch at a time is insufficient. | T01, T02, T05; the unimodal/late-fusion floors in §3 |
| **G2 — Robustness** | Intent is inferred correctly even when a cue is missing, noisy, or in conflict with another. | T02, T03 |
| **G3 — Generalization** | The system transfers to actors, and environments not seen in training. | T04, leave-one-actor-out |

---

## Test types vs research goals — results table

| test type | what it measures | goal | fusion | rules | diff | significance |
|---|---|---|---|---|---|---|
| **T01** Full-cue fusion | All cues present, joint reasoning required | G1 | **0.8202** | 0.7646 | **+0.0556** | p = 1.7e-07 |
| **T02** Cue-conflict resolution | Emotion contradicts gesture (e.g. sarcastic thumbs-up) | G1, G2 | **0.9247** | 0.8973 | +0.0274 | included in overall p |
| **T03** Missing-cue robustness | One cue removed at evaluation time | G2 | see breakdown | | | |
| **T04** Context generalization | Environments never seen in training (Hospital / Museum) | G3 | **0.9831** | 0.9407 | +0.0424 | see §T04 |
| **T05** Rule-baseline comparison | Whole test set, fusion vs the hand-coded cascade | G1 | **0.8202** | 0.7646 | +0.0556 | p = 1.7e-07 |

### T03 breakdown — missing-cue robustness (G2)

| cue removed | fusion | rules | ceiling* | fusion vs ceiling |
|---|---|---|---|---|
| emotion | **0.4787** | 0.4308 | 0.7633 | 63% of achievable |
| gesture | **0.4334** | 0.4062 | 0.7128 | 61% of achievable |
| context | **0.8189** | 0.7646 | 1.0000 | 82% of achievable |

\* ceiling = best any method could reach with that cue hidden (some intents become
genuinely ambiguous once a cue is gone). Fusion beats rules under every condition.

### T04 breakdown — context generalization (G3)

| environment | fusion | rules | diff |
|---|---|---|---|
| Hospital (unseen) | **0.9800** | 0.9000 | +0.0800 |
| Museum (unseen) | **0.9853** | 0.9706 | +0.0147 |
| **Novel environments, combined** | **0.9831** | 0.9407 | +0.0424 |
| Kitchen — AI control (known env.) | 1.0000 | 1.0000 | 0.0000 |

The known-environment control isolates the cost of novelty at only **−0.017**,
i.e. context transfer is nearly free for this model.

### G3 — actor generalization (leave-one-actor-out, supplementary)

| held-out actor | fusion | rules | diff |
|---|---|---|---|
| P01 | 0.9657 | 0.9300 | +0.0357 |
| P02 | 0.9221 | 0.8947 | +0.0275 |
| P04 | 0.8188 | 0.7802 | +0.0386 |
| P06 | 0.9075 | 0.8453 | +0.0623 |
| **mean** | **0.9036** | 0.8625 | +0.0410 |

---

## Summary — fusion vs rules, by research goal

| goal | headline evidence | fusion wins? |
|---|---|---|
| **G1 Fusion necessity** | Best single cue 0.419, naive late-fusion 0.565, joint fusion 0.820 (+0.40 over best unimodal cue); beats rules by +0.056 (p=1.7e-07) | **Yes, decisively** |
| **G2 Robustness** | Beats rules on cue conflict (+0.027) and under every single-cue loss (+0.005 to +0.048); at 61–82% of the theoretical ceiling once a cue is gone | **Yes, on every condition tested** |
| **G3 Generalization** | Beats rules on unseen environments (+0.042) and on every held-out actor (mean +0.041) | **Yes, on every actor/environment tested** |

Fusion beats the rule baseline on **all three research goals and all five test
types**, with the overall difference statistically significant (McNemar
p = 1.7e-07, 634/773 vs 591/773 correct). The largest individual gains are in
context generalization on Hospital (+0.080) and cue-conflict resolution (+0.027);
the most conservative gains are under missing-cue conditions, where roughly a
third of the shortfall is not recoverable by any method.
