# T04 - context generalization on unseen environments

The v4 model is trained only on **Classroom** and **Kitchen**. These clips are **Hospital** and **Museum** - environments it has never seen - processed through the same cue pipeline as the training data, so this isolates the effect of a new environment. Because the context token only covers the two known rooms, context is marked *unobserved* for the novel environments and excluded from attention.

AI Kitchen clips are a **control**: identical generation process, but a context the model knows. 10 seeds, majority vote.

Per-seed accuracy over all AI clips: **0.9783 +/- 0.0076**.

| slice | clips | fusion | rules | diff |
|---|---|---|---|---|
| **Novel environments (Hospital + Museum)** | 118 | 0.9831 | 0.9407 | +0.0424 |
| - Hospital | 50 | 0.9800 | 0.9000 | +0.0800 |
| - Museum | 68 | 0.9853 | 0.9706 | +0.0147 |
| Control: AI Kitchen (known environment) | 11 | 1.0000 | 1.0000 | +0.0000 |
| All AI clips | 129 | 0.9845 | 0.9457 | +0.0388 |

## Per scenario

| scenario | environment | cue signature | intent | clips | fusion | rules |
|---|---|---|---|---|---|---|
| S64_F10 | museum | low / idle | F10 | 20 | 1.000 | 1.000 |
| S65_F07 | museum | hostile / thumbs_up | F07 | 10 | 1.000 | 0.800 |
| S66_F04 | hospital | low / thumbs_down | F04 | 10 | 1.000 | 1.000 |
| S67_F08 | hospital | hostile / thumbs_up | F07 | 10 | 1.000 | 0.900 |
| S68_F02 | museum | alarm / idle | F02 | 18 | 0.944 | 1.000 |
| S69_F06 | hospital | hostile / point | F06 | 11 | 0.909 | 0.727 |
| S70_F04 | museum | low / thumbs_down | F04 | 20 | 1.000 | 1.000 |
| S71_F01 | hospital | positive / wave | F01 | 10 | 1.000 | 1.000 |
| S72_F10 | hospital | low / idle | F10 | 9 | 1.000 | 0.889 |
| S73_F10 | kitchen | low / idle | F10 | 11 | 1.000 | 1.000 |

## Reference

Recorded test set (known environments): fusion **0.8202**, rules 0.7646.

Novel-environment accuracy of **0.9831** against **1.0000** on the AI Kitchen control isolates the cost of environment novelty at **-0.0169** - the remaining difference from the recorded-test figure is attributable to AI-generated footage rather than to the new environment.
