# WP2 - taxonomy ablation

Model, augmentation budget (~24000 synthetic rows), seed list (10 seeds) and evaluation protocol are identical across arms. Only the LABEL SPACE changes. Each arm gets its own rule baseline and its own measured-cue ceiling, since those are properties of the taxonomy.

| taxonomy | what changed | fusion | macro-F1 | rules | ceiling | vs v4 |
|---|---|---|---|---|---|---|
| `v3` | original rubric: 4 cues, fine emotion, anger/disgust split | **0.7624** | 0.7239 | 0.7653 | 0.9519 | -0.0671 (p=1.36e-05) |
| `v4` **(ours)** | ours: gesture-first, superclasses, no motion | **0.8294** | 0.8010 | 0.7886 | 0.9402 | - |
| `v4-emosplit` | revert F07/F08 to the anger-vs-disgust split | **0.7259** | 0.6983 | 0.7697 | 0.9257 | -0.1035 (p=5.9e-15) |
| `v4-finealarm` | revert alarm superclass: only Fear triggers F02 | **0.8324** | 0.8001 | 0.7828 | 0.9402 | +0.0029 (p=0.727) |
| `v4-withmotion` | re-admit motion as a discriminator | **0.8120** | 0.7812 | 0.7799 | 0.9650 | -0.0175 (p=0.0169) |

Per-seed: `v3` 0.7554+/-0.0149, `v4` 0.8257+/-0.0087, `v4-emosplit` 0.7216+/-0.0073, `v4-finealarm` 0.8294+/-0.0086, `v4-withmotion` 0.8055+/-0.0199

The `ceiling` column is the key diagnostic: it is the best score ANY method could reach on that taxonomy given the cues the perception stack actually delivers. A taxonomy that places intent boundaries on unreliable cue distinctions lowers its own ceiling before any model is trained.
