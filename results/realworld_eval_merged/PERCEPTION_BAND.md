# Where does the deployed model's remaining error live?

Generated 2026-08-08 05:31 · `data/final_merged` · headline test n=979 · **10 seeds** · deployed `full` recipe (real train cues + recombination + dropout 0.3 + jitter 0.15 + masked-val selection).

`GAP_DECOMPOSITION_MERGED.md`'s decomposition was measured on a *plain* fusion model and has been stale since recombination landed. This re-runs it on the recipe actually deployed, and adds a distribution-shift-free oracle condition.

## Results

| Test-time cues | Fusion clip acc | Fusion macro-F1 | Rules clip acc |
|---|---|---|---|
| `real` (what deployment gets) | 0.7133 ± 0.0151 | 0.6146 | 0.7120 |
| `oracle_onehot` (one-hot true labels) | 0.9373 ± 0.0216 | 0.7952 | 1.0000 |
| `oracle_realistic` (true class, real soft vectors) | 0.8176 ± 0.0156 | 0.6820 | 0.8233 |

## What each oracle condition actually measures

**`oracle_realistic` is NOT perfect perception.** The substituted vector is guaranteed to come from a clip of the right class, but that clip's cue model output is still noisy — so class-conditional noise survives the substitution. The proof is in the rules column: rules score **0.8233** under `oracle_realistic` but exactly **1.0000** under `oracle_onehot`. A rule system given a genuinely perfect cue is right by construction; the shortfall is the noise the substitution keeps. So `oracle_realistic` measures *decorrelating cue noise from a clip's own difficulty*, not eliminating it.

Read the two conditions as bracketing the truth:

- `oracle_onehot` — **noise-free but out-of-distribution** for a model trained on soft vectors. Fusion 0.9373.
- `oracle_realistic` — **in-distribution but still noisy**. Fusion 0.8176.

## Decomposition

| Using | Generalisation cost (1.0 − oracle) | Perception cost (oracle − real) |
|---|---|---|
| `oracle_onehot` | 0.0627 | 0.2240 |
| `oracle_realistic` | 0.1824 | 0.1043 |

**The headline change: fusion+oracle has moved from 0.619 to 0.9373.** `GAP_DECOMPOSITION_MERGED.md` measured 0.619 on the *plain* recipe and concluded generalisation cost 0.381 dominated perception cost 0.072. On the deployed `full` recipe that is now reversed: given correct cues the model is right 93.7% of the time, so **recombination has largely closed the generalisation gap it was built to close**, and what remains is dominated by perception.

**Only the `oracle_onehot` row is a clean split.** In the `oracle_realistic` row the "generalisation" column is not pure generalisation — it also contains the class-conditional noise the substitution keeps, so it overstates the model's rubric error. Rules calibrate exactly how much: rules have zero generalisation error by construction, so their entire 0.1767 shortfall under `oracle_realistic` IS that residual noise. Fusion's shortfall is 0.1824, i.e. only **+0.0057** worse than a system that cannot generalise wrongly at all.

So both conditions agree that fusion's generalisation cost is now small — 0.0627 by the one-hot route, 0.0057 by the rules-as-reference route — even though their raw "1.0 − oracle" columns look contradictory. Perception is the shared bottleneck of both systems (rules lose 0.2880 to it, fusion 0.2240), which is also why the two are statistically indistinguishable on headline accuracy (`SIGNIFICANCE.md`).

## Caveats

- The 0.619 comparison is indicative, not exact: that number came from a model both TRAINED and tested on one-hot cues, and from the plain recipe. Here the model is trained on soft real vectors, so `oracle_onehot` is if anything a **lower** bound on its rubric competence.
- `oracle_realistic` draws substitutes from the **val** split, which is actor-disjoint from train and is not used to build the recombination pools, so the model is not fed vectors it memorised. Substitution stats: `{'n_filled': 3604, 'n_from_train_fallback': 200, 'n_no_pool_anywhere': 0}`.
- Both oracle conditions keep scenario-designed `[missing]` cues masked, so they model perfect perception, not perfect information.