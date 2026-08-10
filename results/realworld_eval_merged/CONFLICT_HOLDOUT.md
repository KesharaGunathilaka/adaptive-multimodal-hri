# Conflict-holdout generalization test (v2)

Generated 2026-08-07 14:08 · `data/final_merged` · 3-seed ensembles · 284 aligned + 164 conflicting combos overall; 305 "always trained" (aligned + whatever's in real training) vs 143 truly novel (zero presence in training, real or synthetic).

v1 (WORKLOG 2026-08-07) excluded ALL conflicting combos, including 753 real training clips -- this made 4 of 9 intent classes (F02/F07/F08/F10) entirely absent from training, producing an uninformative 0.0% ('never seen this class at all', not a generalization measurement). v2 fixes this: real training is used unfiltered, and recombination is allowed to reinforce any conflicting combo already present in real training data. Only combos with ZERO training presence at all are excluded.

**`restricted`**: trained on real data (unfiltered) + synthetic recombination limited to `always_trained` combos.
**`full_reference`**: standard recipe, all 448 combos, retrained fresh for a clean paired comparison.

| Test set | n | restricted | full_reference | rules |
|---|---|---|---|---|
| real_novel | 372 | 0.2849 / 0.1564 | 0.6828 / 0.3238 | 0.7634 / 0.3962 |
| real_seen | 401 | 0.818 / 0.3272 | 0.7955 / 0.3248 | 0.7257 / 0.3139 |

(cells are accuracy / macro-F1)

## Reading this

On real NOVEL-combo test clips (n=372, true combo never seen in training, real or synthetic) -- `restricted` scores 0.2849, `full_reference` scores 0.6828 (it saw these combos directly), rules score 0.7634. The `full_reference`-minus-`restricted` gap (+0.3979) is the genuine generalization cost of not having a labelled example of this specific combo -- with every class still represented in `restricted`'s training, so this measures compositional generalization, not label-set coverage.

On real SEEN-combo test clips (n=401) -- both fusion models had training exposure -- `restricted` scores 0.818 vs `full_reference`'s 0.7955, expected to be much closer.