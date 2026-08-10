# F02 (emergency) recall fix — class-weighted loss

Generated 2026-08-07 00:53 · `data/final_merged` · headline test · isolated intervention: ONLY F02's loss weight changes, every other class stays at 1.0.

**Reference (Phase 1, `ROBUSTNESS_BATTERY.md` part C):** rules are a fixed, non-adjustable point at precision=0.481, recall=0.727. The unweighted `full` model's F02 recall ceiling across its entire threshold sweep was 0.645 — never reaching rules' 0.727, at any threshold. That is what this fix targets.

| F02 weight | Headline acc | Headline macro-F1 | F02 recall ceiling | F02 recall @ rules' precision | Beats rules' recall (0.727)? |
|---|---|---|---|---|---|
| 1.0 | 0.7191 ± 0.0164 | 0.6241 | 0.645 | 0.645 | no |
| 2.0 | 0.7102 ± 0.0067 | 0.6129 | 0.719 | 0.719 | no |
| 3.0 | 0.7045 ± 0.0093 | 0.6029 | 0.736 | 0.736 | **YES** |
| 5.0 | 0.7031 ± 0.0175 | 0.6044 | 0.818 | 0.678 | **YES** |


Best F02 recall ceiling: **0.818** at weight=5.0 (now beats rules' 0.727). Headline accuracy cost vs the unweighted baseline: -0.0160 (0.7191 -> 0.7031).