"""Rubric-driven cue recombination for `data/final_merged` — the fix motivated
by the 2026-08-03 gap decomposition (`results/realworld_eval_merged/
GAP_DECOMPOSITION_MERGED.md`): with perfect cues the fusion head still only
reaches 0.619 against a 1.0 ceiling, because it trains on the ~40 cue tuples
the 62 recorded V3 rows contain and cannot extrapolate the rubric to the
combinations it never saw. Rules generalise this instantly (rules+oracle =
1.000) because the rubric is *explicit* in `rule_intent()`; fusion has to
infer it from examples.

The idea: teach the fusion head the SAME explicit rubric, but do it by
training on real, noisy cue vectors labelled by the rubric — not on clean
one-hot vectors. One-hot training would just reproduce the rule system's own
brittleness-free-but-noise-free behaviour (and the classroom-only ablation
already showed a learned [MISSING]-token equivalent of that — training on
"clean" representations — did not transfer to noisy ones). Sampling real
vectors keeps the noise robustness that is fusion's actual advantage over
rules (`docs/methodology/06_fusion_model.md` §6.3), while the combinatorial
coverage supplies the semantics fusion currently lacks.

Design, and how it differs from `fusion/model/recombine.py` (the `data/old`
version):

* **Full combinatorial span, not a curated row list.** `data/old`'s
  `SYNTH_ROWS` hand-picked ~19 unrecorded V3 rows. Here every
  context(2) x emotion(7) x gesture(8) x motion(4) = 448 combination is
  generated and labelled by `rule_intent()` (with the F09->F01 remap the
  merged table needs — `merged_gap.rule_predict`). This explicitly includes
  the F02 (emergency) combinations the gap decomposition found the fusion
  head fails even with perfect cues (rows #23/#53/#54) — those combinations
  were originally reachable only by chance sampling from ~40 recorded tuples.
* **Pools are GROUND-TRUTH-indexed, not argmax-indexed.** `data/old`'s
  `build_pools` bucketed real vectors by the MODEL's OWN argmax
  (`am = vals.argmax(1)`), which self-selects vectors the model already
  classifies "correctly" (by its own lights) and under-represents exactly the
  confusable cases recombination should be teaching the model to handle.
  Bucketing by the clip's TRUE label (`gt_<modality>`) is a more principled
  noise model: "here is what a genuine occurrence of thumbs_down really looks
  like as a probability vector, including the times it gets confused with
  point" — the actual deployment distribution for that class.
* **Clip-level, not window-level**, matching the 2026-07-28 aggregation
  decision — pools are built from `merged_gap.build_real`'s per-clip mean
  vectors, TRAIN split only.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..baselines import common
from ..baselines.rule_based import rule_intent
from .datasets import CUE_SLICES  # noqa: F401  (documents the 24-dim layout)

LABELS = {"emo": common.EMO_COLS, "ges": common.GES_COLS,
          "mot": common.MOT_COLS, "ctx": common.CTX_COLS}
CLASS_NAME = {m: [c.split("_", 1)[1] for c in cols] for m, cols in LABELS.items()}
COL_OFFSET = {"emo": 0, "ges": 7, "mot": 15, "ctx": 19}

# rule_intent's vocabulary uses the table's wording ("both hands up", "step
# back", lowercase emotions) -- map each modality's native class name to it.
TO_RULE_WORD = {
    "emo": {"Surprise": "surprise", "Fear": "fear", "Disgust": "disgust",
            "Happy": "happy", "Sad": "sad", "Anger": "anger", "Neutral": "neutral"},
    "ges": {c: c for c in CLASS_NAME["ges"]},          # already rule_intent's spelling
    "mot": {"sitting": "sitting", "standing": "standing", "walking": "walking",
           "stepping_back": "step back"},
    "ctx": {"classroom": "classroom", "kitchen": "kitchen"},
}


@dataclass
class Report:
    n_combos: int
    n_generated: int
    n_skipped_empty_pool: int
    skipped: list[tuple]
    label_counts: dict


def build_pools(real_train: pd.DataFrame, clips: pd.DataFrame) -> dict:
    """(modality, class) -> [n, dim] array of real per-clip mean vectors, for
    clips whose GROUND TRUTH is that class (see module docstring for why
    ground-truth indexing, not argmax indexing)."""
    gt = clips.set_index("clip_id")
    pools: dict[tuple[str, str], np.ndarray] = {}
    gt_col = {"emo": "gt_emotion", "ges": "gt_gesture", "mot": "gt_motion",
             "ctx": "gt_context"}
    mask_col = {"emo": "emotion_masked", "ges": "gesture_masked",
               "mot": "motion_masked", "ctx": "context_masked"}
    for m in ("emo", "ges", "mot", "ctx"):
        cols = LABELS[m]
        labels = gt.reindex(real_train.clip_id)[gt_col[m]].to_numpy()
        masked = gt.reindex(real_train.clip_id)[mask_col[m]].to_numpy()
        obs = real_train[f"{m}_obs"].to_numpy().astype(bool)
        vecs = real_train[cols].to_numpy(np.float32)
        usable = obs & ~masked
        for cls in CLASS_NAME[m]:
            sel = usable & (labels == cls)
            if sel.any():
                pools[(m, cls)] = vecs[sel]
    return pools


# Restrict context to the two environments this dataset actually has (real
# pools and table coverage exist for neither of the other three CLIP context
# classes) -- explicit here rather than relying on generate()'s empty-pool
# skip to filter them out, which would silently iterate 672 dead combos.
REAL_CONTEXTS = ("classroom", "kitchen")


def all_combos() -> list[tuple[str, str, str, str]]:
    """context x emotion x gesture x motion — the full 2*7*8*4 = 448-tuple
    space this dataset can generate a grounded (real-pool-backed) label for."""
    return list(itertools.product(REAL_CONTEXTS, CLASS_NAME["emo"],
                                  CLASS_NAME["ges"], CLASS_NAME["mot"]))


def label_combo(ctx: str, emo: str, ges: str, mot: str) -> str:
    e = TO_RULE_WORD["emo"][emo]
    g = TO_RULE_WORD["ges"][ges]
    m = TO_RULE_WORD["mot"][mot]
    c = TO_RULE_WORD["ctx"][ctx]
    intent = rule_intent(e, g, m, c)
    return "F01" if intent == "F09" else intent      # merged-table remap


def generate(pools: dict, n_per_combo: int = 150, seed: int = 0
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Report]:
    """-> (X [N,24], obs [N,4] all-ones, y [N], Report). Every synthesised
    sample has all four cues observed (obs=1) -- missing-cue robustness is
    trained separately by WindowDataset's modality dropout on top of this,
    exactly as for the real data (fusion/model/datasets.py)."""
    rng = np.random.default_rng(seed)
    combos = all_combos()
    X, Y, skipped, counts = [], [], [], {}
    for ctx, emo, ges, mot in combos:
        need = {"emo": emo, "ges": ges, "mot": mot, "ctx": ctx}
        missing_pool = [(m, cls) for m, cls in need.items() if (m, cls) not in pools]
        if missing_pool:
            skipped.append((ctx, emo, ges, mot, missing_pool))
            continue
        label = label_combo(ctx, emo, ges, mot)
        x = np.zeros((n_per_combo, 24), np.float32)
        for m, cls in need.items():
            pool = pools[(m, cls)]
            pick = rng.integers(0, len(pool), n_per_combo)
            x[:, COL_OFFSET[m]:COL_OFFSET[m] + len(LABELS[m])] = pool[pick]
        X.append(x)
        Y.append(np.full(n_per_combo, common.INTENTS.index(label), np.int64))
        counts[label] = counts.get(label, 0) + n_per_combo

    X = np.concatenate(X) if X else np.zeros((0, 24), np.float32)
    y = np.concatenate(Y) if Y else np.zeros((0,), np.int64)
    obs = np.ones((len(X), 4), np.float32)
    report = Report(n_combos=len(combos), n_generated=len(X),
                    n_skipped_empty_pool=len(skipped), skipped=skipped,
                    label_counts=counts)
    return X, obs, y, report
