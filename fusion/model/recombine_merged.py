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


def classify_combos() -> tuple[set, set]:
    """-> (aligned, conflicting), each a set of (ctx,emo,ges,mot) tuples from
    `all_combos()`. 2026-08-07 conflict-holdout experiment (discussed and
    confirmed with the user against a full manual review of all 164
    conflicting rows): a combo is CONFLICTING if its rubric-derived intent
    differs from the SAME (ctx,ges,mot)'s intent under emotion='Neutral' (the
    "default" reading with no emotional colouring) -- i.e. the emotion
    measurably overrides what the gesture alone would suggest. ALIGNED
    combos (including every neutral-emotion combo, trivially aligned to
    themselves) are the ones where emotion doesn't change the outcome.

    Fear never appears in any aligned combo for any gesture -- confirmed by
    inspection, and expected: `rule_intent`'s `if emo == 'fear': return F02`
    branch fires before any gesture check, so fear unconditionally overrides
    every gesture's default reading."""
    aligned, conflicting = set(), set()
    for ctx, emo, ges, mot in all_combos():
        baseline = label_combo(ctx, "Neutral", ges, mot)
        actual = label_combo(ctx, emo, ges, mot)
        (aligned if actual == baseline else conflicting).add((ctx, emo, ges, mot))
    return aligned, conflicting


BALANCE_MODES = ("uniform", "sqrt", "intent", "family")


def allocate(combos: list[tuple], balance: str = "uniform",
             target_total: int = 44800) -> dict[tuple, int]:
    """-> {combo: n_samples}. Controls the INTENT prior the synthetic set
    imposes (2026-08-08).

    A constant `n_per_combo` gives every cue COMBO equal weight, which makes
    each intent's share proportional to how many combos happen to map to it.
    On this rubric that is severely skewed -- F01 gets 104 of 448 combos
    (23.2%) while F10 gets 8 (1.8%) -- and, more damagingly, it skews the
    CONDITIONAL prior inside a cue family: `Anger+point` splits 6 combos to
    F07 and only 2 to F06, so a model trained this way answers F07 for angry
    pointing even when motion clearly says walking. That is the diagnosed
    cause of row #27 scoring 0.038 (`SCENARIO_TEST_REPORT.md` T02) despite
    clean cues, and of row #55 (`Happy+point`, same 2/8 minority position).

    Modes -- per-intent sample share `s_c`, then split evenly across that
    intent's combos:
      uniform  s_c proportional to n_combos_c  (constant per combo; the
               original behaviour, kept as default so existing callers are
               unchanged)
      sqrt     s_c proportional to sqrt(n_combos_c)  -- partial correction,
               the usual long-tail compromise; rare intents get boosted
               without a handful of combos being replicated hundreds of times
      intent   s_c equal for every intent  -- full balance; note this drives
               F10's 8 combos to ~622 samples each, heavy replication of a
               narrow slice of the cue space
      family   equalise intents WITHIN each (context, emotion, gesture) family,
               i.e. across the 4 motion values only. Measured 2026-08-08:
               `intent` fixes the GLOBAL marginal but barely moves the
               CONDITIONAL one -- after global balancing `Anger+point` still
               carries ~356 F06 samples against ~996 F07, because the family
               still holds 1 walking combo against 3 non-walking. Row #27
               accordingly stayed broken under `intent` (0.038 -> 0.057) while
               row #55 recovered (0.357 -> 0.571). This mode gives every intent
               present in a family equal weight inside it, so the only way to
               separate F06 from F07 there is to actually read motion -- which
               is what the rubric requires.

    `target_total` fixes the overall synthetic budget so the modes are
    compared at equal cost rather than equal-per-combo.
    """
    if balance not in BALANCE_MODES:
        raise ValueError(f"balance must be one of {BALANCE_MODES}, got {balance!r}")
    labels = {c: label_combo(*c) for c in combos}

    if balance == "family":
        fams: dict[tuple, list] = {}
        for c in combos:
            fams.setdefault((c[0], c[1], c[2]), []).append(c)   # (ctx, emo, ges)
        out: dict[tuple, int] = {}
        budget = target_total / max(1, len(fams))
        for fam_combos in fams.values():
            by_intent: dict[str, list] = {}
            for c in fam_combos:
                by_intent.setdefault(labels[c], []).append(c)
            per_intent_budget = budget / len(by_intent)
            for cs in by_intent.values():
                n = max(1, int(round(per_intent_budget / len(cs))))
                for c in cs:
                    out[c] = n
        return out

    per_intent: dict[str, list] = {}
    for c, lab in labels.items():
        per_intent.setdefault(lab, []).append(c)

    if balance == "uniform":
        share = {lab: float(len(cs)) for lab, cs in per_intent.items()}
    elif balance == "sqrt":
        share = {lab: float(np.sqrt(len(cs))) for lab, cs in per_intent.items()}
    else:                                              # "intent"
        share = {lab: 1.0 for lab in per_intent}

    tot_share = sum(share.values())
    out: dict[tuple, int] = {}
    for lab, cs in per_intent.items():
        n_for_intent = target_total * share[lab] / tot_share
        per_combo = max(1, int(round(n_for_intent / len(cs))))
        for c in cs:
            out[c] = per_combo
    return out


def generate(pools: dict, n_per_combo: int = 150, seed: int = 0,
            combo_filter: set | None = None, balance: str = "uniform"
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Report]:
    """-> (X [N,24], obs [N,4] all-ones, y [N], Report). Every synthesised
    sample has all four cues observed (obs=1) -- missing-cue robustness is
    trained separately by WindowDataset's modality dropout on top of this,
    exactly as for the real data (fusion/model/datasets.py).

    `combo_filter`: optional set of (ctx,emo,ges,mot) tuples -- when given,
    only combos in this set are synthesised (all others silently skipped, not
    counted as `n_skipped_empty_pool`). Used by the 2026-08-07 conflict-
    holdout experiment to restrict training to `classify_combos()`'s aligned
    set only, so the model never sees a labelled example -- synthetic OR
    real -- of any conflicting combo.

    `balance`: how many samples each combo gets -- see `allocate()`. The
    default "uniform" reproduces the original constant-`n_per_combo`
    behaviour exactly."""
    rng = np.random.default_rng(seed)
    combos = all_combos()
    if combo_filter is not None:
        combos = [c for c in combos if c in combo_filter]
    alloc = allocate(combos, balance=balance,
                     target_total=n_per_combo * len(combos))
    X, Y, skipped, counts = [], [], [], {}
    for ctx, emo, ges, mot in combos:
        need = {"emo": emo, "ges": ges, "mot": mot, "ctx": ctx}
        missing_pool = [(m, cls) for m, cls in need.items() if (m, cls) not in pools]
        if missing_pool:
            skipped.append((ctx, emo, ges, mot, missing_pool))
            continue
        label = label_combo(ctx, emo, ges, mot)
        n = alloc[(ctx, emo, ges, mot)]
        x = np.zeros((n, 24), np.float32)
        for m, cls in need.items():
            pool = pools[(m, cls)]
            pick = rng.integers(0, len(pool), n)
            x[:, COL_OFFSET[m]:COL_OFFSET[m] + len(LABELS[m])] = pool[pick]
        X.append(x)
        Y.append(np.full(n, common.INTENTS.index(label), np.int64))
        counts[label] = counts.get(label, 0) + n

    X = np.concatenate(X) if X else np.zeros((0, 24), np.float32)
    y = np.concatenate(Y) if Y else np.zeros((0,), np.int64)
    obs = np.ones((len(X), 4), np.float32)
    report = Report(n_combos=len(combos), n_generated=len(X),
                    n_skipped_empty_pool=len(skipped), skipped=skipped,
                    label_counts=counts)
    return X, obs, y, report
