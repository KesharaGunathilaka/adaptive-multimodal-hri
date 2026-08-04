"""Gap decomposition on the COMPLETE `data/final_merged` dataset (both contexts,
all 62 V3 rows, all views) — extends the classroom-only study in
`results/realworld_eval_final/GAP_DECOMPOSITION.md` to the finished dataset.

Shared building blocks (REAL/ORACLE table construction, the rule baseline with
its missing-cue and F09->F01 handling, the ceiling check) live in
`scripts/realworld_eval/merged_gap.py` so `scripts/30_merged_recombination.py`
can reuse them without re-deriving; this script is now just the runner + report.

Design, restated:

  ceiling            = best possible given label ambiguity (table structure alone)
  rules   + ORACLE   = does the hand-written rubric hit that ceiling? (sanity check)
  fusion  + ORACLE   = a fusion model trained AND evaluated on perfect cues --
                       isolates whether the ARCHITECTURE can learn the rubric
  fusion  + REAL     = a fusion model trained AND evaluated on real (noisy)
                       model predictions -- what the deployed system actually gets
  rules   + REAL     = the rule baseline on the same real cues

  ceiling - (fusion+oracle)       = fusion generalisation cost
  (fusion+oracle) - (fusion+real) = perception cost

Protocol: clip-level, 4s mean-pooled aggregation, actor-disjoint val
(`split` column), NO training-time augmentation (plain fusion -- the oracle
condition means "perfect inputs", not "perfect inputs + noise injected"),
3 seeds, headline test = `split=='test' & headline_eval` (excludes row #58's
24 train-derived clips).

    .venv/Scripts/python scripts/29_merged_gap_decomposition.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.model import train as T  # noqa: E402
from fusion.tracking import start_run  # noqa: E402
from scripts.realworld_eval import merged_gap as G  # noqa: E402

SEEDS = (0, 1, 2)


def fusion_runs(tbl, tag: str, device) -> tuple[list[dict], list[dict]]:
    splits = {s: tbl[tbl.split == s] for s in ("train", "val", "test")}
    hl = tbl[(tbl.split == "test") & tbl.headline_eval]
    per_seed_full, per_seed_headline = [], []
    for seed in SEEDS:
        t0 = time.time()
        model, val_acc = T.train_fusion(splits, seed=seed, dropout_p=0.0,
                                        jitter_sigma=0.0, device=device,
                                        missing_mode="exclude")
        pred_te = T._eval_arrays(model, *T.frame_arrays(splits["test"]), device)
        pred_hl = T._eval_arrays(model, *T.frame_arrays(hl), device)
        r_full = G.eval_clip(splits["test"].y.to_numpy(), pred_te)
        r_hl = G.eval_clip(hl.y.to_numpy(), pred_hl)
        per_seed_full.append(r_full)
        per_seed_headline.append(r_hl)
        print(f"  {tag} seed{seed}: val_acc={val_acc:.4f} "
              f"test={r_full} headline={r_hl} ({time.time()-t0:.0f}s)", flush=True)
    return per_seed_full, per_seed_headline


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.OUT_DIR.mkdir(parents=True, exist_ok=True)

    clips, windows = G.load_clips_and_windows()
    print(f"{clips.clip_id.nunique()} clips, {clips.v3_row.nunique()} V3 rows, "
          f"{len(windows)} windows", flush=True)

    ceiling = G.check_ceiling(clips)
    print(f"\nceiling check: {ceiling['n_colliding_tuples']} colliding cue "
          f"tuple(s) -> clip-weighted ceiling = {ceiling['clip_weighted_ceiling']} "
          f"(n={ceiling['n_clips']})", flush=True)

    real = G.to_common_schema(G.build_real(clips, windows))
    oracle = G.to_common_schema(G.build_oracle(clips))

    results = {"ceiling": ceiling}

    # ── rules ────────────────────────────────────────────────────────────────
    for tag, tbl, cues in [("rules_oracle", oracle, "oracle"), ("rules_real", real, "real")]:
        pred = G.rule_predict(tbl)
        is_test = (tbl.split == "test").to_numpy()
        is_headline = is_test & tbl.headline_eval.to_numpy()
        r_full = G.eval_clip(tbl.y.to_numpy()[is_test], pred[is_test])
        r_hl = G.eval_clip(tbl.y.to_numpy()[is_headline], pred[is_headline])
        results[tag] = {"full_test": r_full, "headline_test": r_hl}
        print(f"{tag}: full={r_full}  headline={r_hl}", flush=True)
        with start_run("03_diagnostics", f"final_merged__{tag}", dataset="final_merged",
                       split_kind="scenarios", cues=cues,
                       params={"model": "rule_based", "aggregation": "clip_mean"},
                       notes="gap decomposition, complete dataset (both contexts)") as run:
            run.log_metrics({"test_clip_acc": r_full["acc"],
                             "test_clip_macro_f1": r_full["macro_f1"],
                             "headline_clip_acc": r_hl["acc"],
                             "headline_clip_macro_f1": r_hl["macro_f1"],
                             "table_ceiling": ceiling["clip_weighted_ceiling"]})

    # ── fusion ───────────────────────────────────────────────────────────────
    for tag, tbl, cues in [("fusion_oracle", oracle, "oracle"), ("fusion_real", real, "real")]:
        full_runs, hl_runs = fusion_runs(tbl, tag, device)
        results[tag] = {"full_test": G.agg(full_runs), "headline_test": G.agg(hl_runs)}
        with start_run("03_diagnostics", f"final_merged__{tag}", dataset="final_merged",
                       split_kind="scenarios", cues=cues,
                       params={"model": "attention_fusion", "seeds": len(SEEDS),
                               "aggregation": "clip_mean", "missing_mode": "exclude",
                               "dropout_p": 0.0},
                       notes="gap decomposition, complete dataset (both contexts)") as run:
            f, h = results[tag]["full_test"], results[tag]["headline_test"]
            run.log_metrics({"test_clip_acc": f["acc_mean"], "test_clip_acc_std": f["acc_std"],
                             "test_clip_macro_f1": f["macro_f1_mean"],
                             "headline_clip_acc": h["acc_mean"],
                             "headline_clip_acc_std": h["acc_std"],
                             "headline_clip_macro_f1": h["macro_f1_mean"],
                             "table_ceiling": ceiling["clip_weighted_ceiling"]})

    # ── per-row diagnosis (oracle vs real, fusion, seed 0) ──────────────────
    hl_o = oracle[(oracle.split == "test") & oracle.headline_eval]
    hl_r = real[(real.split == "test") & real.headline_eval]
    model_o, _ = T.train_fusion({s: oracle[oracle.split == s] for s in ("train", "val", "test")},
                                seed=0, dropout_p=0.0, missing_mode="exclude", device=device)
    model_r, _ = T.train_fusion({s: real[real.split == s] for s in ("train", "val", "test")},
                                seed=0, dropout_p=0.0, missing_mode="exclude", device=device)
    pred_o = T._eval_arrays(model_o, *T.frame_arrays(hl_o), device)
    pred_r = T._eval_arrays(model_r, *T.frame_arrays(hl_r), device)
    import pandas as pd
    per_row = pd.DataFrame({
        "v3_row": hl_o.v3_row.to_numpy(), "context": hl_o.context.to_numpy(),
        "intent": hl_o.intent.to_numpy(),
        "oracle_hit": (pred_o == hl_o.y.to_numpy()),
        "real_hit": (pred_r == hl_r.y.to_numpy()),
    }).groupby(["v3_row", "context", "intent"]).mean(numeric_only=True).reset_index()
    per_row["drop"] = (per_row.oracle_hit - per_row.real_hit).round(2)
    per_row = per_row.sort_values("drop", ascending=False)
    per_row.to_csv(G.OUT_DIR / "gap_per_row_merged.csv", index=False)

    (G.OUT_DIR / "gap_decomposition_merged.json").write_text(json.dumps(results, indent=2))

    write_report(results, per_row, clips)
    print(f"\n-> {G.OUT_DIR / 'GAP_DECOMPOSITION_MERGED.md'}")


def write_report(results: dict, per_row, clips) -> None:
    c = results["ceiling"]
    lines = [
        "# Gap decomposition — complete `data/final_merged` (both contexts)",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · "
        f"{clips.clip_id.nunique()} clips · {clips.v3_row.nunique()} V3 rows · "
        "protocol matches `results/realworld_eval_final/GAP_DECOMPOSITION.md` "
        "(clip-level, 4s mean-pooled, actor-disjoint val, no train-time "
        "augmentation, 3 seeds).",
        "",
        f"**Ceiling**: {c['n_colliding_tuples']} colliding cue tuple(s) at the "
        f"intent level (F09's removal deleted the classroom direction "
        f"collision) -> clip-weighted ceiling = **{c['clip_weighted_ceiling']}** "
        f"(n={c['n_clips']}).",
        "",
        "## Headline test (excl. row #58's 24 train-derived clips)",
        "",
        "| Configuration | Clip acc | Clip macro-F1 |",
        "|---|---|---|",
    ]
    for tag, label in [("rules_oracle", "Rules + oracle cues"),
                       ("fusion_oracle", "Fusion + oracle cues"),
                       ("rules_real", "Rules + real cues"),
                       ("fusion_real", "Fusion + real cues")]:
        h = results[tag]["headline_test"]
        acc = h.get("acc", h.get("acc_mean"))
        f1 = h.get("macro_f1", h.get("macro_f1_mean"))
        std = f" ± {h['acc_std']}" if "acc_std" in h else ""
        lines.append(f"| {label} | {acc}{std} | {f1} |")

    ro, fo = results["rules_oracle"]["headline_test"], results["fusion_oracle"]["headline_test"]
    fr = results["fusion_real"]["headline_test"]
    fo_acc = fo.get("acc_mean", fo.get("acc"))
    fr_acc = fr.get("acc_mean", fr.get("acc"))
    gen_cost = round(c["clip_weighted_ceiling"] - fo_acc, 3)
    perc_cost = round(fo_acc - fr_acc, 3)
    lines += [
        "", "## Decomposition",
        "",
        f"- ceiling {c['clip_weighted_ceiling']} - fusion+oracle {fo_acc} = "
        f"**fusion generalisation cost {gen_cost}**",
        f"- fusion+oracle {fo_acc} - fusion+real {fr_acc} = "
        f"**perception cost {perc_cost}**",
        "",
        (f"- rules+oracle scored {ro.get('acc')} against the {c['clip_weighted_ceiling']} "
         "ceiling — the rubric is fully expressible and the rule implementation "
         "is not a strawman (after remapping the legacy F09 branch to F01; see "
         "`merged_gap.rule_predict`'s docstring for why that remap belongs "
         "there and not in the shared `fusion/baselines/rule_based.py`, which "
         "`data/old` still needs)."
         if abs(ro.get("acc", 0) - c["clip_weighted_ceiling"]) < 1e-6 else
         f"- rules+oracle scored {ro.get('acc')} against the {c['clip_weighted_ceiling']} "
         "ceiling — investigate this shortfall before trusting the decomposition "
         "below; it means the rule implementation, not the table, is the "
         "source of the gap."),
        "",
        "## Worst 15 rows by oracle-minus-real drop (perception-attributable failures)",
        "",
        per_row.head(15).to_markdown(index=False),
        "",
        "## Worst 15 rows where oracle ALSO fails (fusion-generalisation-attributable)",
        "",
        per_row[per_row.oracle_hit < 0.5].sort_values("oracle_hit").head(15)
            .to_markdown(index=False),
    ]
    (G.OUT_DIR / "GAP_DECOMPOSITION_MERGED.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
