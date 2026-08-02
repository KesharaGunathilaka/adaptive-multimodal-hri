"""Build the annotation tables for `data/final_merged`, which shipped without any.

Writes, into `data/final_merged/annotations/`:

    scenarios.csv      the 62 live V3 rows from final_dataset_merged.docx
    retired_rows.csv   rows the document blanked (#30), kept so the number is explained
    clips.csv          every video: probe + SHA-256 + provenance + label join + usability
    derived_rows.csv   rows materialised by masking another row's clips (#6, #9)
    takes.csv          synchronised multi-view takes, with a blank person_id to fill
    INTEGRITY.md       everything the data disagrees with itself about

Three things this does that `14_final_annotations.py` could not:

* **Provenance survives the rename.** Every clip was renamed to `S{row}_F{intent}_c{NNN}`,
  destroying the old filename that carried the recording session and the `data/old`
  clip_id. Both are recovered by SHA-256 against `data/final` and `data/old`, so
  `source`, `recorded_at` and `person_id` are still available — `source` matters
  because the emotion/motion models were fine-tuned on the `curated_clip` half and
  pooling the two overstates both (WORKLOG 2026-07-27).

* **Duplicate content is resolved, not counted twice.** 350 files share content with
  another file. Two kinds, handled differently: **derived rows** (#6 = #5 with context
  masked, #9 = #8 with gesture masked, #58 = #49 with emotion+gesture masked) keep both
  copies and record the link, so features are extracted once and masking is applied at
  load; plain re-saves are excluded with a `dup_of` pointer to the canonical copy.
  `derived_rows.csv` flags #58's two hazards — its source is in the *other* split, and
  the mask *changes* the intent rather than preserving it.

* **The folder<->row check is real again.** `14_final_annotations.py` verified numbering
  by reading the old scenario ID out of the filename; after the rename that check
  compares each folder against itself and passes vacuously. Here the check is that
  every live V3 row has exactly one folder, that folder's F-tag equals the row's
  intent, and its context matches — plus a scan for rows whose observable cue tuple
  collides with another row's.

    .venv/Scripts/python scripts/20_merged_annotations.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.realworld_eval.merged_common import (  # noqa: E402
    ANNOT_DIR, CHANNELS, CLIPS_CSV, CLIPS_ROOT, DERIVED_CSV, FINAL, INTEGRITY_MD,
    OLD_CLIPS_CSV, RETIRED_CSV, SCENARIOS_CSV, TAKES_CSV, _OLD_CURATED_RE,
    _SCENARIO_RE, classify_view, parse_v3_table, probe, recorded_at, sha256,
    video_files)

CACHE = ANNOT_DIR / ".probe_cache.json"


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """to_csv, but say which file Excel is holding open instead of a traceback."""
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        raise SystemExit(
            f"cannot write {path.name} — it is open in another program "
            f"(Excel locks the file). Close it and re-run.\n  {path}")

# ── derived rows declared in code ───────────────────────────────────────────
# #6 and #9 announce themselves with a .txt note on disk; #58 does not, so it is
# declared here. User decision 2026-07-31, taken with these three objections on
# the record:
#   1. Masking flips the label on identical pixels — with the gesture visible the
#      footage is #49 (F01, farewell), with it masked it is #58 (F06, give way).
#      #6/#9 preserve their source row's intent; this one does not.
#   2. #49 is a TRAIN row and #58 is a TEST row, so 24 clips' frames appear on
#      both sides of the split. Any #58 score is contaminated to that extent.
#   3. #58's own rationale makes direction the deciding cue ("walk toward robot");
#      #49's footage walks toward the exit, which masking does not change.
# Evidence that the 24 clips were shot as #49: the 2026-07-28 kitchen session ran
# in scenario order (S47 15:37-15:41, these 15:43-15:47, S50 15:48-15:54), while
# #58's own windows are 2026-07-27 13:30-16:19.
DERIVED_ROWS = {
    58: {
        "source_v3_row": 49,
        "mask_modalities": "emotion, gesture",
        "note_text": "user decision 2026-07-31: the 24 clips shared with S49_F01 are "
                     "#49's footage re-used for #58 with emotion and gesture masked. "
                     "CAVEAT: #49 is train and #58 is test, so these frames are on "
                     "both sides of the split; and the mask changes the intent "
                     "(F01 -> F06) rather than preserving it.",
    },
}

# Same 11 files claimed by two folders recorded back to back. Both rows carry the
# identical cue tuple (kitchen/neutral/idle/stand) and the same intent F05, so the
# assignment is label-neutral; S40 gets them because it is otherwise the smaller set.
SHARED_OWNER = {
    ("kitchen/S40_F05", "kitchen/S41_F05"): "kitchen/S40_F05",
    ("kitchen/S38_F04", "kitchen/S63_F04"): "kitchen/S38_F04",
}


# ── inventory ───────────────────────────────────────────────────────────────
def build_raw_inventory() -> pd.DataFrame:
    """Probe + hash every merged clip, caching so re-runs are instant."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    files = video_files()
    print(f"  {len(files)} video files ({len(cache)} cached)")

    rows, unreadable, fresh = [], [], 0
    for i, p in enumerate(files, 1):
        rel = p.relative_to(CLIPS_ROOT).as_posix()
        rec = cache.get(rel)
        if rec is None or rec.get("bytes") != p.stat().st_size:
            info = probe(p)
            if info is None:
                unreadable.append(rel)
                continue
            rec = {**info, "bytes": p.stat().st_size, "sha256": sha256(p)}
            cache[rel] = rec
            fresh += 1
            if fresh % 200 == 0:
                print(f"    hashed {fresh} new ... ({i}/{len(files)})", flush=True)
        rows.append({"filepath": rel, "clip_id": p.stem, "ext": p.suffix.lower(), **rec})

    CACHE.write_text(json.dumps(cache))
    if unreadable:
        print(f"  WARNING: {len(unreadable)} unreadable: {unreadable[:5]}")

    df = pd.DataFrame(rows)
    df["context"] = df.filepath.str.split("/").str[0]
    df["scenario_dir"] = df.filepath.str.split("/").str[1]
    df["v3_row"] = pd.to_numeric(
        df.scenario_dir.str.extract(r"^S(\d+)_F\d+$", flags=re.I)[0],
        errors="coerce").astype("Int64")
    df["view"] = [classify_view(r.width, r.height, r.fps) for r in df.itertuples()]
    return df


def attach_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Recover, by SHA-256, what the rename threw away."""
    old = pd.read_csv(OLD_CLIPS_CSV)
    by_hash = old.drop_duplicates("sha256").set_index("sha256")
    print(f"  data/old: {len(by_hash)} hashed clips with person_id")

    # data/final still holds the original filenames; hash it once
    fin_cache = ANNOT_DIR / ".final_hashes.json"
    if fin_cache.exists():
        fmap = json.loads(fin_cache.read_text())
    else:
        fmap, files = {}, video_files(FINAL / "raw" / "clips")
        print(f"  hashing {len(files)} files under data/final for provenance ...")
        for i, p in enumerate(files, 1):
            fmap.setdefault(sha256(p), []).append(
                p.relative_to(FINAL / "raw" / "clips").as_posix())
            if i % 500 == 0:
                print(f"    {i}/{len(files)}", flush=True)
        fin_cache.write_text(json.dumps(fmap))

    old_paths, old_names, person, source, old_scen, rec_at = [], [], [], [], [], []
    for r in df.itertuples():
        cands = fmap.get(r.sha256, [])
        # prefer the copy that came from this clip's own folder
        same = [c for c in cands if c.split("/")[1] == r.scenario_dir] or cands
        op = same[0] if same else None
        name = Path(op).stem if op else None
        old_paths.append(op)
        old_names.append(name)
        rec_at.append(recorded_at(name) if name else None)

        cur = _OLD_CURATED_RE.match(name) if name else None
        old_scen.append(cur.group(1) if cur else None)
        if cur:
            source.append("curated_clip")
        elif name:
            ts = recorded_at(name)
            source.append(f"raw_take_{ts[:10].replace('-', '')}" if ts else "raw_take")
        else:
            source.append("unknown")

        pid = by_hash.person_id.get(r.sha256)
        person.append(pid if isinstance(pid, str) else None)

    df["old_filepath"] = old_paths
    df["old_clip_id"] = old_names
    df["old_scenario_id"] = old_scen
    df["recorded_at"] = rec_at
    df["source"] = source
    df["person_id"] = person
    print(f"  provenance: {df.old_filepath.notna().sum()}/{len(df)} clips traced to "
          f"data/final; person_id known for {df.person_id.notna().sum()}")
    return df


# ── duplicate / usability resolution ────────────────────────────────────────
def resolve_usability(df: pd.DataFrame, v3: pd.DataFrame) -> pd.DataFrame:
    """Decide, per clip, whether it is a usable independent sample and why not."""
    df = df.sort_values("filepath").reset_index(drop=True)
    df["usable"] = True
    df["exclude_reason"] = ""
    df["exclude_kind"] = ""
    df["dup_of"] = None
    df["derived_from_row"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    df["derived_from_clip_id"] = None

    groups = defaultdict(list)
    for r in df.itertuples():
        groups[r.sha256].append(r.Index)

    # 1. derived rows: same footage, deliberately re-labelled with a channel masked.
    # A folder can hold several copies of one source clip, so link every member of
    # the target row in the group, not just one.
    derived_src = {int(r.v3_row): int(r.derived_from_row)
                   for _, r in v3.iterrows() if pd.notna(r.derived_from_row)}
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        by_row = defaultdict(list)
        for i in idxs:
            if pd.notna(df.at[i, "v3_row"]):
                by_row[int(df.at[i, "v3_row"])].append(i)
        for tgt, src in derived_src.items():
            if tgt not in by_row or src not in by_row:
                continue
            for i in by_row[tgt]:
                df.at[i, "derived_from_row"] = src
                df.at[i, "derived_from_clip_id"] = df.at[by_row[src][0], "clip_id"]

    # 2. everything else that shares content: keep one copy, point the rest at it
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        live = [i for i in idxs if df.at[i, "usable"]]
        if len(live) < 2:
            continue
        # a derived-row copy is a legitimate second label, not a redundant file
        if any(pd.notna(df.at[i, "derived_from_row"]) for i in live):
            continue
        folders = {df.at[i, "filepath"].rsplit("/", 1)[0] for i in live}
        keep = live[0]
        if len(folders) > 1:
            owner = next((o for k, o in SHARED_OWNER.items() if set(k) == folders), None)
            if owner:
                keep = next((i for i in live
                             if df.at[i, "filepath"].startswith(owner + "/")), live[0])
        for i in live:
            if i == keep:
                continue
            df.at[i, "usable"] = False
            df.at[i, "dup_of"] = df.at[keep, "clip_id"]
            same_folder = (df.at[i, "filepath"].rsplit("/", 1)[0]
                           == df.at[keep, "filepath"].rsplit("/", 1)[0])
            df.at[i, "exclude_kind"] = ("duplicate within the same folder" if same_folder
                                        else "duplicate claimed by two folders")
            df.at[i, "exclude_reason"] = (
                f"exact duplicate of {df.at[keep, 'filepath']}"
                + ("" if same_folder else
                   " — two folders claim it; both rows carry the same cue tuple and "
                   "intent, so the assignment is label-neutral"))
    return df


# ── label join ──────────────────────────────────────────────────────────────
def join_labels(df: pd.DataFrame, v3: pd.DataFrame) -> pd.DataFrame:
    v3i = v3.set_index("v3_row")
    cols = ["split_design", "context", "intent", "action", "emotion_v3", "gesture_v3",
            "motion_v3", "missing_v3", "gt_emotion", "scenario_desc", "goal", "test_tags",
            *[f"{c}_masked" for c in CHANNELS]]
    for c in cols:
        if c == "context":
            continue
        df[c] = df.v3_row.map(lambda r, c=c: v3i[c].get(r) if pd.notna(r) else None)

    caveats = []
    for r in df.itertuples():
        n = []
        if pd.isna(r.v3_row) or r.v3_row not in v3i.index:
            n.append("no V3 row for this folder")
        elif r.emotion_masked:
            n.append("emotion [MISSING] by design")
        if pd.notna(r.derived_from_row):
            n.append(f"derived from V3 #{int(r.derived_from_row)} by masking")
        if not r.usable:
            n.append(r.exclude_reason)
        caveats.append("; ".join(n))
    df["caveat"] = caveats
    df["scoreable"] = df.usable & df.gt_emotion.notna()

    # three synchronised views per take; camera clocks are offset by ~25 min, so
    # pair by recording order within a folder+view, not by timestamp
    df["take_index"] = (df.sort_values("clip_id")
                          .groupby(["context", "scenario_dir", "view"]).cumcount())
    return df.sort_values(["context", "scenario_dir", "view", "clip_id"]).reset_index(drop=True)


# ── checks ──────────────────────────────────────────────────────────────────
def verify(v3: pd.DataFrame, clips: pd.DataFrame) -> list[str]:
    """Every structural claim this root makes, asserted rather than assumed."""
    issues = []
    folders = {}
    for p in sorted(CLIPS_ROOT.glob("*/S*_F*")):
        m = _SCENARIO_RE.match(p.name)
        if m:
            folders[int(m.group(1))] = (p.parent.name, "F" + m.group(2))

    rows = set(v3.v3_row)
    for r in sorted(rows - set(folders)):
        issues.append(f"V3 row #{r} has no clip folder")
    for f in sorted(set(folders) - rows):
        issues.append(f"folder S{f:02d}_* has no live V3 row")
    for _, r in v3.iterrows():
        if r.v3_row not in folders:
            continue
        ctx, tag = folders[r.v3_row]
        if tag != r.intent or ctx != r.context:
            issues.append(f"#{r.v3_row}: table=({r.context},{r.intent}) "
                          f"folder=({ctx},{tag})")

    coll = defaultdict(list)
    for _, r in v3.iterrows():
        coll[r.observed_tuple].append((r.v3_row, r.intent))
    for tup, v in coll.items():
        if len({i for _, i in v} ) > 1:
            issues.append(f"cue collision {tup}: " +
                          ", ".join(f"#{r}->{i}" for r, i in v))

    for _, r in v3.iterrows():
        cue = {c for c in CHANNELS
               if str(r["context" if c == "context" else f"{c}_v3"]) == "[missing]"}
        declared = {x.strip() for x in str(r.missing_v3).lower().split(",") if x.strip()}
        if cue - declared:
            issues.append(f"#{r.v3_row}: cue column masks {sorted(cue - declared)} "
                          f"but the Missing column does not say so")
        if r.split_design == "test" and not r.test_tags:
            issues.append(f"#{r.v3_row}: test row with no T-tag")
        if r.split_design == "train" and r.test_tags:
            issues.append(f"#{r.v3_row}: train row still tagged {r.test_tags}")

    per = clips[clips.usable].groupby(["v3_row", "view"]).size().unstack(fill_value=0)
    for row in per.index:
        if per.loc[row].get("realsense_480p", 0) == 0:
            issues.append(f"#{int(row)}: no RealSense (deployment-view) clips")

    issues += verify_numbering(clips)
    return issues


def verify_numbering(clips: pd.DataFrame) -> list[str]:
    """Folder number == V3 row, checked against the pre-rename labels.

    The clips migrated from `data/old` still carry their old scenario ID — not in
    the filename any more, but recovered by SHA-256 into `old_scenario_id`. Their
    cue tuple is in `data/old/labels.csv`, so the folder each one now lives in can
    be checked against the V3 row that folder claims to be. A silent mismatch here
    would mis-score every downstream evaluation, so it fails loudly.
    """
    labels_csv = ROOT / "data" / "old" / "labels.csv"
    if not labels_csv.exists():
        return [f"{labels_csv} missing — folder<->row numbering left unverified"]
    old = pd.read_csv(labels_csv).set_index("scenario_id")

    issues, checked = [], 0
    mig = clips[clips.old_scenario_id.notna() & clips.v3_row.notna()]
    for (folder, oid), g in mig.groupby(["scenario_dir", "old_scenario_id"]):
        if oid not in old.index:
            continue
        checked += 1
        got, want = old.loc[oid], g.iloc[0]
        same = (str(got.emotion).lower() == str(want.emotion_v3).lower()
                and str(got.context).lower() == str(want.context).lower())
        if not same:
            issues.append(
                f"{folder} holds {len(g)} clips of old {oid} "
                f"(labels.csv: {got.context}/{got.emotion}) but claims V3 "
                f"#{int(want.v3_row)} ({want.context}/{want.emotion_v3})")
    print(f"  folder number == V3 row verified on {checked} migrated scenario(s)")
    return issues


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)

    print("V3 table:")
    v3, retired = parse_v3_table()
    print(f"  {len(v3)} live rows, {len(retired)} retired ({list(retired.v3_row)})")

    print("Inventory:")
    clips = build_raw_inventory()
    print("Provenance:")
    clips = attach_provenance(clips)

    # derived rows come from the .txt notes the recording team left on disk, plus
    # the ones declared in DERIVED_ROWS above
    derived = []
    for note in sorted(CLIPS_ROOT.rglob("S*.txt")):
        m = _SCENARIO_RE.match(note.parent.name)
        tgt = int(m.group(1))
        text = note.read_text().strip()
        src = re.search(r"\bS(\d{1,2})\b", text)
        if not src:
            raise SystemExit(f"{note} names no source scenario: {text!r}")
        chan = next((c for c in CHANNELS if c in text.lower()), None)
        derived.append({"target_v3_row": tgt, "source_v3_row": int(src.group(1)),
                        "mask_modalities": chan, "declared_by": note.name,
                        "note_text": text})
    for tgt, spec in DERIVED_ROWS.items():
        derived.append({"target_v3_row": tgt, "declared_by": "DERIVED_ROWS (code)",
                        **spec})
    dv = pd.DataFrame(derived).sort_values("target_v3_row").reset_index(drop=True)
    v3["derived_from_row"] = v3.v3_row.map(
        dict(zip(dv.target_v3_row, dv.source_v3_row))).astype("Int64")

    print("Usability:")
    clips = resolve_usability(clips, v3)
    clips = join_labels(clips, v3)
    print(f"  usable {clips.usable.sum()}/{len(clips)}; "
          f"excluded {(~clips.usable).sum()}")

    print("Checks:")
    issues = verify(v3, clips)
    for i in issues:
        print(f"  ! {i}")
    if not issues:
        print("  all structural checks passed")

    # ── write ──
    v3["n_clips"] = v3.v3_row.map(clips[clips.usable].groupby("v3_row").size()).fillna(0).astype(int)
    v3["observed_tuple"] = v3.observed_tuple.map(lambda t: "|".join(t))
    write_csv(v3, SCENARIOS_CSV)
    write_csv(retired, RETIRED_CSV)

    keep = ["clip_id", "context", "scenario_dir", "v3_row", "intent", "action",
            "split_design", "filepath", "view", "width", "height", "fps", "n_frames",
            "duration_s", "bytes", "sha256", "source", "old_clip_id", "old_filepath",
            "old_scenario_id", "recorded_at", "person_id", "take_index",
            "emotion_v3", "gesture_v3", "motion_v3", "missing_v3", "gt_emotion",
            *[f"{c}_masked" for c in CHANNELS],
            "derived_from_row", "derived_from_clip_id", "dup_of",
            "usable", "exclude_kind", "exclude_reason", "scoreable", "caveat",
            "scenario_desc"]
    write_csv(clips[keep], CLIPS_CSV)

    v3i = v3.set_index("v3_row")
    dv["split_design"] = dv.target_v3_row.map(v3i.split_design)
    dv["source_split"] = dv.source_v3_row.map(v3i.split_design)
    dv["intent"] = dv.target_v3_row.map(v3i.intent)
    dv["source_intent"] = dv.source_v3_row.map(v3i.intent)
    # only the clips actually carrying the derived link, not the whole folder
    ok = clips[clips.usable]
    dv["n_derived_clips"] = dv.target_v3_row.map(
        ok[ok.derived_from_row.notna()].groupby("v3_row").size()).fillna(0).astype(int)
    dv["n_own_clips"] = dv.target_v3_row.map(
        ok[ok.derived_from_row.isna()].groupby("v3_row").size()).fillna(0).astype(int)
    # a derived row whose source sits in the other split shares frames across it
    dv["crosses_split"] = dv.split_design.ne(dv.source_split)
    write_csv(dv, DERIVED_CSV)

    takes = (clips[clips.usable]
             .groupby(["context", "scenario_dir", "v3_row", "intent", "split_design",
                       "take_index"])
             .agg(n_views=("view", "nunique"), views=("view", lambda s: "+".join(sorted(s))),
                  example_clip=("clip_id", "first"),
                  recorded_at=("recorded_at", "first"),
                  person_id=("person_id", lambda s: s.dropna().iloc[0] if s.notna().any() else None))
             .reset_index())
    write_csv(takes, TAKES_CSV)

    write_integrity(v3, retired, clips, dv, issues)

    print(f"\n  -> {SCENARIOS_CSV.name}  ({len(v3)} rows)")
    print(f"  -> {RETIRED_CSV.name}     ({len(retired)} rows)")
    print(f"  -> {CLIPS_CSV.name}       ({len(clips)} clips, {clips.usable.sum()} usable)")
    print(f"  -> {DERIVED_CSV.name}     ({len(dv)} rows)")
    print(f"  -> {TAKES_CSV.name}       ({len(takes)} takes, "
          f"{takes.person_id.notna().sum()} with a known actor)")
    print(f"  -> {INTEGRITY_MD.name}")

    print("\nBy split (usable clips):")
    print(clips[clips.usable].groupby(["split_design", "context"]).size().to_string())
    print("\nBy view:")
    print(clips[clips.usable].groupby(["view", "source"]).size().to_string())
    print("\nIntent distribution (usable clips):")
    print(clips[clips.usable].groupby(["intent", "split_design"]).size().unstack(fill_value=0).to_string())


def write_integrity(v3, retired, clips, derived, issues) -> None:
    bad = clips[~clips.usable]
    L = ["# INTEGRITY — data/final_merged", "",
         f"Generated by `scripts/20_merged_annotations.py`. "
         f"{len(clips)} videos, {clips.usable.sum()} usable, {len(v3)} live V3 rows.", ""]

    L += ["## Structural checks", ""]
    L += [f"- ⚠ {i}" for i in issues] or ["- all passed"]
    L += ["", "## Retired rows", ""]
    for _, r in retired.iterrows():
        L.append(f"- **#{r.v3_row}** — {r.override_note}")

    L += ["", "## Overrides applied to the .docx", ""]
    for _, r in v3[v3.override_note.ne("")].iterrows():
        L.append(f"- **#{r.v3_row}** — {r.override_note}")

    L += ["", "## Derived rows (same footage, one or more channels masked)", ""]
    for _, r in derived.iterrows():
        L.append(f"- **#{r.target_v3_row}** = #{r.source_v3_row} with `{r.mask_modalities}` "
                 f"masked — {r.n_derived_clips} derived clips"
                 + (f" (+ {r.n_own_clips} of its own)" if r.n_own_clips else "")
                 + f". Declared by `{r.declared_by}`: \"{r.note_text}\" "
                 f"Extract features once from #{r.source_v3_row}; mask at load.")
        if r.crosses_split:
            L.append(f"  - ⚠ **#{r.source_v3_row} is {r.source_split}, "
                     f"#{r.target_v3_row} is {r.split_design}** — {r.n_derived_clips} clips' "
                     f"frames are on both sides of the split, so any #{r.target_v3_row} "
                     f"score is contaminated to that extent. Report it separately from the "
                     f"row's own {r.n_own_clips} clips.")
        if r.intent != r.source_intent:
            L.append(f"  - ⚠ masking **changes the intent** "
                     f"({r.source_intent} → {r.intent}) on identical footage; the other "
                     f"derived rows preserve it.")

    L += ["", "## Excluded clips", "",
          f"{len(bad)} of {len(clips)} clips are excluded from training and scoring "
          f"(`usable=False`); each carries a `dup_of` pointer to the copy that was kept.", ""]
    for kind, g in bad.groupby("exclude_kind"):
        L.append(f"- **{len(g)} clip(s)** — {kind}")
        L.append(f"  - folders: {', '.join(sorted(g.scenario_dir.unique()))}")
        L.append(f"  - e.g. `{g.iloc[0].filepath}` → kept `{g.iloc[0].dup_of}`")

    L += ["", "## What the test rows actually test", "",
          "A test row whose observable cue tuple already occurs in training is a "
          "memorisation check, not a generalisation one — the fusion head has seen "
          "that exact input. Per `GAP_DECOMPOSITION.md` the generalisation band is "
          "the dominant error source, so this split is worth watching.", ""]
    train_tuples = set(v3[v3.split_design.eq("train")].observed_tuple)
    te = v3[v3.split_design.eq("test")]
    seen = te[te.observed_tuple.isin(train_tuples)]
    L.append(f"- **{len(te) - len(seen)} of {len(te)} test rows** present a cue tuple "
             f"unseen in training.")
    if len(seen):
        L.append(f"- **{len(seen)} test row(s) repeat a training tuple:**")
        for _, r in seen.iterrows():
            twin = v3[v3.split_design.eq("train")
                      & v3.observed_tuple.eq(r.observed_tuple)].v3_row.tolist()
            L.append(f"  - `#{r.v3_row}` ({r.observed_tuple} -> {r.intent}) "
                     f"repeats train row(s) {', '.join(f'#{t}' for t in twin)}")

    L += ["", "## Deployment-view coverage", ""]
    per = clips[clips.usable].pivot_table(index=["context", "scenario_dir"], columns="view",
                                          values="clip_id", aggfunc="count", fill_value=0)
    gap = per[per.get("realsense_480p", 0) == 0]
    if len(gap):
        L.append("Rows with **no RealSense 640×480 clips** — unscoreable in the view every "
                 "headline number is reported on:")
        L += [f"- `{c}/{s}` ({int(r.sum())} phone clips only)" for (c, s), r in gap.iterrows()]
    else:
        L.append("Every row has RealSense coverage.")

    INTEGRITY_MD.write_text("\n".join(L) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
