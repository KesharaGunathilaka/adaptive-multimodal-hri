"""Row-by-row proof that the generated CSVs match `final_dataset_merged.docx`.

Deliberately does NOT import the production parser
(`scripts/realworld_eval/merged_common.parse_v3_table`). It re-reads the .docx
from scratch with its own extraction and its own normalisation, so a bug in the
parser shows up as a mismatch instead of being reproduced on both sides.

Comparison is whitespace-insensitive and case-insensitive — 'thumbs down' and
'thumbsdown' are the same value — because Word's run splitting is a rendering
detail, not data. Anything that survives that is a real difference.

    .venv/Scripts/python scripts/22_verify_csvs.py            # summary + failures
    .venv/Scripts/python scripts/22_verify_csvs.py --full     # every row printed

Exit code is the number of mismatching rows.
"""
from __future__ import annotations

import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs" / "final_dataset_merged.docx"
ANN = ROOT / "data" / "final_merged" / "annotations"

DOC_COLS = ["num", "split", "context", "scenario", "emotion", "gesture", "motion",
            "missing", "intent", "goal", "why", "action", "test"]
# .docx column -> scenarios.csv column
MAP = {"split": "split_design", "context": "context", "scenario": "scenario_desc",
       "emotion": "emotion_v3", "gesture": "gesture_v3", "motion": "motion_v3",
       "missing": "missing_v3", "intent": "intent", "goal": "goal", "why": "why",
       "action": "action", "test": "test_tags"}
CHANNELS = ("context", "emotion", "gesture", "motion")
GT = {"happy": "Happy", "sad": "Sad", "angry": "Anger", "anger": "Anger",
      "disgust": "Disgust", "surprise": "Surprise", "fear": "Fear",
      "neutral": "Neutral", "[missing]": ""}


def key(s) -> str:
    """Whitespace- and case-insensitive comparison key; '—' means empty."""
    s = "" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)
    s = s.replace("—", " ").replace("–", " ")
    return re.sub(r"\s+", "", s).lower()


def read_docx() -> dict[int, dict]:
    xml = zipfile.ZipFile(DOCX).read("word/document.xml").decode("utf8")
    tx = re.compile(r"<w:t(?: [^>]*)?>(.*?)</w:t>", re.S)
    out = {}
    for tr in re.findall(r"<w:tr[ >](.*?)</w:tr>", xml, re.S):
        cs = re.split(r"</w:tc>", tr)[:-1]
        if len(cs) != 13:
            continue
        v = [re.sub(r"\s+", " ", " ".join(tx.findall(c))).strip() for c in cs]
        if not v[0].isdigit():
            continue
        out[int(v[0])] = dict(zip(DOC_COLS, v))
    return out


def expected_masks(r: dict) -> dict[str, bool]:
    """Designed missingness, derived from the raw row two ways at once."""
    decl = {x.strip() for x in key(r["missing"]).split(",") if x.strip()}
    return {c: (key(r[c]) == "[missing]") or (c in decl) for c in CHANNELS}


def main() -> int:
    full = "--full" in sys.argv
    doc = read_docx()
    try:
        sc = pd.read_csv(ANN / "scenarios.csv")
        clips = pd.read_csv(ANN / "clips.csv")
    except PermissionError as e:
        raise SystemExit(f"cannot read: {e}\nClose the file in Excel and re-run.")

    print(f".docx           : {len(doc)} scenario rows")
    print(f"scenarios.csv: {len(sc)} rows")
    print(f"clips.csv       : {len(clips)} clips "
          f"({int(clips.usable.sum())} usable)\n")

    bad_rows: dict[int, list[str]] = defaultdict(list)

    # ── A. membership ───────────────────────────────────────────────────────
    only_doc = sorted(set(doc) - set(sc.v3_row))
    only_csv = sorted(set(sc.v3_row) - set(doc))
    for r in only_doc:
        bad_rows[r].append("present in .docx but MISSING from scenarios.csv")
    for r in only_csv:
        bad_rows[r].append("present in scenarios.csv but NOT in the .docx")
    if sc.v3_row.duplicated().any():
        for r in sorted(sc.v3_row[sc.v3_row.duplicated()]):
            bad_rows[r].append("duplicated in scenarios.csv")

    # ── B. field-by-field ───────────────────────────────────────────────────
    sci = sc.set_index("v3_row")
    hdr = f"  {'row':>4}  {'cells':>5}  status"
    lines = []
    for n in sorted(set(doc) & set(sci.index)):
        d, c = doc[n], sci.loc[n]
        for dc, cc in MAP.items():
            if key(d[dc]) != key(c[cc]):
                bad_rows[n].append(
                    f"{cc}: .docx={d[dc]!r}  csv={c[cc]!r}")

        # derived columns, recomputed here from the raw row
        m = expected_masks(d)
        for ch, val in m.items():
            if bool(c[f"{ch}_masked"]) != val:
                bad_rows[n].append(
                    f"{ch}_masked: expected {val}, csv={bool(c[f'{ch}_masked'])}")
        want_gt = GT.get(key(d["emotion"]), "?")
        if key(c.gt_emotion) != key(want_gt):
            bad_rows[n].append(f"gt_emotion: expected {want_gt or '(blank)'!r}, "
                               f"csv={c.gt_emotion!r}")
        want_obs = "|".join("?" if m[ch] else re.sub(r"\s+", " ", d[ch]).strip().lower()
                            for ch in CHANNELS)
        if key(c.observed_tuple) != key(want_obs):
            bad_rows[n].append(f"observed_tuple: expected {want_obs!r}, "
                               f"csv={c.observed_tuple!r}")

        # n_clips must equal the usable clips whose folder is this row
        want_n = int((clips.usable & clips.v3_row.eq(n)).sum())
        if int(c.n_clips) != want_n:
            bad_rows[n].append(f"n_clips: csv says {int(c.n_clips)}, "
                               f"clips.csv has {want_n} usable")

        ok = n not in bad_rows
        if full or not ok:
            lines.append(f"  {n:>4}  {len(MAP)+7:>5}  "
                         + ("OK" if ok else f"{len(bad_rows[n])} MISMATCH"))

    # ── C. clips.csv label fields must echo the scenario row ────────────────
    echo = ["split_design", "context", "intent", "action", "emotion_v3", "gesture_v3",
            "motion_v3", "missing_v3", "gt_emotion", "scenario_desc",
            *[f"{c}_masked" for c in CHANNELS]]
    clip_bad = 0
    for n, g in clips.groupby("v3_row"):
        if n not in sci.index:
            print(f"  ! clips.csv has clips for row #{n}, which is not in the table")
            clip_bad += len(g)
            continue
        want = sci.loc[n]
        for col in echo:
            vals = {key(x) for x in g[col].fillna("")}
            if vals != {key(want[col])}:
                bad_rows[n].append(
                    f"clips.csv {col}: {sorted(vals)} != scenario row {key(want[col])!r}")
                clip_bad += 1

    # ── D. folder <-> row ───────────────────────────────────────────────────
    for n, g in clips.groupby("v3_row"):
        dirs = set(g.scenario_dir)
        if len(dirs) != 1:
            bad_rows[n].append(f"row spread over folders {sorted(dirs)}")
        else:
            folder = dirs.pop()
            mm = re.match(r"^S(\d+)_F(\d+)$", folder)
            if not mm:
                bad_rows[n].append(f"folder name {folder!r} is not S<row>_F<intent>")
            else:
                if int(mm.group(1)) != n:
                    bad_rows[n].append(f"folder {folder} does not carry row number {n}")
                if key("F" + mm.group(2)) != key(sci.loc[n, "intent"]):
                    bad_rows[n].append(
                        f"folder {folder} tag != table intent {sci.loc[n, 'intent']}")

    # ── E. the other three tables ───────────────────────────────────────────
    extra: list[str] = []

    retired = pd.read_csv(ANN / "retired_rows.csv")
    if len(retired):
        for r in retired.v3_row:
            if r in doc:
                extra.append(f"retired_rows.csv lists #{r}, but it is a live row "
                             f"in the .docx")
    elif any(not d["split"].strip() for d in doc.values()):
        extra.append("the .docx still contains a blank row but retired_rows.csv is empty")

    dv = pd.read_csv(ANN / "derived_rows.csv")
    ok_clips = clips[clips.usable]
    for _, r in dv.iterrows():
        t, s = int(r.target_v3_row), int(r.source_v3_row)
        for label, n in (("target", t), ("source", s)):
            if n not in doc:
                extra.append(f"derived_rows.csv {label} #{n} is not a row in the .docx")
        if t in sci.index and key(r.split_design) != key(sci.loc[t, "split_design"]):
            extra.append(f"derived_rows.csv #{t}: split_design {r.split_design!r} "
                         f"!= table {sci.loc[t, 'split_design']!r}")
        if t in sci.index and key(r.intent) != key(sci.loc[t, "intent"]):
            extra.append(f"derived_rows.csv #{t}: intent {r.intent!r} "
                         f"!= table {sci.loc[t, 'intent']!r}")
        got_d = int((ok_clips.v3_row.eq(t) & ok_clips.derived_from_row.eq(s)).sum())
        got_o = int((ok_clips.v3_row.eq(t) & ok_clips.derived_from_row.isna()).sum())
        if int(r.n_derived_clips) != got_d:
            extra.append(f"derived_rows.csv #{t}: n_derived_clips={int(r.n_derived_clips)} "
                         f"but clips.csv has {got_d}")
        if int(r.n_own_clips) != got_o:
            extra.append(f"derived_rows.csv #{t}: n_own_clips={int(r.n_own_clips)} "
                         f"but clips.csv has {got_o}")
        if t in sci.index and s in sci.index:
            want = sci.loc[t, "split_design"] != sci.loc[s, "split_design"]
            if bool(r.crosses_split) != want:
                extra.append(f"derived_rows.csv #{t}: crosses_split={bool(r.crosses_split)} "
                             f"but #{s} is {sci.loc[s, 'split_design']} and "
                             f"#{t} is {sci.loc[t, 'split_design']}")
        if int(r.n_derived_clips) + int(r.n_own_clips) != int(sci.loc[t, "n_clips"]):
            extra.append(f"derived_rows.csv #{t}: derived+own != the row's n_clips")

    tk = pd.read_csv(ANN / "takes.csv")
    for col in ("context", "intent", "split_design"):
        for _, r in tk.iterrows():
            n = int(r.v3_row)
            if n in sci.index and key(r[col]) != key(sci.loc[n, col]):
                extra.append(f"takes.csv #{n}: {col} {r[col]!r} "
                             f"!= table {sci.loc[n, col]!r}")
                break
    n_take_clips = int(ok_clips.groupby(["context", "scenario_dir", "take_index"])
                       .ngroups)
    if len(tk) != n_take_clips:
        extra.append(f"takes.csv has {len(tk)} rows but clips.csv yields "
                     f"{n_take_clips} distinct takes")

    integ = ANN / "INTEGRITY.md"
    if integ.stat().st_mtime < DOCX.stat().st_mtime:
        extra.append("INTEGRITY.md is older than the .docx — re-run "
                     "scripts/20_merged_annotations.py")

    # ── report ──────────────────────────────────────────────────────────────
    if lines:
        print(hdr)
        print("\n".join(lines))
        print()
    if bad_rows:
        print("=" * 78)
        print(f"MISMATCHING ROWS ({len(bad_rows)})")
        print("=" * 78)
        for n in sorted(bad_rows):
            print(f"\n  row #{n}")
            for m in bad_rows[n]:
                print(f"     - {m}")
        print()
    if extra:
        print("=" * 78)
        print(f"derived_rows.csv / takes.csv / retired_rows.csv ({len(extra)})")
        print("=" * 78)
        for x in extra:
            print(f"  - {x}")
        print()

    checked = len(set(doc) & set(sci.index)) * (len(MAP) + 7) + len(clips) * len(echo)
    print(f"{checked:,} field comparisons across {len(doc)} table rows "
          f"and {len(clips):,} clips")
    print(f"tables checked: scenarios, clips, derived_rows, takes, retired_rows, "
          f"INTEGRITY.md")
    if bad_rows or extra:
        print(f"{len(doc) - len(bad_rows)}/{len(doc)} rows match; "
              f"{len(extra)} other inconsistency(ies)")
    else:
        print(f"ALL {len(doc)} rows match exactly — every table agrees with the .docx")
    return len(bad_rows) + len(extra)


if __name__ == "__main__":
    sys.exit(main())
