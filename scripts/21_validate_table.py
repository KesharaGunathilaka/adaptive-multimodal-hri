"""Validate `docs/final_dataset_merged.docx` against itself. No other document.

Every check here is internal: the scenario table is compared to the legend tables
that sit above it in the same file, and to itself. Nothing is compared against an
earlier version of the dataset.

    .venv/Scripts/python scripts/21_validate_table.py

Exit code is the number of ERROR-level findings, so it can gate a build.
"""
from __future__ import annotations

import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

# Findings contain non-ASCII (e.g. the arrow in "#1->A01"), and the default
# Windows console codec is cp1252 -- without this the script dies with
# UnicodeEncodeError *while printing its own error list*, hiding the result.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs" / "final_dataset_merged.docx"

COLS = ["num", "split", "context", "scenario", "emotion", "gesture", "motion",
        "missing", "intent", "goal", "why", "action", "test"]
CHANNELS = ("context", "emotion", "gesture", "motion")

ERRORS: list[str] = []
WARNS: list[str] = []


def err(tag, msg):
    ERRORS.append(f"[{tag}] {msg}")


def warn(tag, msg):
    WARNS.append(f"[{tag}] {msg}")


# ── read ────────────────────────────────────────────────────────────────────
def read():
    xml = zipfile.ZipFile(DOCX).read("word/document.xml").decode("utf8")
    tx = re.compile(r"<w:t(?: [^>]*)?>(.*?)</w:t>", re.S)
    tables, scen, order = [], {}, []
    for tbl in re.findall(r"<w:tbl[ >](.*?)</w:tbl>", xml, re.S):
        rows = []
        for tr in re.findall(r"<w:tr[ >](.*?)</w:tr>", tbl, re.S):
            cs = re.split(r"</w:tc>", tr)[:-1]
            v = [re.sub(r"\s+", " ", " ".join(tx.findall(c))).strip() for c in cs]
            rows.append(v)
            if len(v) == 13 and v[0].isdigit():
                scen[int(v[0])] = dict(zip(COLS, v))
                order.append(int(v[0]))
        tables.append(rows)
    return tables, scen, order


def legends(tables):
    """The four code tables and the channel-vocabulary table."""
    out = {"intent": {}, "action": {}, "goal": {}, "test": {}, "vocab": {}}
    for rows in tables:
        if not rows or len(rows[0]) < 2:
            continue
        head = [h.lower() for h in rows[0]]
        body = rows[1:]
        if head[:2] == ["code", "name"]:
            out["intent"] = {r[0]: r[1] for r in body if r and r[0]}
            out["intent_def"] = {r[0]: r[2] for r in body if len(r) > 2}
        elif head[:2] == ["code", "action"]:
            out["action"] = {r[0]: r[1] for r in body if r and r[0]}
        elif head[:2] == ["code", "goal"]:
            out["goal"] = {r[0]: r[1] for r in body if r and r[0]}
        elif head[:2] == ["code", "test"]:
            out["test"] = {r[0]: r[1] for r in body if r and r[0]}
        elif head[:1] == ["channel"]:
            for r in body:
                key = re.sub(r"\s*\(.*", "", r[0]).strip().lower()
                vals = [v.strip().lower() for v in r[1].split(",") if v.strip()]
                claimed = re.search(r"\((\d+)", r[0])
                out["vocab"][key] = {"values": vals,
                                     "claims": int(claimed.group(1)) if claimed else None,
                                     "label": r[0]}
    return out


norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
codes = lambda s: [c.strip().upper() for c in re.split(r"[,/]", str(s))
                   if c.strip() and c.strip() != "—"]


def main() -> int:
    tables, scen, order = read()
    L = legends(tables)
    rows = scen
    n = len(rows)
    print(f"docs/{DOCX.name}: {n} scenario rows, "
          f"{len(L['intent'])} intents, {len(L['action'])} actions\n")

    # ── S: structure ────────────────────────────────────────────────────────
    if order != sorted(order):
        bad = [b for a, b in zip(order, order[1:]) if b < a]
        err("S1", f"rows are out of order in the document at {bad}")
    dup = {x for x in order if order.count(x) > 1}
    if dup:
        err("S1", f"duplicate row numbers: {sorted(dup)}")
    gaps = [i for i in range(min(order), max(order) + 1) if i not in order]
    if gaps:
        warn("S1", f"row numbering skips {gaps} — fine if deliberate, but the count "
                   f"({n} rows) no longer matches the highest number ({max(order)})")

    required = ["split", "context", "scenario", "emotion", "gesture", "motion",
                "intent", "action", "goal", "why"]
    for i, r in rows.items():
        empty = [c for c in required if not r[c].strip().strip("—")]
        if empty:
            err("S2", f"#{i}: empty required cell(s): {empty}")

    # ── V: vocabulary vs the legend tables in this same document ────────────
    voc = L["vocab"]
    for ch, key in (("emotion", "emotion"), ("gesture", "gesture"),
                    ("motion", "motion"), ("context", "context")):
        allowed = set(voc.get(key, {}).get("values", []))
        if not allowed:
            continue
        allowed |= {"[missing]"}
        if key == "gesture":
            allowed |= {"idle", "none", "neutral (none)"}
        if key == "context":
            allowed = {v.split("(")[0].strip() for v in allowed}
        seen = defaultdict(list)
        for i, r in rows.items():
            seen[norm(r[ch])].append(i)
        for val, who in sorted(seen.items()):
            if val not in allowed:
                err("V1", f"{ch} value {val!r} (rows {who}) is not in the legend "
                          f"'{voc[key]['label']}' = {sorted(allowed)}")
        unused = allowed - set(seen) - {"[missing]"}
        if unused:
            warn("V2", f"{ch}: legend lists {sorted(unused)} but no row uses "
                       f"{'it' if len(unused) == 1 else 'them'}")
        claims = voc.get(key, {}).get("claims")
        listed = len(voc[key]["values"])
        if claims and claims != listed and key not in ("gesture", "context"):
            err("V3", f"legend '{voc[key]['label']}' claims {claims} values but "
                      f"lists {listed}: {voc[key]['values']}")

    for col, table in (("intent", "intent"), ("action", "action"),
                       ("goal", "goal"), ("test", "test")):
        defined = set(L[table])
        used = set()
        for i, r in rows.items():
            for c in codes(r[col]):
                used.add(c)
                if c not in defined:
                    err("V4", f"#{i}: {col} code {c!r} is not defined in the legend "
                              f"({sorted(defined)})")
        dead = defined - used
        if dead:
            warn("V5", f"{col} legend defines {sorted(dead)} but no row uses "
                       f"{'it' if len(dead) == 1 else 'them'}")

    # ── C: semantics ────────────────────────────────────────────────────────
    def masked(r):
        decl = {x.strip() for x in norm(r["missing"]).replace("—", "").split(",") if x.strip()}
        return {c for c in CHANNELS if norm(r[c if c == "context" else c]) == "[missing]"} | decl

    def observed(r):
        m = masked(r)
        return tuple("?" if c in m else norm(r[c]) for c in CHANNELS)

    for i, r in rows.items():
        cue = {c for c in CHANNELS if norm(r[c]) == "[missing]"}
        decl = {x.strip() for x in norm(r["missing"]).replace("—", "").split(",") if x.strip()}
        if cue - decl:
            err("C1", f"#{i}: cue column marks {sorted(cue - decl)} as [MISSING] but "
                      f"the Missing column says {sorted(decl) or 'nothing'}")
        stray = decl - set(CHANNELS)
        if stray:
            err("C1", f"#{i}: Missing column names unknown channel(s) {sorted(stray)}")

    groups = defaultdict(list)
    for i, r in rows.items():
        groups[observed(r)].append(i)
    for tup, who in sorted(groups.items()):
        intents = {norm(rows[i]["intent"]) for i in who}
        actions = {norm(rows[i]["action"]) for i in who}
        if len(intents) > 1:
            err("C2", f"cue collision {tup}: " +
                      ", ".join(f"#{i}→{rows[i]['intent']}" for i in who) +
                      " — indistinguishable from the observable cues alone")
        elif len(actions) > 1:
            err("C3", f"action collision {tup} (all {rows[who[0]]['intent']}): " +
                      ", ".join(f"#{i}→{rows[i]['action']} ({rows[i]['split']})" for i in who) +
                      " — same cues, same intent, different action")
        elif len(who) > 1:
            warn("C4", f"rows {who} are identical in cues, intent and action "
                       f"{tup} → {rows[who[0]]['intent']}/{rows[who[0]]['action']}"
                       + (" — one is train, one is test, so the test row cannot "
                          "measure generalisation"
                          if len({norm(rows[i]['split']) for i in who}) > 1 else ""))

    for i, r in rows.items():
        sp, tt = norm(r["split"]), codes(r["test"])
        if sp == "test" and not tt:
            err("C5", f"#{i}: Test row with an empty Test column — it will drop out "
                      f"of every T-subset")
        if sp == "train" and tt:
            err("C5", f"#{i}: Train row tagged {tt} — it will be counted into a "
                      f"test subset")

    # every intent should exist on both sides of the split
    by_intent = defaultdict(lambda: defaultdict(int))
    for r in rows.values():
        by_intent[norm(r["intent"])][norm(r["split"])] += 1
    for it, d in sorted(by_intent.items()):
        if not d.get("train"):
            err("C6", f"intent {it.upper()} has no train row")
        if not d.get("test"):
            err("C6", f"intent {it.upper()} has no test row")

    # the intent legend's own wording, checked against the rows assigned to it
    f10 = [i for i, r in rows.items() if norm(r["intent"]) == "f10"]
    off = [i for i in f10 if norm(rows[i]["gesture"]) not in ("idle", "none")]
    if off:
        err("C7", f"F10 is defined as 'without any directed signal (gesture = none)' "
                  f"but rows {off} carry a gesture: "
                  + ", ".join(f"#{i}={rows[i]['gesture']}" for i in off))

    # ── C8: prose citing rows ───────────────────────────────────────────────
    for i, r in rows.items():
        cited = {int(x) for x in re.findall(r"#\s?(\d{1,2})\b", f"{r['scenario']} {r['why']}")}
        dead = sorted(c for c in cited if c not in rows)
        if dead:
            err("C8", f"#{i}: justification cites row(s) {dead}, which do not exist")
        if i in cited:
            warn("C8", f"#{i}: justification cites itself")

    # ── C9: scenario prose vs the cue columns ───────────────────────────────
    MOTION = {"sit": r"\bsit|seated|sits\b", "stand": r"\bstand|stands|standing\b",
              "walk": r"\bwalk|walks|walking|runs|running|heading\b",
              "step back": r"step back|steps back|stepping back|jumps back|"
                           r"runs back|recoils|back from|away from"}
    GESTURE = {"wave": r"\bwave|waves|waving\b", "point": r"\bpoint|points|pointing\b",
               "raise hand": r"raises? (a )?hand|raising (a )?hand",
               "beckoning": r"beckon", "thumbs up": r"thumbs up",
               "thumbs down": r"thumbs down",
               "both hands up": r"both hands|both arms|hands up|arms overhead"}
    for i, r in rows.items():
        text = norm(f"{r['scenario']}")
        for col, pats in (("motion", MOTION), ("gesture", GESTURE)):
            val = norm(r[col])
            if val == "[missing]":
                continue
            declared_hit = re.search(pats[val], text) if val in pats else None
            others = [k for k, p in pats.items() if k != val and re.search(p, text)]
            if others and not declared_hit:
                warn("C9", f"#{i}: {col} column says {val!r} but the scenario text "
                           f"describes {others} — \"{r['scenario'][:90]}...\"")

    # a motion word used in prose that the vocabulary does not contain
    vocab_motion = set(voc.get("motion", {}).get("values", []))
    for i, r in rows.items():
        if re.search(r"\bruns?|running\b", norm(r["scenario"])) and "run" not in vocab_motion:
            warn("C10", f"#{i}: scenario says the person runs, but 'run' is not in the "
                        f"motion vocabulary {sorted(vocab_motion)} — it is silently "
                        f"normalised to {r['motion']!r}")

    # ── report ──────────────────────────────────────────────────────────────
    for title, items in (("ERRORS", ERRORS), ("WARNINGS", WARNS)):
        print(f"{'='*76}\n{title} ({len(items)})\n{'='*76}")
        for x in items:
            print(f"  {x}")
        print()
    print(f"{len(ERRORS)} error(s), {len(WARNS)} warning(s)")
    return len(ERRORS)


if __name__ == "__main__":
    sys.exit(main())
