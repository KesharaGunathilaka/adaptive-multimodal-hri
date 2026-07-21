"""Policy demo + F02 safety check: run the deployed fusion model over the
test-subject windows, push every window through the intent->action policy,
and report (a) the action distribution per scenario, (b) F02 recall at the
window and clip level (the safety-critical number, handover §9).

Writes results/fusion_v1/POLICY_DEMO.md.

Run from repo root:  .venv/Scripts/python scripts/11_policy_demo.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.actions import policy  # noqa: E402
from fusion.baselines import common  # noqa: E402
from fusion.model.model import AttentionFusion  # noqa: E402
from fusion.model.train import frame_arrays  # noqa: E402

OUT = ROOT / "results" / "fusion_v1" / "POLICY_DEMO.md"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionFusion(missing_mode="exclude").to(device)
    model.load_state_dict(torch.load(
        ROOT / "jetson_deploy" / "fusion" / "fusion_attn.pt",
        map_location=device, weights_only=True))
    model.eval()

    df = common.load_windows()
    test = df[df.split_subject == "test"].copy()
    X, obs = frame_arrays(test)
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(X).to(device),
                                    torch.from_numpy(obs).to(device)), dim=1).cpu().numpy()

    ctx = test["context_gt"].to_numpy()
    decisions = [policy.decide(p, context_label=c) for p, c in zip(probs, ctx)]
    test["action"] = [d.action for d in decisions]
    test["decided_intent"] = [d.intent for d in decisions]
    test["fallback"] = [d.fallback for d in decisions]
    test["emergency"] = [d.emergency for d in decisions]

    lines = ["# Policy demo — deployed fusion + intent->action layer "
             "(test subjects)", "",
             f"tau={policy.TAU}, tau_emergency={policy.TAU_EMERGENCY}", "",
             "| scenario | intent | top actions (window share) | fallback% |",
             "|---|---|---|---|"]
    for sid, g in test.groupby("scenario_id"):
        acts = Counter(g["action"])
        top = ", ".join(f"{a} {n/len(g):.0%}" for a, n in acts.most_common(3))
        lines.append(f"| {sid} | {g['intent'].iat[0]} | {top} "
                     f"| {g['fallback'].mean():.0%} |")

    # F02 safety numbers
    f02 = test[test.intent == "F02"]
    win_recall = (f02["decided_intent"] == "F02").mean()
    clip_hit = f02.groupby("clip_id")["emergency"].any()
    false_rate = test[test.intent != "F02"]["emergency"].mean()
    lines += ["", "## F02 emergency safety check", "",
              f"- window-level F02 recall: **{win_recall:.3f}** "
              f"({len(f02)} windows)",
              f"- clip-level 'any emergency fired': **{clip_hit.mean():.3f}** "
              f"({clip_hit.sum()}/{len(clip_hit)} clips)",
              f"- false-emergency rate on non-F02 windows: "
              f"**{false_rate:.3f}**"]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
