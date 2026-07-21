"""Intent -> robot action policy — SELF-CONTAINED COPY for the Jetson package.

Source of truth: ../../fusion/actions/policy.py in the main repo; re-copy on change.

Original docstring follows.

Intent -> robot action policy (handover §9 stretch goal).

Mapping follows the Final_Dataset V3 legend (§2.2) and its multi-action rules:
  * F02: the intent classifier only says "emergency"; the policy layer picks
    the hazard handling from scene context — classroom routes to the teacher
    (A14), kitchen dispatches fire/medical (A02/A03 — hazard type is a scene-
    content question the fusion input can't answer, so A02 is emitted with
    A03 listed as alternate).
  * F09: A09 (wave back) when the robot was engaged with the person, else
    A10 — exposed as the `engaged` flag, default A10 (hold position).

Confidence gating (handover §9):
  * max softmax < TAU        -> fall back to F05/A06 (safe: hold, stay aware).
  * F02 bypass: if P(F02) >= TAU_EMERGENCY (deliberately low), act on F02
    immediately — emergency misses are a safety issue, false alarms are not.
"""
from dataclasses import dataclass

INTENTS = [f"F{i:02d}" for i in range(1, 11)]

TAU = 0.5
TAU_EMERGENCY = 0.30

ACTIONS = {
    "A01": "Positive acknowledgment; prompt/prepare next task",
    "A02": "Halt all motion + fire/smoke alert",
    "A03": "Halt all motion + medical alert",
    "A04": "Approach/follow to indicated spot; await instruction",
    "A05": "Offer task guidance / answer, supportive tone",
    "A06": "Hold position; do not interrupt; stay aware",
    "A07": "Acknowledge; suggest break/alternative activity",
    "A08": "Calm assistance, soft tone (de-escalation)",
    "A09": "Wave back; do not follow",
    "A10": "Acknowledge farewell; hold position",
    "A11": "Move aside promptly; clear path",
    "A12": "Encourage; suggest alternative approach",
    "A13": "Follow; offer to fetch/carry",
    "A14": "Halt risky action; check surroundings; notify supervisor",
    "A15": "Ask clarifying question",
}

_BASE = {"F01": "A01", "F03": "A04", "F04": "A05", "F05": "A06",
         "F06": "A11", "F07": "A08", "F08": "A07", "F10": "A12"}


@dataclass
class Decision:
    intent: str
    action: str
    confidence: float
    fallback: bool          # True when the TAU gate forced F05/A06
    emergency: bool         # True when the F02 bypass fired
    alternates: tuple = ()

    def describe(self):
        return (f"{self.intent} -> {self.action} ({ACTIONS[self.action]}) "
                f"p={self.confidence:.2f}"
                + (" [FALLBACK]" if self.fallback else "")
                + (" [EMERGENCY]" if self.emergency else ""))


def decide(intent_probs, context_label="unknown", engaged=False):
    """intent_probs: length-10 softmax in F01..F10 order.
    context_label: scene classifier argmax label (classroom/kitchen/...).
    engaged: was the robot interacting with this person (F09 A09-vs-A10 rule).
    """
    p_f02 = float(intent_probs[INTENTS.index("F02")])
    if p_f02 >= TAU_EMERGENCY:
        if context_label == "classroom":
            return Decision("F02", "A14", p_f02, False, True)
        return Decision("F02", "A02", p_f02, False, True, alternates=("A03",))

    top = int(max(range(10), key=lambda i: intent_probs[i]))
    conf = float(intent_probs[top])
    intent = INTENTS[top]
    if conf < TAU:
        return Decision("F05", "A06", conf, True, False)

    if intent == "F02":                       # top-1 F02 below bypass never
        if context_label == "classroom":      # happens (TAU > TAU_EMERGENCY)
            return Decision("F02", "A14", conf, False, True)
        return Decision("F02", "A02", conf, False, True, alternates=("A03",))
    if intent == "F09":
        return Decision("F09", "A09" if engaged else "A10", conf, False, False)
    return Decision(intent, _BASE[intent], conf, False, False)
