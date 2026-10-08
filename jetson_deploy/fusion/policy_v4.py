"""Intent -> robot action policy (v4) — SELF-CONTAINED COPY for the Jetson
package.

Source of truth: ../../fusion/actions/policy.py conceptually; this is the v4
re-derivation. Differs from the retired policy.py in one structural way: F09
("farewell") was retired and folded into F01 on 2026-08-03 — direction
(arriving vs leaving) is a robot ACTION question the cue set cannot answer, not
a separate intent — so there is no F09 branch or engaged-flag A09/A10 split here.
F01 always maps to A01; a downstream action-selection layer (out of scope for
the intent classifier) can still use direction/velocity to choose A01 vs a
farewell variant if the platform wants that distinction.

Confidence gating (unchanged from v3):
  * max softmax < TAU        -> fall back to F05/A06 (safe: hold, stay aware).
  * F02 bypass: if P(F02) >= TAU_EMERGENCY (deliberately low), act on F02
    immediately — emergency misses are a safety issue, false alarms are not.
"""
from dataclasses import dataclass

INTENTS = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F10"]

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
    "A11": "Move aside promptly; clear path",
    "A12": "Encourage; suggest alternative approach",
    "A14": "Halt risky action; check surroundings; notify supervisor",
}

_BASE = {"F01": "A01", "F03": "A04", "F04": "A05", "F05": "A06",
         "F06": "A11", "F07": "A08", "F08": "A07", "F10": "A12"}


@dataclass
class Decision:
    intent: str
    action: str
    confidence: float
    fallback: bool          # True when the TAU gate forced F05/A06
    emergency: bool          # True when the F02 bypass fired
    alternates: tuple = ()

    def describe(self):
        return (f"{self.intent} -> {self.action} ({ACTIONS[self.action]}) "
                f"p={self.confidence:.2f}"
                + (" [FALLBACK]" if self.fallback else "")
                + (" [EMERGENCY]" if self.emergency else ""))


def decide(intent_probs, context_label="unknown"):
    """intent_probs: length-9 softmax in INTENTS order.
    context_label: scene classifier argmax label (classroom/kitchen/... — the
    FULL scene set, not restricted to what the fusion head's context token
    knows; F02 hazard routing uses it directly, same as v3)."""
    p_f02 = float(intent_probs[INTENTS.index("F02")])
    if p_f02 >= TAU_EMERGENCY:
        if context_label == "classroom":
            return Decision("F02", "A14", p_f02, False, True)
        return Decision("F02", "A02", p_f02, False, True, alternates=("A03",))

    n = len(INTENTS)
    top = int(max(range(n), key=lambda i: intent_probs[i]))
    conf = float(intent_probs[top])
    intent = INTENTS[top]
    if conf < TAU:
        return Decision("F05", "A06", conf, True, False)

    if intent == "F02":                       # top-1 F02 below bypass never
        if context_label == "classroom":      # happens (TAU > TAU_EMERGENCY)
            return Decision("F02", "A14", conf, False, True)
        return Decision("F02", "A02", conf, False, True, alternates=("A03",))
    return Decision(intent, _BASE[intent], conf, False, False)
