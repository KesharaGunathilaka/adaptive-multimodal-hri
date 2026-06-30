"""The context model's output contract.

`ContextState` is the single structured object the context model emits per frame
for the downstream policy / fusion module. It bundles the three context signals
(scene, objects, gaze) plus the fused interpretation (which object the user is
attending to, and the inferred activity / engagement).

This module is dependency-light on purpose (dataclasses + math only) so the
fusion rules are easy to read, test, and reuse.
"""

from dataclasses import dataclass, field, asdict
import math
import time


# ─────────────────────────────────────────────
# Data carriers
# ─────────────────────────────────────────────
@dataclass
class DetectedObject:
    label: str
    category: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    track_id: int | None = None


@dataclass
class GazeInfo:
    has_face: bool = False
    yaw: float = 0.0  # degrees, + = user looking toward image-right
    pitch: float = 0.0  # degrees, + = user looking up
    gaze_point: tuple | None = None  # (x, y) where the gaze ray lands on the image
    looking_at_robot: bool = False  # gaze roughly toward the camera (= robot's eye)
    engagement: float = 0.0  # 0..1, how directly the user faces the robot
    face_bbox: tuple | None = None


@dataclass
class ContextState:
    scene: str = "unknown"
    scene_confidence: float = 0.0
    objects: list = field(default_factory=list)  # list[DetectedObject]
    gaze: GazeInfo = field(default_factory=GazeInfo)
    attention_object: DetectedObject | None = None  # object the user is looking at
    activity: str = "unknown"
    engaged: bool = False  # is the user engaging the robot?
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


# ─────────────────────────────────────────────
# Fusion rules
# ─────────────────────────────────────────────
def _bbox_contains(bbox, x, y):
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist_point_to_bbox(bbox, x, y):
    """Euclidean distance from a point to a bbox (0 if inside)."""
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return math.hypot(dx, dy)


def resolve_attention(gaze, objects, max_dist=200.0):
    """Decide which detected object the user is looking at.

    Returns the DetectedObject, or None if the user is looking at the robot,
    no face is present, or no object is near the gaze point.
    """
    if not gaze.has_face or gaze.gaze_point is None or not objects:
        return None
    if gaze.looking_at_robot:
        return None  # attending to the robot, not a scene object

    gx, gy = gaze.gaze_point

    # Prefer an object whose box actually contains the gaze point.
    contained = [o for o in objects if _bbox_contains(o.bbox, gx, gy)]
    if contained:
        return min(
            contained,
            key=lambda o: math.dist(_bbox_center(o.bbox), (gx, gy)),
        )

    # Otherwise the nearest object, if the gaze point is close enough to it.
    nearest = min(objects, key=lambda o: _dist_point_to_bbox(o.bbox, gx, gy))
    if _dist_point_to_bbox(nearest.bbox, gx, gy) <= max_dist:
        return nearest
    return None


# category -> activity for the simple (scene-independent) cases
_ACTIVITY_BY_CATEGORY = {
    "phone": "using_phone",
    "book": "reading",
    "appliance": "cooking",
    "container": "eating_or_drinking",
    "person": "interacting_with_person",
    "furniture": "resting",
}


def infer_activity(scene, attention_object, gaze):
    """Map the fused context to a coarse activity label for the policy."""
    if not gaze.has_face:
        return "no_user"
    if gaze.looking_at_robot:
        return "engaging_robot"
    if attention_object is None:
        return "looking_away"

    category = attention_object.category
    if category == "computer":
        # Same object, different meaning depending on environment.
        return {"office": "working", "classroom": "studying"}.get(
            scene, "using_computer"
        )
    return _ACTIVITY_BY_CATEGORY.get(category, f"attending_{category}")
