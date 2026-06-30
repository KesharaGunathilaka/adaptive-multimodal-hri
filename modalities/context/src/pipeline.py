"""Context model fusion pipeline.

Fuses the three context signals — scene (environment), objects, and gaze — into
a single ``ContextState`` per frame, the object the downstream policy / fusion
module consumes.
"""
from pathlib import Path
import sys
import time

import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.config import (
    ATTENTION_MAX_DIST,
    GAZE_REQUIRES_PERSON,
    SCENE_EVERY,
)
from modalities.context.scene_classification.src.classifier import SceneClassifier
from modalities.context.object_detection.detector import ContextDetector
from modalities.context.gaze_estimation.gaze_estimator import GazeEstimator
from modalities.context.src.context_state import (
    ContextState,
    DetectedObject,
    GazeInfo,
    infer_activity,
    resolve_attention,
)


class ContextPipeline:
    def __init__(self, scene_every=SCENE_EVERY, gaze_requires_person=GAZE_REQUIRES_PERSON):
        self.scene = SceneClassifier()
        self.detector = ContextDetector()
        self.gaze = GazeEstimator()

        self.scene_every = scene_every
        self.gaze_requires_person = gaze_requires_person

        self._frame_idx = 0
        self._last_scene = {"label": "unknown", "confidence": 0.0}

    def process_frame(self, frame):
        """Run all three sub-models and fuse them into a ContextState."""
        self._frame_idx += 1

        # --- Scene (throttled) ---
        if self.scene_every <= 1 or self._frame_idx % self.scene_every == 1:
            self._last_scene = self.scene.predict(frame)
        scene_label = self._last_scene["label"]
        scene_conf = self._last_scene["confidence"]

        # --- Objects (every frame) ---
        annotated, detections, _counts, _stable = self.detector.process_frame(frame)
        objects = [
            DetectedObject(
                label=d["label"], category=d["category"], confidence=d["confidence"],
                bbox=d["bbox"], track_id=d["track_id"],
            )
            for d in detections
        ]

        # --- Gaze (optionally gated on a person being present) ---
        person_present = any(o.label == "person" for o in objects)
        if person_present or not self.gaze_requires_person:
            gaze = self.gaze.estimate(frame)
        else:
            gaze = GazeInfo(has_face=False)

        # --- Fusion ---
        attention = resolve_attention(gaze, objects, max_dist=ATTENTION_MAX_DIST)
        activity = infer_activity(scene_label, attention, gaze)

        state = ContextState(
            scene=scene_label, scene_confidence=scene_conf, objects=objects, gaze=gaze,
            attention_object=attention, activity=activity, engaged=gaze.looking_at_robot,
            timestamp=time.time(),
        )
        return annotated, state

    def close(self):
        self.gaze.close()


def _draw_overlay(frame, state, fps):
    """Annotate a frame with the fused context (used by the inference scripts)."""
    h, w = frame.shape[:2]

    if state.gaze.has_face and state.gaze.gaze_point and state.gaze.face_bbox:
        fb = state.gaze.face_bbox
        cx, cy = (fb[0] + fb[2]) // 2, (fb[1] + fb[3]) // 2
        gp = (int(state.gaze.gaze_point[0]), int(state.gaze.gaze_point[1]))
        cv2.arrowedLine(frame, (cx, cy), gp, (0, 255, 0), 2, tipLength=0.2)

    if state.attention_object is not None:
        x1, y1, x2, y2 = state.attention_object.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"attending: {state.attention_object.label}",
                    (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.rectangle(frame, (0, 0), (w, 58), (30, 30, 30), -1)
    line1 = f"scene: {state.scene} ({state.scene_confidence:.2f})   activity: {state.activity}"
    engaged = "ENGAGED" if state.engaged else "not engaged"
    line2 = f"{engaged}   objects: {len(state.objects)}   {fps:.1f} FPS"
    cv2.putText(frame, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, line2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame
