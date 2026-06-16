"""Context model pipeline.

Fuses the three context signals — scene (environment), objects, and gaze — into
a single `ContextState` per frame, the object the downstream policy / fusion
module consumes.

Run directly for a webcam demo:
    python modalities/context/context_pipeline.py
"""

from pathlib import Path
import sys
import time

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.scene_classification.scene_classifier import SceneClassifier
from modalities.context.object_detection.detector import ContextDetector
from modalities.context.gaze_estimation.gaze_estimator import GazeEstimator
from modalities.context.context_state import (
    ContextState,
    DetectedObject,
    resolve_attention,
    infer_activity,
)

# Scene changes slowly, so we only re-classify every N frames to save compute.
SCENE_EVERY = 5


class ContextPipeline:
    def __init__(self, scene_every=SCENE_EVERY, gaze_requires_person=True):
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
        if self._frame_idx % self.scene_every == 1 or self.scene_every == 1:
            self._last_scene = self.scene.predict(frame)
        scene_label = self._last_scene["label"]
        scene_conf = self._last_scene["confidence"]

        # --- Objects (every frame) ---
        annotated, detections, _counts, _stable = self.detector.process_frame(frame)
        objects = [
            DetectedObject(
                label=d["label"],
                category=d["category"],
                confidence=d["confidence"],
                bbox=d["bbox"],
                track_id=d["track_id"],
            )
            for d in detections
        ]

        # --- Gaze (gated on a person being present, to save GPU on Jetson) ---
        person_present = any(o.label == "person" for o in objects)
        if person_present or not self.gaze_requires_person:
            gaze = self.gaze.estimate(frame)
        else:
            gaze = _no_face()

        # --- Fusion ---
        attention = resolve_attention(gaze, objects)
        activity = infer_activity(scene_label, attention, gaze)

        state = ContextState(
            scene=scene_label,
            scene_confidence=scene_conf,
            objects=objects,
            gaze=gaze,
            attention_object=attention,
            activity=activity,
            engaged=gaze.looking_at_robot,
            timestamp=time.time(),
        )
        return annotated, state

    def close(self):
        self.gaze.close()


def _no_face():
    from modalities.context.context_state import GazeInfo

    return GazeInfo(has_face=False)


def _draw_overlay(frame, state, fps):
    """Annotate a frame with the fused context for the demo."""
    h, w = frame.shape[:2]

    # Gaze ray + attention highlight.
    if state.gaze.has_face and state.gaze.gaze_point and state.gaze.face_bbox:
        fb = state.gaze.face_bbox
        cx, cy = (fb[0] + fb[2]) // 2, (fb[1] + fb[3]) // 2
        gp = (int(state.gaze.gaze_point[0]), int(state.gaze.gaze_point[1]))
        cv2.arrowedLine(frame, (cx, cy), gp, (0, 255, 0), 2, tipLength=0.2)

    if state.attention_object is not None:
        x1, y1, x2, y2 = state.attention_object.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            frame,
            f"attending: {state.attention_object.label}",
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # Top info panel.
    cv2.rectangle(frame, (0, 0), (w, 58), (30, 30, 30), -1)
    line1 = f"scene: {state.scene} ({state.scene_confidence:.2f})   activity: {state.activity}"
    engaged = "ENGAGED" if state.engaged else "not engaged"
    line2 = f"{engaged}   objects: {len(state.objects)}   {fps:.1f} FPS"
    cv2.putText(frame, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, line2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def main():
    print("Initializing context pipeline (scene + objects + gaze)...")
    pipeline = ContextPipeline()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        sys.exit(1)

    print("Pipeline ready. Press 'q' or 'ESC' to quit.")
    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, state = pipeline.process_frame(frame)

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        _draw_overlay(annotated, state, fps)
        cv2.imshow("Context Model", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    pipeline.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
