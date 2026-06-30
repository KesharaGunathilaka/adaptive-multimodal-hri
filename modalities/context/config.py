"""
Top-level configuration for the context modality (fusion + pipeline).

The three sub-models keep their own configs (scene_classification/config.py,
object_detection/config.py, gaze_estimation/config.py). This file holds only the
parameters of the fusion layer that combines them into a ContextState.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Scene changes slowly, so we only re-classify every N frames to save compute.
SCENE_EVERY = 5

# Only run gaze estimation when a person is detected (saves GPU on the Jetson).
GAZE_REQUIRES_PERSON = True

# Max distance (pixels) from the projected gaze point to an object's box for
# that object to count as the user's attention target.
ATTENTION_MAX_DIST = 200.0
