"""
Configuration for MediaPipe-based gaze estimation.

Gaze = head pose (solvePnP on facial landmarks) refined by iris position.
Chosen over heavier appearance models (e.g. L2CS-Net) so it stays CPU-light and
leaves the Jetson Orin Nano GPU for YOLO + the scene CNN.

Several constants below (signs / gains / thresholds) are approximate and may
need a quick on-device calibration — they are isolated here for that reason.
"""

# ─────────────────────────────────────────────
# Face localisation (two-stage)
# ─────────────────────────────────────────────
# Face Mesh's built-in detector is short-range and misses small / distant /
# angled faces (the common "no face detected" case on captured video). We first
# run a full-range face detector to locate the face, then crop + upscale around
# it before running Face Mesh. This dramatically improves recall.
USE_FACE_DETECTOR = True
FACE_DETECT_MODEL = 1  # 0 = short range (~2m), 1 = full range (~5m)
FACE_DETECT_CONFIDENCE = 0.4
# Pad the detected face box by this fraction on each side before cropping, so
# Face Mesh sees the whole head (forehead/chin) it needs for landmarks.
FACE_CROP_MARGIN = 0.35
# Upscale the crop so the smaller side is at least this many pixels — small,
# distant faces become large enough for reliable landmarks.
MIN_CROP_SIZE = 256

# ─────────────────────────────────────────────
# MediaPipe Face Mesh
# ─────────────────────────────────────────────
MAX_NUM_FACES = 1
# Lower thresholds = more tolerant detection/tracking (fewer dropped frames).
MIN_DETECTION_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3

# ─────────────────────────────────────────────
# Engagement (is the user looking at the robot?)
# ─────────────────────────────────────────────
# Combined gaze angle below this (degrees) counts as "looking at the robot".
ENGAGE_ANGLE_DEG = 18.0
# Angle at which engagement score decays to 0.
ENGAGE_MAX_DEG = 50.0

# ─────────────────────────────────────────────
# Iris refinement & gaze-point projection
# ─────────────────────────────────────────────
# How strongly horizontal iris offset adds to head yaw (degrees per unit offset,
# where offset is the iris position in the eye normalised to [-0.5, 0.5]).
IRIS_YAW_GAIN = 40.0
# Scale of the projected gaze ray, in multiples of the face width. Larger =
# gaze point lands further from the face for the same angle.
GAZE_RAY_FACE_WIDTHS = 3.0
# Flip if left/right gaze comes out mirrored on your camera.
YAW_SIGN = 1.0
PITCH_SIGN = 1.0
