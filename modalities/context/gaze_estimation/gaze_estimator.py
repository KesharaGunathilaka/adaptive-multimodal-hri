"""MediaPipe-based gaze estimator.

Estimates head pose via solvePnP on facial landmarks, refines the horizontal
direction with iris position, and reports a `GazeInfo`: gaze angles, a projected
gaze point on the image, and whether/how strongly the user is engaging the robot
(i.e. looking toward the camera).
"""

import math
from pathlib import Path
import sys

import cv2
import numpy as np
import mediapipe as mp

# Make repo-root imports work whether run standalone or imported by the pipeline.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.gaze_estimation import config as cfg
from modalities.context.context_state import GazeInfo

# Landmark indices (MediaPipe Face Mesh, 468 + 10 iris points).
_NOSE_TIP = 1
_CHIN = 152
_LEFT_EYE_OUTER = 263   # user's left eye, outer corner
_RIGHT_EYE_OUTER = 33   # user's right eye, outer corner
_LEFT_MOUTH = 291
_RIGHT_MOUTH = 61

# Eye corners for iris normalisation (inner/outer).
_RIGHT_EYE = (33, 133)   # (outer, inner)
_LEFT_EYE = (263, 362)   # (outer, inner)
_RIGHT_IRIS_CENTER = 468
_LEFT_IRIS_CENTER = 473

# Generic 3D face model (mm) matching the six landmarks above.
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -63.6, -12.5),  # Chin
        (43.3, 32.7, -26.0),  # Left eye outer corner
        (-43.3, 32.7, -26.0),  # Right eye outer corner
        (28.9, -28.9, -24.1),  # Left mouth corner
        (-28.9, -28.9, -24.1),  # Right mouth corner
    ],
    dtype=np.float64,
)


class GazeEstimator:
    def __init__(self):
        # Stage 1: full-range face detector used to locate the face (robust to
        # small / distant faces). Stage 2: Face Mesh run on the cropped face.
        self.detector = (
            mp.solutions.face_detection.FaceDetection(
                model_selection=cfg.FACE_DETECT_MODEL,
                min_detection_confidence=cfg.FACE_DETECT_CONFIDENCE,
            )
            if cfg.USE_FACE_DETECTOR
            else None
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            # static mode: re-detect each frame, since we feed independent crops.
            static_image_mode=True,
            max_num_faces=cfg.MAX_NUM_FACES,
            refine_landmarks=True,  # needed for iris landmarks
            min_detection_confidence=cfg.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=cfg.MIN_TRACKING_CONFIDENCE,
        )

    def estimate(self, frame_bgr, roi=None):
        """Return a GazeInfo for the given BGR frame (no face -> has_face=False).

        roi: optional (x1, y1, x2, y2) hint (e.g. a YOLO person box) to search
        for the face within, instead of running the face detector on the frame.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Locate the face box (detector, caller ROI, or whole frame).
        box = roi if roi is not None else self._detect_face_box(rgb, w, h)

        landmarks = None
        ox, oy, bw, bh = 0, 0, w, h  # crop offset/size to map landmarks back

        if box is not None:
            x1, y1, x2, y2 = self._expand_and_clamp(box, w, h)
            crop = rgb[y1:y2, x1:x2]
            if crop.size:
                mesh_input = crop
                ch, cw = crop.shape[:2]
                # Upscale small crops so landmarks are reliable.
                smaller = min(cw, ch)
                if 0 < smaller < cfg.MIN_CROP_SIZE:
                    scale = cfg.MIN_CROP_SIZE / smaller
                    mesh_input = cv2.resize(crop, (int(cw * scale), int(ch * scale)))
                lm = self._run_mesh(mesh_input)
                if lm is not None:
                    landmarks = lm
                    ox, oy, bw, bh = x1, y1, (x2 - x1), (y2 - y1)

        # Fallback: run Face Mesh on the whole frame if the crop path failed.
        if landmarks is None:
            lm = self._run_mesh(rgb)
            if lm is None:
                return GazeInfo(has_face=False)
            landmarks = lm
            ox, oy, bw, bh = 0, 0, w, h

        def pt(idx):
            lm = landmarks[idx]
            return np.array([ox + lm.x * bw, oy + lm.y * bh], dtype=np.float64)

        # --- Head pose via solvePnP ---
        image_points = np.array(
            [
                pt(_NOSE_TIP),
                pt(_CHIN),
                pt(_LEFT_EYE_OUTER),
                pt(_RIGHT_EYE_OUTER),
                pt(_LEFT_MOUTH),
                pt(_RIGHT_MOUTH),
            ],
            dtype=np.float64,
        )

        focal = float(w)
        cam_matrix = np.array(
            [[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64
        )
        dist = np.zeros((4, 1))

        ok, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS, image_points, cam_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return GazeInfo(has_face=True)

        head_pitch, head_yaw = _rotation_to_pitch_yaw(rvec)

        # --- Iris refinement (horizontal) ---
        iris_offset = self._horizontal_iris_offset(pt)
        yaw = cfg.YAW_SIGN * (head_yaw + iris_offset * cfg.IRIS_YAW_GAIN)
        pitch = cfg.PITCH_SIGN * head_pitch

        # --- Engagement: how directly the user faces the robot/camera ---
        gaze_angle = math.hypot(yaw, pitch)
        looking_at_robot = gaze_angle < cfg.ENGAGE_ANGLE_DEG
        engagement = max(0.0, min(1.0, 1.0 - gaze_angle / cfg.ENGAGE_MAX_DEG))

        # --- Project a gaze point onto the image plane ---
        xs = [ox + lm.x * bw for lm in landmarks]
        ys = [oy + lm.y * bh for lm in landmarks]
        face_bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        face_width = max(face_bbox[2] - face_bbox[0], 1)
        nose = image_points[0]
        ray_len = cfg.GAZE_RAY_FACE_WIDTHS * face_width
        gaze_x = nose[0] + ray_len * math.tan(math.radians(yaw))
        gaze_y = nose[1] - ray_len * math.tan(math.radians(pitch))

        return GazeInfo(
            has_face=True,
            yaw=float(yaw),
            pitch=float(pitch),
            gaze_point=(float(gaze_x), float(gaze_y)),
            looking_at_robot=bool(looking_at_robot),
            engagement=float(engagement),
            face_bbox=face_bbox,
        )

    def _horizontal_iris_offset(self, pt):
        """Average horizontal iris offset in [-0.5, 0.5] (0 = centred)."""
        offsets = []
        for (outer_idx, inner_idx), iris_idx in (
            (_RIGHT_EYE, _RIGHT_IRIS_CENTER),
            (_LEFT_EYE, _LEFT_IRIS_CENTER),
        ):
            outer_x = pt(outer_idx)[0]
            inner_x = pt(inner_idx)[0]
            iris_x = pt(iris_idx)[0]
            span = inner_x - outer_x
            if abs(span) < 1e-6:
                continue
            # ratio: 0 at outer corner, 1 at inner corner; 0.5 = centred.
            ratio = (iris_x - outer_x) / span
            offsets.append(ratio - 0.5)
        if not offsets:
            return 0.0
        return float(np.mean(offsets))

    def _run_mesh(self, rgb_img):
        """Run Face Mesh; return the first face's landmark list or None."""
        results = self.face_mesh.process(rgb_img)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    def _detect_face_box(self, rgb, w, h):
        """Locate the largest face with the full-range detector (pixel box)."""
        if self.detector is None:
            return None
        results = self.detector.process(rgb)
        if not results.detections:
            return None

        def area(det):
            b = det.location_data.relative_bounding_box
            return max(b.width, 0) * max(b.height, 0)

        best = max(results.detections, key=area)
        b = best.location_data.relative_bounding_box
        x1 = b.xmin * w
        y1 = b.ymin * h
        return (x1, y1, x1 + b.width * w, y1 + b.height * h)

    @staticmethod
    def _expand_and_clamp(box, w, h):
        """Pad a face box by FACE_CROP_MARGIN and clamp to the frame."""
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        x1 -= bw * cfg.FACE_CROP_MARGIN
        x2 += bw * cfg.FACE_CROP_MARGIN
        y1 -= bh * cfg.FACE_CROP_MARGIN
        y2 += bh * cfg.FACE_CROP_MARGIN
        x1 = int(max(0, min(x1, w - 1)))
        y1 = int(max(0, min(y1, h - 1)))
        x2 = int(max(0, min(x2, w)))
        y2 = int(max(0, min(y2, h)))
        return x1, y1, x2, y2

    def close(self):
        self.face_mesh.close()
        if self.detector is not None:
            self.detector.close()


def _rotation_to_pitch_yaw(rvec):
    """Convert a solvePnP rotation vector to (pitch, yaw) in degrees."""
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))

    # Bring pitch into a small range around 0 for a frontal face.
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return pitch, yaw
