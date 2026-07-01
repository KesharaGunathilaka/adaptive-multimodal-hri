import sys
import os
from collections import deque
import numpy as np

# Add configuration paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import MOTION_LABELS, MOTION_COLORS

class PoseClassifier:
    """
    Classifies the user's posture based on relative keypoint vertical distances.
    """
    def classify(self, landmarks) -> str:
        if landmarks is None:
            return "Unknown"
        lm = landmarks.landmark

        # Get vertical offsets of key joints
        nose_y     = lm[0].y
        hip_y      = (lm[23].y + lm[24].y) / 2
        knee_y     = (lm[25].y + lm[26].y) / 2
        ankle_y    = (lm[27].y + lm[28].y) / 2
        shoulder_y = (lm[11].y + lm[12].y) / 2

        body_height = abs(ankle_y - nose_y) + 1e-6
        torso_ratio = abs(hip_y - shoulder_y) / body_height
        leg_ratio   = abs(ankle_y - knee_y)  / body_height

        if body_height < 0.25:
            return "Lying"
        if torso_ratio > 0.35 and leg_ratio < 0.15:
            return "Sitting"
        if knee_y < hip_y + 0.05:
            return "Crouching"
        return "Standing"

class MotionEngine:
    """
    Combines rule-based postural checking with a rolling keypoint velocity
    tracker to classify dynamic action states stably.
    """
    def __init__(self):
        self.pose_classifier = PoseClassifier()
        self.keypoints_queue = deque(maxlen=30)
        self.smooth_probs = np.ones(8) / 8
        
        # Hyperparameters matching original implementation
        self.alpha = 0.25         # Smoothing factor (EMA)
        self.run_thresh = 1.30     # Velocity cut-off for running
        self.static_thresh = 0.22  # Velocity cut-off for static pose

    def process(self, landmarks):
        """
        Takes raw pose landmarks, processes the frame, and returns resolved label, confidence, and posture.
        """
        pose_label = self.pose_classifier.classify(landmarks)

        if landmarks:
            pts = np.zeros((33, 3), dtype=np.float32)
            for i, lm in enumerate(landmarks.landmark):
                pts[i] = [lm.x, lm.y, lm.z]
            self.keypoints_queue.append(pts)
        else:
            self.keypoints_queue.clear()

        # Default fallback values
        motion_label = "Standing Still"
        confidence = 0.90
        probs = np.zeros(8)
        probs[1] = 0.90

        if len(self.keypoints_queue) >= 4 and landmarks:
            arr = np.array(self.keypoints_queue)   # (seq_len, 33, 3)
            vel = np.diff(arr, axis=0)             # (seq_len-1, 33, 3)

            # 1. Hips velocity calculation (Speed of movement)
            body_vel = vel[:, [23, 24], :2]        # Left/Right hips XY coordinates
            body_speed = float(np.mean(np.abs(body_vel)) * 100.0)

            # 2. Directed walking trend analysis (last 15 frames)
            look_back = min(15, len(arr) - 1)
            hip_w = np.linalg.norm(arr[-look_back-1:, 23, :2] - arr[-look_back-1:, 24, :2], axis=1)
            dh = float(hip_w[-1] - hip_w[0])
            path_h = float(np.sum(np.abs(np.diff(hip_w))))
            eff_h = abs(dh) / (path_h + 1e-5)

            # Average Hips/Shoulders coordinates translation
            hips_xy = arr[-look_back-1:, [23, 24], :2].mean(axis=1)
            dx_hip = float(hips_xy[-1, 0] - hips_xy[0, 0])
            dy_hip = float(hips_xy[-1, 1] - hips_xy[0, 1])
            eff_x_hip = abs(dx_hip) / (float(np.sum(np.abs(np.diff(hips_xy[:, 0])))) + 1e-5)
            eff_y_hip = abs(dy_hip) / (float(np.sum(np.abs(np.diff(hips_xy[:, 1])))) + 1e-5)

            sh_xy = arr[-look_back-1:, [11, 12], :2].mean(axis=1)
            dx_sh = float(sh_xy[-1, 0] - sh_xy[0, 0])
            dy_sh = float(sh_xy[-1, 1] - sh_xy[0, 1])
            eff_x_sh = abs(dx_sh) / (float(np.sum(np.abs(np.diff(sh_xy[:, 0])))) + 1e-5)
            eff_y_sh = abs(dy_sh) / (float(np.sum(np.abs(np.diff(sh_xy[:, 1])))) + 1e-5)

            # Directional translation check
            is_translating_across = False
            if abs(dx_hip) > 0.024 and eff_x_hip > 0.70:
                if abs(dx_sh) > 0.020 and eff_x_sh > 0.70 and np.sign(dx_hip) == np.sign(dx_sh):
                    is_translating_across = True

            is_translating_vert = False
            if abs(dy_hip) > 0.020 and eff_y_hip > 0.70:
                if abs(dy_sh) > 0.016 and eff_y_sh > 0.70 and np.sign(dy_hip) == np.sign(dy_sh):
                    is_translating_vert = True

            # Directed walk resolution
            is_directed_walk = False
            walk_type = "Walking"
            if is_translating_across and abs(dx_hip) > abs(dy_hip) * 1.5:
                is_directed_walk = True
                walk_type = "Walk Across"
            elif is_translating_vert or (abs(dh) > 0.012 and eff_h > 0.70):
                is_directed_walk = True
                walk_type = "Walking"

            probs = np.zeros(8)
            if body_speed >= self.run_thresh:
                # Running
                if dx_hip < -0.01:
                    probs[4] = 0.85  # Run Backward
                else:
                    probs[5] = 0.85  # Run
            elif is_directed_walk:
                # Walking
                if walk_type == "Walk Across":
                    probs[3] = 0.80  # Walk Across
                else:
                    probs[2] = 0.80  # Walking
            else:
                # Static States
                if pose_label == "Sitting":
                    probs[0] = 0.95  # Sitting Still
                elif pose_label == "Crouching":
                    probs[6] = 0.90  # Leaning Forward
                elif body_speed < 0.08:
                    probs[7] = 0.90  # Frozen Stand
                else:
                    probs[1] = 0.90  # Standing Still

            # Smooth probability vector using EMA
            self.smooth_probs = self.alpha * probs + (1 - self.alpha) * self.smooth_probs
            voted_idx = int(np.argmax(self.smooth_probs))
            motion_label = MOTION_LABELS[voted_idx]
            confidence = float(self.smooth_probs[voted_idx])

        elif not landmarks:
            self.smooth_probs = np.ones(8) / 8
            motion_label = "Standing Still"
            confidence = 0.0

        return motion_label, confidence, pose_label
