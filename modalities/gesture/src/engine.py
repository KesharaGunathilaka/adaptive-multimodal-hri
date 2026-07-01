import sys
import os
import time
import numpy as np

# Add configuration paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import GESTURE_COLORS, DEFAULT_CHECKPOINT

from .models import KeyPointClassifier
from .tracker import HandMotionTracker
from .transforms import calc_landmark_list, pre_process_landmark, hand_y_norm

# Landmarks index mappings
WRIST = 0
INDEX_TIP = 8
MIDDLE_MCP = 9

class GestureEngine:
    def __init__(self, model_path=DEFAULT_CHECKPOINT):
        self.tracker = HandMotionTracker()
        self.last_gesture = "None"
        self.last_time = time.time()
        
        # Landmark smoothing state (Exponential Moving Average)
        self.smoothed_landmarks = {0: None, 1: None}
        self.alpha = 0.45  # Smoothing factor

        # Initialize PyTorch model classifier
        self.keypoint_classifier = KeyPointClassifier(model_path=model_path)

    def stable_update(self, gesture):
        now = time.time()
        if gesture != self.last_gesture:
            if now - self.last_time > 0.3:  # 300ms smoothing delay
                self.last_gesture = gesture
                self.last_time = now
        return self.last_gesture

    def process(self, hand_results, frame_shape):
        """
        Processes a list of hand landmarks and returns resolved dynamic scenario name and display color.
        """
        if not hand_results:
            self.tracker.reset_inactive([])
            # Reset smoothing states
            for i in [0, 1]:
                self.smoothed_landmarks[i] = None
            return self.stable_update("No hands"), GESTURE_COLORS["No hands"]

        n_hands = len(hand_results)
        per_hand = []
        active_ids = []

        for hl in hand_results:
            lm = hl.landmark
            wx = lm[WRIST].x
            wy = lm[WRIST].y

            # Compute stable spatial ID
            hand_id = self.tracker.get_track_id(wx, wy)
            active_ids.append(hand_id)

            # ── Real-time Landmark Smoothing Filter ──
            raw_landmark_list = calc_landmark_list(np.zeros(frame_shape), hl)
            if self.smoothed_landmarks[hand_id] is None:
                self.smoothed_landmarks[hand_id] = raw_landmark_list
            else:
                self.smoothed_landmarks[hand_id] = [
                    [
                        int(self.alpha * curr[0] + (1.0 - self.alpha) * prev[0]),
                        int(self.alpha * curr[1] + (1.0 - self.alpha) * prev[1])
                    ]
                    for curr, prev in zip(raw_landmark_list, self.smoothed_landmarks[hand_id])
                ]
            smoothed_list = self.smoothed_landmarks[hand_id]

            # Write smoothed coordinates back to landmarks
            for idx, pt in enumerate(smoothed_list):
                hl.landmark[idx].x = pt[0] / frame_shape[1]
                hl.landmark[idx].y = pt[1] / frame_shape[0]

            hy = hand_y_norm(hl.landmark)
            
            # Calculate hand scale (distance from Wrist to Middle MCP)
            w_x, w_y = hl.landmark[WRIST].x, hl.landmark[WRIST].y
            m_x, m_y = hl.landmark[MIDDLE_MCP].x, hl.landmark[MIDDLE_MCP].y
            hand_scale = np.sqrt((w_x - m_x)**2 + (w_y - m_y)**2)
            if hand_scale == 0:
                hand_scale = 1.0
            
            beck_a = (hl.landmark[INDEX_TIP].y - hl.landmark[5].y) / hand_scale

            # Predict static pose index using PyTorch model on smoothed coordinates
            pre_processed_landmark_list = pre_process_landmark(smoothed_list)
            hand_sign_id, hand_sign_conf = self.keypoint_classifier(pre_processed_landmark_list)

            # Filter low-confidence predictions for specific gestures
            if hand_sign_id in [2, 3, 4, 5] and hand_sign_conf < 0.80:
                hand_sign_id = -1

            # Update tracker history
            self.tracker.update(hand_id, wx, hy, beck_a)

            per_hand.append({
                'sign': hand_sign_id,
                'sign_conf': hand_sign_conf,
                'hy': hy,
                'wx': wx,
                'id': hand_id,
                'lm': hl.landmark
            })

        # Clear inactive hands' smoothing states
        for i in [0, 1]:
            if i not in active_ids:
                self.smoothed_landmarks[i] = None
        self.tracker.reset_inactive(active_ids)

        # ── Two-Hand Scenarios (Arms Up / Arms Waving) ──────────────────────
        if n_hands == 2:
            h0 = per_hand[0]
            h1 = per_hand[1]

            h0_raised = self.tracker.is_raised(h0['id'], h0['hy'])
            h1_raised = self.tracker.is_raised(h1['id'], h1['hy'])
            both_high = h0_raised and h1_raised
            both_raising_poses = (h0['sign'] in [0, 1, -1] and h1['sign'] in [0, 1, -1])
            if both_raising_poses and both_high:
                return self.stable_update("Arms Up"), GESTURE_COLORS["Arms Up"]

            w0, bw0 = self.tracker.is_waving(h0['id'])
            w1, bw1 = self.tracker.is_waving(h1['id'])
            if (w0 or bw0) and (w1 or bw1):
                return self.stable_update("Arms Waving"), GESTURE_COLORS["Arms Waving"]

        # ── Single-Hand Scenarios (Highest hand is dominant) ──────────────────
        primary = min(per_hand, key=lambda x: x['hy'])
        sign = primary['sign']
        hy = primary['hy']
        hid = primary['id']

        is_wave, is_brief = self.tracker.is_waving(hid)

        # Beckoning
        if sign == 5 or self.tracker.is_beckoning(hid):
            return self.stable_update("Beckoning"), GESTURE_COLORS["Beckoning"]

        # Wave
        if is_wave and sign == 0:
            return self.stable_update("Wave"), GESTURE_COLORS["Wave"]

        # Brief Wave
        if is_brief and sign == 0:
            return self.stable_update("Brief Wave"), GESTURE_COLORS["Brief Wave"]

        # Pointing
        if sign == 2:
            return self.stable_update("Pointing"), GESTURE_COLORS["Pointing"]

        # Thumbs up
        if sign == 3:
            return self.stable_update("Thumbs up"), GESTURE_COLORS["Thumbs up"]

        # Thumbs down
        if sign == 4:
            return self.stable_update("Thumbs down"), GESTURE_COLORS["Thumbs down"]

        # One Hand Raised
        if sign in [0, 1, -1] and self.tracker.is_raised(hid, hy):
            return self.stable_update("One Hand Raised"), GESTURE_COLORS["One Hand Raised"]

        return self.stable_update("None"), GESTURE_COLORS["None"]
