import collections
import numpy as np

class HandMotionTracker:
    def __init__(self):
        # Track up to 2 hands stably using spatial coordinates
        self.x_hist = {0: collections.deque(maxlen=25), 1: collections.deque(maxlen=25)}
        self.y_hist = {0: collections.deque(maxlen=25), 1: collections.deque(maxlen=25)}
        self.beck_hist = {0: collections.deque(maxlen=15), 1: collections.deque(maxlen=15)}
        self.last_positions = {0: None, 1: None}

    def get_track_id(self, wx, wy):
        """Assign ID (0 or 1) based on proximity to previous frame's hand coordinates."""
        if self.last_positions[0] is None and self.last_positions[1] is None:
            self.last_positions[0] = (wx, wy)
            return 0

        if self.last_positions[0] is not None and self.last_positions[1] is None:
            d0 = (wx - self.last_positions[0][0])**2 + (wy - self.last_positions[0][1])**2
            if d0 < 0.06:
                self.last_positions[0] = (wx, wy)
                return 0
            else:
                self.last_positions[1] = (wx, wy)
                return 1

        if self.last_positions[0] is None and self.last_positions[1] is not None:
            d1 = (wx - self.last_positions[1][0])**2 + (wy - self.last_positions[1][1])**2
            if d1 < 0.06:
                self.last_positions[1] = (wx, wy)
                return 1
            else:
                self.last_positions[0] = (wx, wy)
                return 0

        d0 = (wx - self.last_positions[0][0])**2 + (wy - self.last_positions[0][1])**2
        d1 = (wx - self.last_positions[1][0])**2 + (wy - self.last_positions[1][1])**2
        if d0 < d1:
            self.last_positions[0] = (wx, wy)
            return 0
        else:
            self.last_positions[1] = (wx, wy)
            return 1

    def update(self, hand_id, wx, wy, beck_angle):
        self.x_hist[hand_id].append(wx)
        self.y_hist[hand_id].append(wy)
        self.beck_hist[hand_id].append(beck_angle)

    def is_raised(self, hand_id, current_hy):
        """
        Check if the hand is raised based on height threshold and upward vertical trajectory.
        """
        if current_hy < 0.35:
            return True
        h = self.y_hist[hand_id]
        if len(h) < 10:
            return current_hy < 0.42
        
        arr = list(h)
        first_half = arr[:len(arr)//2]
        max_past_y = max(first_half)
        current_y = arr[-1]
        
        # Checked vertical travel (moving up = y coordinate decreases)
        if (max_past_y - current_y) > 0.12 and current_y < 0.45:
            return True
        return current_hy < 0.42

    def is_waving(self, hand_id):
        h = self.x_hist[hand_id]
        if len(h) < 10:
            return False, False

        arr = np.array(h)
        excursion = arr.max() - arr.min()
        diffs = np.diff(arr)
        reversals = int(np.sum(np.diff(np.sign(diffs)) != 0))

        # Horizontal waving rules
        wave = excursion > 0.14 and reversals >= 2
        brief_wave = excursion > 0.08 and reversals >= 1 and len(h) <= 15
        return wave, brief_wave

    def is_beckoning(self, hand_id):
        h = self.beck_hist[hand_id]
        if len(h) < 8:
            return False

        arr = np.array(h)
        excursion = arr.max() - arr.min()
        diffs = np.diff(arr)
        reversals = int(np.sum(np.diff(np.sign(diffs)) != 0))

        return excursion > 0.15 and reversals >= 2

    def reset_inactive(self, active_ids):
        """Clear tracking history for hands that disappeared from the frame."""
        for i in [0, 1]:
            if i not in active_ids:
                self.last_positions[i] = None
                self.x_hist[i].clear()
                self.y_hist[i].clear()
                self.beck_hist[i].clear()
