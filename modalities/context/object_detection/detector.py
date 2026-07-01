from collections import deque, defaultdict
from pathlib import Path

from ultralytics import YOLO

# Config import works both as a package (pipeline:
# `modalities.context.object_detection.detector`) and as a top-level script
# (running the realtime/video scripts from inside this folder).
try:
    from . import config as cfg
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as cfg


class ContextDetector:
    """YOLO object detector tailored to HRI context.

    Detects only the HRI-relevant subset of COCO classes, groups them into
    higher-level categories, optionally tracks them across frames for stable
    identities, and reports a temporally smoothed set of objects that are
    reliably present (rather than noisy per-frame detections).
    """

    def __init__(
        self,
        use_tracking=cfg.USE_TRACKING,
        smooth_window=cfg.SMOOTH_WINDOW,
        presence_min_ratio=cfg.PRESENCE_MIN_RATIO,
    ):
        # Load the model (downloads automatically if not present)
        self.model = YOLO(cfg.MODEL_PATH)

        # Dynamically find the COCO class IDs for our target objects
        self.target_ids = [
            class_id
            for class_id, class_name in self.model.names.items()
            if class_name in cfg.TARGET_CLASSES
        ]

        self.use_tracking = use_tracking
        self.presence_min_ratio = presence_min_ratio
        # Rolling history of the category set seen in each recent frame.
        self._history = deque(maxlen=smooth_window)

    def process_frame(self, frame):
        """Run inference (or tracking) and return HRI context for one frame.

        Returns:
            annotated_frame: frame with boxes drawn (Ultralytics plotter)
            detections: list of dicts {label, category, confidence, bbox, track_id}
            counts: {category: count} for THIS frame (raw, may flicker)
            stable_categories: set of categories reliably present over the window
        """
        if self.use_tracking:
            results = self.model.track(
                source=frame,
                conf=cfg.CONFIDENCE_THRESHOLD,
                imgsz=cfg.INFERENCE_IMG_SIZE,
                classes=self.target_ids,
                persist=True,
                tracker=cfg.TRACKER_CONFIG,
                verbose=False,
            )[0]
        else:
            results = self.model.predict(
                source=frame,
                conf=cfg.CONFIDENCE_THRESHOLD,
                imgsz=cfg.INFERENCE_IMG_SIZE,
                classes=self.target_ids,
                verbose=False,
            )[0]

        detections = []
        counts = defaultdict(int)
        frame_categories = set()

        for box in results.boxes:
            cls_id = int(box.cls)
            label = self.model.names[cls_id]
            category = cfg.OBJECT_CATEGORIES.get(label, "other")
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            track_id = int(box.id) if box.id is not None else None

            detections.append(
                {
                    "label": label,
                    "category": category,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "track_id": track_id,
                }
            )

            counts[category] += 1
            frame_categories.add(category)

        # Update temporal history and compute the stable category set.
        self._history.append(frame_categories)
        stable_categories = self._stable_categories()

        # Generate an annotated frame using Ultralytics' built-in plotter
        annotated_frame = results.plot()

        return annotated_frame, detections, dict(counts), stable_categories

    def reset(self):
        """Clear temporal history and tracker state (e.g. switching to a new video).

        Without this, ByteTrack IDs and the smoothing window would carry over
        from the end of one clip into the start of the next.
        """
        self._history.clear()
        self.model.predictor = None

    def _stable_categories(self):
        """Categories present in >= presence_min_ratio of the recent frames."""
        if not self._history:
            return set()

        tally = defaultdict(int)
        for categories in self._history:
            for category in categories:
                tally[category] += 1

        window = len(self._history)
        return {
            category
            for category, hits in tally.items()
            if hits / window >= self.presence_min_ratio
        }
