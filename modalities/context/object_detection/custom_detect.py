from pathlib import Path
import sys

# Ensure the package can find its own config regardless of working directory.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ultralytics import YOLO
import config as cfg

# ─────────────────────────────────────────────
# Load YOLO-World model once
# ─────────────────────────────────────────────
# YOLO-World auto-downloads on first use. After set_classes() it will
# only look for the objects you listed in CUSTOM_CLASSES.
_model_path = _THIS_DIR / cfg.MODEL_PATH
if _model_path.exists():
    model = YOLO(str(_model_path))
else:
    # Auto-download from ultralytics hub
    model = YOLO(cfg.MODEL_PATH)

# Tell the model exactly which objects to detect
model.set_classes(cfg.CUSTOM_CLASSES)


# ─────────────────────────────────────────────
# Zone helpers
# ─────────────────────────────────────────────
def _get_zone_x(cx_norm: float) -> str:
    """Return the horizontal zone label for a normalised x centre."""
    for zone, (lo, hi) in cfg.ZONES_X.items():
        if lo <= cx_norm < hi:
            return zone
    return "right"  # edge case: cx_norm == 1.0


def _get_zone_y(cy_norm: float) -> str:
    """Return the vertical zone label for a normalised y centre."""
    for zone, (lo, hi) in cfg.ZONES_Y.items():
        if lo <= cy_norm < hi:
            return zone
    return "bottom"


def _get_confidence_threshold(category: str) -> float:
    """Return the confidence threshold for a category, with fallback."""
    return cfg.CATEGORY_THRESHOLDS.get(category, cfg.CONFIDENCE_THRESHOLD)


# ─────────────────────────────────────────────
# Main detection function
# ─────────────────────────────────────────────
def detect_context_objects(frame):
    """
    Run YOLOv8 on *frame* and return a structured detection dict.

    Returns
    -------
    dict with keys:
        "counts"     : {category: int}          — number of objects per category
        "detections" : list of per-object dicts  — each with category, label,
                       confidence, bbox (xyxy), and zone
        "zones"      : {zone_label: [category…]} — which objects are in each zone
    """
    h, w = frame.shape[:2]

    results = model(
        frame,
        verbose=False,
        imgsz=cfg.INFERENCE_IMG_SIZE,
        conf=cfg.CONFIDENCE_THRESHOLD,
        iou=cfg.INFERENCE_IOU_THRESHOLD,
        augment=cfg.INFERENCE_AUGMENT,
    )[0]

    # Initialise counts for every known category
    counts = {cat: 0 for cat in set(cfg.OBJECT_CATEGORIES.values())}
    detections = []
    zones = {}

    for box in results.boxes:
        cls_id = int(box.cls)
        label = model.names[cls_id]

        # Map to category — if not in OBJECT_CATEGORIES, use the label itself
        category = cfg.OBJECT_CATEGORIES.get(label, label)
        conf = float(box.conf)

        # Apply per-category confidence threshold
        if conf < _get_confidence_threshold(category):
            continue

        # Bounding box (pixel coords)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Zone from normalised centre
        cx_norm = ((x1 + x2) / 2) / w
        cy_norm = ((y1 + y2) / 2) / h
        zone_x = _get_zone_x(cx_norm)
        zone_y = _get_zone_y(cy_norm)
        zone_label = f"{zone_x}-{zone_y}"

        counts[category] += 1

        det = {
            "category": category,
            "label": label,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
            "zone": zone_label,
        }
        detections.append(det)

        # Aggregate by zone
        zones.setdefault(zone_label, []).append(category)

    return {
        "counts": counts,
        "detections": detections,
        "zones": zones,
    }


# ─────────────────────────────────────────────
# Temporal Smoothing (tracker)
# ─────────────────────────────────────────────
def _iou(box_a, box_b):
    """Compute Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


class DetectionSmoother:
    """
    Temporal smoother that stabilises frame-by-frame detections.

    - Objects must be detected for ``MIN_HITS_TO_SHOW`` consecutive frames
      before they are drawn (prevents single-frame false positives).
    - Once shown, objects stay visible for ``PERSISTENCE_FRAMES`` frames
      after their last detection (prevents pop-out flickering).
    - Uses IoU matching to associate detections across frames.

    Usage::

        smoother = DetectionSmoother()
        for frame in video:
            raw = detect_context_objects(frame)
            smoothed = smoother.update(raw)
            draw(frame, smoothed)
    """

    def __init__(self):
        # Each tracked object: {det dict, hits, frames_since_seen, visible}
        self._tracks = []

    def update(self, raw_result):
        """
        Accept a raw detection result from ``detect_context_objects``
        and return a smoothed version.
        """
        new_dets = raw_result["detections"]

        # ── Match new detections to existing tracks ──
        matched_track_ids = set()
        matched_det_ids = set()

        # Greedy IoU matching: best match first
        pairs = []
        for t_idx, track in enumerate(self._tracks):
            for d_idx, det in enumerate(new_dets):
                if det["category"] != track["det"]["category"]:
                    continue
                score = _iou(track["det"]["bbox"], det["bbox"])
                if score >= cfg.IOU_MATCH_THRESHOLD:
                    pairs.append((score, t_idx, d_idx))

        pairs.sort(key=lambda x: x[0], reverse=True)

        for score, t_idx, d_idx in pairs:
            if t_idx in matched_track_ids or d_idx in matched_det_ids:
                continue
            # Update track with new detection
            self._tracks[t_idx]["det"] = new_dets[d_idx]
            self._tracks[t_idx]["hits"] += 1
            self._tracks[t_idx]["frames_since_seen"] = 0
            if self._tracks[t_idx]["hits"] >= cfg.MIN_HITS_TO_SHOW:
                self._tracks[t_idx]["visible"] = True
            matched_track_ids.add(t_idx)
            matched_det_ids.add(d_idx)

        # ── Age unmatched tracks ──
        for t_idx, track in enumerate(self._tracks):
            if t_idx not in matched_track_ids:
                track["frames_since_seen"] += 1

        # ── Create new tracks for unmatched detections ──
        for d_idx, det in enumerate(new_dets):
            if d_idx not in matched_det_ids:
                self._tracks.append({
                    "det": det,
                    "hits": 1,
                    "frames_since_seen": 0,
                    "visible": (cfg.MIN_HITS_TO_SHOW <= 1),
                })

        # ── Remove dead tracks ──
        self._tracks = [
            t for t in self._tracks
            if t["frames_since_seen"] <= cfg.PERSISTENCE_FRAMES
        ]

        # ── Build smoothed output ──
        smoothed_dets = []
        smoothed_counts = {cat: 0 for cat in set(cfg.OBJECT_CATEGORIES.values())}
        smoothed_zones = {}

        for track in self._tracks:
            if not track["visible"]:
                continue

            det = track["det"]
            smoothed_dets.append(det)
            smoothed_counts[det["category"]] += 1
            smoothed_zones.setdefault(det["zone"], []).append(det["category"])

        return {
            "counts": smoothed_counts,
            "detections": smoothed_dets,
            "zones": smoothed_zones,
        }

