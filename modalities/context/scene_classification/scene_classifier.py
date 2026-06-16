"""Reusable scene (environment) classifier for the context model.

Wraps the trained EfficientNet-B0 scene model behind a simple `predict(frame)`
interface with temporal smoothing and confidence gating, so it can be used both
by the realtime/video scripts and by the fused context pipeline.
"""

import json
from collections import deque
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms

# Make repo-root imports work regardless of where this is launched from.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.scene_classification.scene_model import SceneModel

# Canonical class order. This MUST match torchvision ImageFolder, which sorts
# class folders ALPHABETICALLY. The trained scene.pth therefore maps
# index 0->classroom, 1->kitchen. (Office was dropped: captured "classroom"
# clips were confidently misread as office, so we model only the two
# environments we actually deploy in.) Read from classes.json when available.
DEFAULT_CLASSES = ["classroom", "kitchen"]

_DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "scene_model" / "scene.pth"


class SceneClassifier:
    def __init__(
        self,
        weights_path=None,
        classes=None,
        device=None,
        smooth_window=15,
        conf_threshold=0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        self.classes = classes or self._load_classes(weights_path)
        self.conf_threshold = conf_threshold

        if not weights_path.exists():
            raise FileNotFoundError(f"Scene model weights not found: {weights_path}")

        self.model = SceneModel(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(
            torch.load(str(weights_path), map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # Must match the training/validation transforms exactly.
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self._prob_history = deque(maxlen=smooth_window)

    def _load_classes(self, weights_path):
        """Prefer a classes.json saved next to the weights; else fall back."""
        classes_file = weights_path.parent / "classes.json"
        if classes_file.exists():
            try:
                return json.loads(classes_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return list(DEFAULT_CLASSES)

    def reset(self):
        """Clear temporal smoothing history (e.g. when switching video sources)."""
        self._prob_history.clear()

    @torch.no_grad()
    def predict(self, frame_bgr):
        """Classify a single BGR frame (as read from OpenCV).

        Returns a dict:
            label:          smoothed scene label, or "uncertain" below threshold
            confidence:     smoothed confidence for that label
            raw_label:      argmax of this single frame (no smoothing)
            raw_confidence: confidence of the raw label
            probs:          {class: smoothed_probability}
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

        probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        raw_idx = int(probs.argmax())

        # Temporal smoothing: average the recent probability vectors.
        self._prob_history.append(probs)
        avg_probs = np.mean(self._prob_history, axis=0)
        smooth_idx = int(avg_probs.argmax())
        smooth_conf = float(avg_probs[smooth_idx])

        label = self.classes[smooth_idx] if smooth_conf >= self.conf_threshold else "uncertain"

        return {
            "label": label,
            "confidence": smooth_conf,
            "raw_label": self.classes[raw_idx],
            "raw_confidence": float(probs[raw_idx]),
            "probs": {c: float(p) for c, p in zip(self.classes, avg_probs)},
        }
