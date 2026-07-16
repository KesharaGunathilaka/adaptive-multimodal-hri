"""
Lightweight ImageNet-pretrained backbones with the classifier head
swapped for ``NUM_CLASSES`` emotion outputs. All four are candidates for
Jetson Orin Nano deployment, plus a temporal CNN-LSTM variant that models
expression dynamics over a clip (same idea as the gesture/motion modalities).

Uses the modern ``weights="DEFAULT"`` API (torchvision >= 0.13) instead of the
deprecated ``pretrained=True`` flag.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm

from config import NUM_CLASSES


def get_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=True):
    w = "DEFAULT" if pretrained else None
    model = tvm.mobilenet_v2(weights=w)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def get_mobilenet_v3_large(num_classes=NUM_CLASSES, pretrained=True):
    w = "DEFAULT" if pretrained else None
    model = tvm.mobilenet_v3_large(weights=w)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def get_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True):
    w = "DEFAULT" if pretrained else None
    model = tvm.efficientnet_b0(weights=w)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_mnasnet(num_classes=NUM_CLASSES, pretrained=True):
    w = "DEFAULT" if pretrained else None
    model = tvm.mnasnet1_0(weights=w)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


class EmotionCNNLSTM(nn.Module):
    """MobileNetV2 features per frame + LSTM over the frame sequence.

    Input is either ``(B, T, C, H, W)`` (a clip) or ``(B, C, H, W)`` (a single
    frame, treated as a length-1 sequence), so the model drops into every
    existing frame-level script unchanged. Logits come from the mean of the
    LSTM outputs, which keeps predictions stable for variable-length clips
    (real-world eval only keeps frames where a face was detected).

    Unidirectional and small (backbone 2.2M + LSTM 1.6M params, ~15 MB fp32)
    so it stays streamable on Jetson Orin Nano.
    """

    is_sequence = True  # signals clip-level scripts to feed whole sequences

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True,
                 hidden_size=256, num_layers=1, dropout=0.3):
        super().__init__()
        base = tvm.mobilenet_v2(weights="DEFAULT" if pretrained else None)
        self.features = base.features
        self.feat_dim = base.last_channel  # 1280
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(self.feat_dim, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def load_backbone(self, state_dict):
        """Init ``features`` from a plain MobileNetV2 checkpoint (best_*.pth)."""
        feat_state = {k[len("features."):]: v for k, v in state_dict.items()
                      if k.startswith("features.")}
        self.features.load_state_dict(feat_state)

    def forward(self, x):
        if x.dim() == 4:  # single frames -> length-1 sequences
            x = x.unsqueeze(1)
        b, t = x.shape[:2]
        feats = self.pool(self.features(x.flatten(0, 1))).flatten(1)
        out, _ = self.lstm(feats.view(b, t, self.feat_dim))
        return self.classifier(out.mean(dim=1))


def get_mobilenet_v2_lstm(num_classes=NUM_CLASSES, pretrained=True):
    return EmotionCNNLSTM(num_classes=num_classes, pretrained=pretrained)


# Registry used by every script and the inference code.
ALL_MODELS = {
    "MobileNetV2": get_mobilenet_v2,
    "MobileNetV3-Large": get_mobilenet_v3_large,
    "EfficientNet-B0": get_efficientnet_b0,
    "MNASNet1_0": get_mnasnet,
    "MobileNetV2-LSTM": get_mobilenet_v2_lstm,
}


def build_model(name, num_classes=NUM_CLASSES, pretrained=True):
    """Instantiate a model by registry name."""
    if name not in ALL_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(ALL_MODELS)}")
    return ALL_MODELS[name](num_classes=num_classes, pretrained=pretrained)


def safe_name(name):
    """Filesystem-safe variant of a model name (for checkpoint / report paths)."""
    return name.replace(" ", "_").replace("-", "_")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    total += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total / (1024 * 1024)
