import torch.nn as nn
import torchvision.models as models

from config import NUM_CLASSES


def get_mobilenet_v2(num_classes=NUM_CLASSES):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def get_mobilenet_v3_large(num_classes=NUM_CLASSES):
    model = models.mobilenet_v3_large(pretrained=True)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def get_efficientnet_b0(num_classes=NUM_CLASSES):
    model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_mnasnet(num_classes=NUM_CLASSES):
    model = models.mnasnet1_0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


ALL_MODELS = {
    "MobileNetV2": get_mobilenet_v2,
    "MobileNetV3-Large": get_mobilenet_v3_large,
    "EfficientNet-B0": get_efficientnet_b0,
    "MNASNet1_0": get_mnasnet,
}


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    total += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total / (1024 * 1024)
