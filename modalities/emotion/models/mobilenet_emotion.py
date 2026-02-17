import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES


def get_model():
    model = models.mobilenet_v2(pretrained=True)

    # Replace final layer
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

    return model
