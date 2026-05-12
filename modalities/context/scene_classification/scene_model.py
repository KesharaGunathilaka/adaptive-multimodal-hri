import torch.nn as nn
from torchvision import models


class SceneModel(nn.Module):
    def __init__(self, num_classes=6):
        super(SceneModel, self).__init__()

        self.model = models.mobilenet_v3_small(pretrained=True)
        self.model.classifier[3] = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.model(x)
