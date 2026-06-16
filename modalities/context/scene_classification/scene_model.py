import torch.nn as nn
from torchvision import models


class SceneModel(nn.Module):
    def __init__(
        self, num_classes=2
    ):  # target environments: classroom, kitchen
        super(SceneModel, self).__init__()

        # EfficientNet-B0 is the optimal balance of accuracy and edge speed.
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # EfficientNet-B0's classifier is a Sequential block: (0): Dropout, (1): Linear
        in_features = self.model.classifier[1].in_features

        # Replace the final linear layer to output your specific scene classes
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        self.fc = self.model.classifier[1]

    def forward(self, x):
        return self.model(x)
