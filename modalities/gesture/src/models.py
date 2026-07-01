import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import model checkpoint path from configuration
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DEFAULT_CHECKPOINT

class KeyPointClassifierNet(nn.Module):
    """
    3-Layer Multi-Layer Perceptron (MLP) architecture matching Keras model.
    """
    def __init__(self, num_classes=6):
        super(KeyPointClassifierNet, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(42, 64)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.relu1(self.fc1(x))
        x = self.dropout2(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout3(x)
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

class KeyPointClassifier(object):
    def __init__(self, model_path=DEFAULT_CHECKPOINT, num_classes=6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KeyPointClassifierNet(num_classes=num_classes)
        
        # Load state dictionary
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: PyTorch checkpoint not found at {model_path}.")
            
        self.model.to(self.device).eval()

    def __call__(self, landmark_list):
        # Convert landmarks to PyTorch tensor
        input_tensor = torch.tensor([landmark_list], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1).squeeze(0)

        # Get class ID and probability score
        result_index = torch.argmax(probabilities).item()
        result_confidence = probabilities[result_index].item()

        return result_index, result_confidence
