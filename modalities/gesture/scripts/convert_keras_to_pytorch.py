import os
import sys
import numpy as np

try:
    import tensorflow as tf
    import torch
    import torch.nn as nn
except ImportError as e:
    print(f"Error: {e}")
    print("This script requires both 'tensorflow' and 'torch' installed.")
    sys.exit(1)

ROOT = os.path.dirname(os.path.abspath(__file__))
KERAS_PATH = os.path.join(ROOT, "..", "checkpoints", "keypoint_classifier.hdf5")
PYTORCH_PATH = os.path.join(ROOT, "..", "checkpoints", "keypoint_classifier.pth")

class KeyPointClassifierNet(nn.Module):
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

def main():
    if not os.path.exists(KERAS_PATH):
        print(f"Keras model not found at: {KERAS_PATH}")
        return

    print("Loading Keras model...")
    keras_model = tf.keras.models.load_model(KERAS_PATH)

    py_model = KeyPointClassifierNet(num_classes=6)
    py_state_dict = py_model.state_dict()
    
    keras_dense_layers = [l for l in keras_model.layers if "dense" in l.name.lower()]
    pytorch_fc_names = ["fc1.weight", "fc1.bias", 
                        "fc2.weight", "fc2.bias", 
                        "fc3.weight", "fc3.bias", 
                        "fc4.weight", "fc4.bias"]

    state_dict_updates = {}
    for i, keras_layer in enumerate(keras_dense_layers):
        weights, bias = keras_layer.get_weights()
        weight_name = pytorch_fc_names[i * 2]
        bias_name = pytorch_fc_names[i * 2 + 1]

        state_dict_updates[weight_name] = torch.tensor(weights.T, dtype=torch.float32)
        state_dict_updates[bias_name] = torch.tensor(bias, dtype=torch.float32)

    py_state_dict.update(state_dict_updates)
    py_model.load_state_dict(py_state_dict)
    py_model.eval()

    torch.save(py_model.state_dict(), PYTORCH_PATH)
    print(f"Successfully converted and saved PyTorch weights to: {PYTORCH_PATH}")

if __name__ == "__main__":
    main()
