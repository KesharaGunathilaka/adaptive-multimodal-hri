import os
import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    print(f"Error: {e}")
    print("This script requires PyTorch installed.")
    sys.exit(1)

# Configurations
ROOT = os.path.dirname(os.path.abspath(__file__))
KEYPOINT_CSV = r"d:\FYP\FYP_Motion & Gesture\Gesture_final\model\keypoint_classifier\keypoint.csv"
SAVE_PATH = os.path.join(ROOT, "..", "checkpoints", "keypoint_classifier.pth")
NUM_CLASSES = 6
BATCH_SIZE = 128
EPOCHS = 150
SEED = 42

# Ensure reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

class KeypointDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

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

def rotate_keypoints(landmarks, angle_deg):
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s, c]])
    pts = landmarks.reshape(21, 2)
    rotated = np.dot(pts, R.T)
    return rotated.flatten()

def main():
    csv_path = KEYPOINT_CSV
    if not os.path.exists(csv_path):
        print(f"Error: Could not find keypoint.csv at {csv_path}")
        return

    print(f"Loading dataset from: {csv_path}")
    X_dataset = np.loadtxt(csv_path, delimiter=',', dtype='float32', usecols=list(range(1, 43)))
    y_dataset = np.loadtxt(csv_path, delimiter=',', dtype='int32', usecols=(0))

    # Split train/test
    indices = np.arange(X_dataset.shape[0])
    np.random.shuffle(indices)
    X_dataset = X_dataset[indices]
    y_dataset = y_dataset[indices]

    split_idx = int(0.75 * X_dataset.shape[0])
    X_train, X_test = X_dataset[:split_idx], X_dataset[split_idx:]
    y_train, y_test = y_dataset[:split_idx], y_dataset[split_idx:]

    # Data Augmentation (Rotation)
    print("Applying rotation augmentation...")
    X_aug = []
    y_aug = []
    for x_val, y_val in zip(X_train, y_train):
        X_aug.append(x_val)
        y_aug.append(y_val)
        
        if y_val in [0, 1, 2]:
            angles = np.random.uniform(-180, 180, size=5)
        elif y_val in [3, 4]:
            angles = np.random.uniform(-30, 30, size=5)
        else:
            angles = np.random.uniform(-45, 45, size=5)
            
        for angle in angles:
            X_aug.append(rotate_keypoints(x_val, angle))
            y_aug.append(y_val)

    X_train = np.array(X_aug, dtype='float32')
    y_train = np.array(y_aug, dtype='int32')

    train_dataset = KeypointDataset(X_train, y_train)
    test_dataset = KeypointDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeyPointClassifierNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_acc = correct / total

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"Training completed. Best validation accuracy: {best_acc*100:.2f}%")
    print(f"PyTorch model saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
