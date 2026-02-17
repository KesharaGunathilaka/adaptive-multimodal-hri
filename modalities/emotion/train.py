import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models.mobilenet_emotion import get_model
from utils.transforms import get_train_transforms, get_test_transforms
from config import *
import os
from tqdm import tqdm

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ImageFolder("data/train", transform=get_train_transforms())
test_dataset = ImageFolder("data/test", transform=get_test_transforms())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = get_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    # Create a progress bar for the training loader
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update the progress bar with current loss
        train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    # Validation
    model.eval()
    correct = 0
    total = 0

    # Create a progress bar for the test loader
    val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")

    with torch.no_grad():
        for images, labels in val_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update the progress bar with current accuracy
            val_pbar.set_postfix({'acc': f"{100 * correct / total:.2f}%"})

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Val Acc: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "checkpoints/best_model.pth")

print("Training Complete")