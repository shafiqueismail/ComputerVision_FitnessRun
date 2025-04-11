import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ============================
# CONFIGURATION
# ============================

data_root = "./CustomModels/imageClassificationModel"
train_dir = os.path.join(data_root, "Train")
val_dir = os.path.join(data_root, "Validate")
test_dir = os.path.join(data_root, "finalTest")

batch_size = 32
num_epochs = 10
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# DATA TRANSFORMS
# ============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ============================
# LOAD DATASETS
# ============================

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ============================
# LOAD MODEL
# ============================

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# ============================
# LOSS & OPTIMIZER
# ============================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ============================
# TRAINING LOOP
# ============================

print("Starting training...\n")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {running_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    print(f"Validation Accuracy: {val_accuracy:.2%}\n")

# ============================
# FINAL TEST ACCURACY
# ============================

print("Evaluating on final test set...\n")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total
print(f"Final Test Accuracy: {test_accuracy:.2%}")

# ============================
# SAVE MODEL
# ============================

model_path = os.path.join(data_root, "squat_classifier.pt")
torch.save(model.state_dict(), model_path)
print(f"Model saved as {model_path}")

# ============================
# SHOW TEST IMAGES WITH RESULTS
# ============================

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = (inp * 0.5) + 0.5  # unnormalize
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

class_names = test_dataset.classes  # ['invalid_squat', 'valid_squat']

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for i in range(images.size(0)):
            img = images[i].cpu()
            actual = class_names[labels[i].item()]
            predicted = class_names[preds[i].item()]
            result = "correct" if actual == predicted else "incorrect"
            imshow(img, f"Actual: {actual} | Estimated: {predicted} | {result}")


# CrossEntropyLoss: good for multi-class classification.

# Adam: adjusts learning rate automatically per weight.

# Forward pass (compute predictions).

# Backward pass (compute gradients).

# Optimizer step (update weights).