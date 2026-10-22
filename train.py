"""
Simple Knee Osteoarthritis Classifier (5-class KL grading: 0-4)

Uses the Kaggle "Knee Osteoarthritis Dataset with Severity" dataset, which
comes pre-organized as:

    dataset/
        train/
            0/  1/  2/  3/  4/   (KL grade folders, each full of images)
        val/
            0/  1/  2/  3/  4/
        test/
            0/  1/  2/  3/  4/

No manual preprocessing needed -- this uses torchvision's ImageFolder, which
automatically reads the folder names as class labels.

Usage:
    python train.py

That's it. Trains a ResNet-18 (pretrained, fine-tuned) for a few epochs and
prints test accuracy at the end.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models

# ---------------------------------------------------------------------------
# Settings -- edit these if your dataset folder is named differently
# ---------------------------------------------------------------------------
DATA_DIR = "dataset"     # expects dataset/train, dataset/val, dataset/test
NUM_CLASSES = 5          # KL grades 0-4
BATCH_SIZE = 32
NUM_EPOCHS = 15          # more epochs, but we'll keep the BEST one, not the last
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Training gets light augmentation (helps reduce overfitting -- the model
# sees slightly different versions of each image every epoch instead of
# memorizing exact pixels). Val/test stay un-augmented so evaluation is fair.
train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.15, contrast=0.15),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
val_data = datasets.ImageFolder(f"{DATA_DIR}/val", transform=eval_transform)
test_data = datasets.ImageFolder(f"{DATA_DIR}/test", transform=eval_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

print(f"Train: {len(train_data)} images | Val: {len(val_data)} | Test: {len(test_data)}")
print(f"Classes (KL grades): {train_data.classes}")

# ---------------------------------------------------------------------------
# Model -- pretrained ResNet-18, final layer swapped for our 5 classes
# ---------------------------------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)


def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Training loop -- keeps the BEST validation checkpoint, not just the last
# epoch (the last epoch is often slightly overfit compared to the best one).
# ---------------------------------------------------------------------------
best_val_acc = -1.0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_data)
    val_acc = evaluate(val_loader)
    scheduler.step(val_acc)

    marker = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "knee_oa_model_best.pt")
        marker = "  <- new best, saved"

    print(f"Epoch {epoch}/{NUM_EPOCHS} | train_loss={train_loss:.4f} | val_accuracy={val_acc:.3f}{marker}")

# ---------------------------------------------------------------------------
# Final test evaluation -- using the BEST checkpoint, not the last epoch
# ---------------------------------------------------------------------------
model.load_state_dict(torch.load("knee_oa_model_best.pt"))
test_acc = evaluate(test_loader)
print(f"\nBest val accuracy during training: {best_val_acc:.3f}")
print(f"Final test accuracy (best checkpoint): {test_acc:.3f}")
print("Best model saved to knee_oa_model_best.pt")
