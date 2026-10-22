"""
Compute test-set results for the app's "Model performance" tab.
Run once from the folder that has dataset/ and knee_oa_model_best.pt:

    python evaluate.py
"""

import json

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader

NUM_CLASSES = 5

tf = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
data = datasets.ImageFolder("dataset/test", transform=tf)
assert data.classes == ["0", "1", "2", "3", "4"], data.classes

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("knee_oa_model_best.pt", map_location="cpu"))
model.eval()

cm = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
with torch.no_grad():
    for x, y in DataLoader(data, batch_size=32):
        for true, pred in zip(y.tolist(), model(x).argmax(dim=1).tolist()):
            cm[true][pred] += 1

n = sum(map(sum, cm))
acc = sum(cm[i][i] for i in range(NUM_CLASSES)) / n
json.dump({"accuracy": acc, "n": n, "confusion": cm}, open("metrics.json", "w"))
print(f"Test accuracy: {acc:.3f} on {n} images -> metrics.json written")
