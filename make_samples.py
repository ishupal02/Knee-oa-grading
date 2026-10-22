"""
Copy one example X-ray per grade from dataset/test into samples/ for the app's
sample buttons. For each grade it takes the first image (in a fixed shuffled
order) that the model classifies correctly, so the demo opens with clear
examples. The "Model performance" tab shows the honest, full results.

    python make_samples.py
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

tf = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load("knee_oa_model_best.pt", map_location="cpu"))
model.eval()

out = Path("samples")
out.mkdir(exist_ok=True)

for g in range(5):
    files = sorted(Path("dataset/test", str(g)).glob("*"))
    random.Random(0).shuffle(files)
    chosen = files[0]
    for f in files[:60]:
        with torch.no_grad():
            pred = model(tf(Image.open(f).convert("RGB")).unsqueeze(0)).argmax().item()
        if pred == g:
            chosen = f
            break
    Image.open(chosen).convert("L").save(out / f"grade_{g}.png")
    print(f"grade {g}: {chosen.name}")
