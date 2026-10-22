"""
Predict the KL grade (0-4) of a single knee X-ray image using the trained
model (knee_oa_model_best.pt).

Usage:
    python predict.py path/to/xray_image.png
"""

import sys

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

MODEL_PATH = "knee_oa_model_best.pt"
NUM_CLASSES = 5
IMAGE_SIZE = 224
CLASS_NAMES = ["0", "1", "2", "3", "4"]

KL_DESCRIPTIONS = {
    "0": "Grade 0 - No radiographic signs of osteoarthritis",
    "1": "Grade 1 - Doubtful joint space narrowing, possible osteophytes",
    "2": "Grade 2 - Definite osteophytes, possible joint space narrowing",
    "3": "Grade 3 - Moderate osteophytes, definite joint space narrowing",
    "4": "Grade 4 - Large osteophytes, severe joint space narrowing",
}


def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def predict(image_path):
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0)

    model = load_model()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        idx = probs.argmax().item()

    print(f"\nImage: {image_path}")
    print(f"Predicted KL grade: {CLASS_NAMES[idx]} ({probs[idx].item()*100:.1f}% confidence)")
    print(f"  -> {KL_DESCRIPTIONS[CLASS_NAMES[idx]]}")
    print("\nAll class probabilities:")
    for cls, p in zip(CLASS_NAMES, probs.tolist()):
        print(f"  Grade {cls}: {p*100:5.1f}%  {'#' * int(p * 40)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/xray_image.png")
        sys.exit(1)
    predict(sys.argv[1])
