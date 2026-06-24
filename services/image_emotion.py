# services/image_emotion.py
from pathlib import Path

import os
import torch
import io
from PIL import Image
from torchvision import transforms

# -----------------------------
# Model path (robust)
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "image_model.pt"
# -----------------------------
# Load model
# -----------------------------
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.eval()

# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# Correct ArtEmis labels
# -----------------------------
EMOTIONS = [
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
    "something else"
]

# -----------------------------
# Prediction function
# -----------------------------
def predict_image_emotion(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze(0)

    probs = probs.tolist()

    distribution = {
        EMOTIONS[i]: float(probs[i])
        for i in range(len(EMOTIONS))
    }

    top_idx = int(torch.argmax(torch.tensor(probs)))

    return {
        "emotion": EMOTIONS[top_idx],
        "distribution": distribution
    }