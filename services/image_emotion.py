from pathlib import Path
import torch
import io
from PIL import Image
from torchvision import transforms

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "image_model.pt"

# -----------------------------
# Labels (ArtEmis)
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
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# Lazy model loader (IMPORTANT FIX)
# -----------------------------
_model = None

def load_model():
    global _model
    if _model is None:
        _model = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )
        _model.eval()
    return _model

# -----------------------------
# Prediction
# -----------------------------
def predict_image_emotion(image_bytes):
    model = load_model()

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