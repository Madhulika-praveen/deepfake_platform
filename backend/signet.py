# backend/signet.py
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Global model holder
# -------------------------
_SIGNET_MODEL = None

# -------------------------
# SigNet architecture
# -------------------------
class SigNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 76 * 108, 128)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Load model (FIXED)
# -------------------------
def load_signet(path):
    global _SIGNET_MODEL

    model = SigNet()
    ckpt = torch.load(path, map_location="cpu")

    # ---- HANDLE DIFFERENT CHECKPOINT FORMATS ----
    if isinstance(ckpt, dict):
        state_dict = ckpt
    elif isinstance(ckpt, tuple):
        state_dict = ckpt[0]   # ✅ THIS IS YOUR CASE
    else:
        raise RuntimeError(f"Unsupported SigNet checkpoint type: {type(ckpt)}")

    model.load_state_dict(state_dict)
    model.eval()

    _SIGNET_MODEL = model
    return model

# -------------------------
# Preprocessing
# -------------------------
_transform = T.Compose([
    T.Grayscale(),
    T.Resize((155, 220)),
    T.ToTensor()
])

def _embed(model, img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    x = _transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(x)
    return emb.numpy()

# -------------------------
# Compare signatures
# -------------------------
def predict_signet(b1: bytes, b2: bytes):
    if _SIGNET_MODEL is None:
        raise RuntimeError("SigNet model not loaded")

    e1 = _embed(_SIGNET_MODEL, b1)
    e2 = _embed(_SIGNET_MODEL, b2)
    sim = cosine_similarity(e1, e2)[0][0]
    return float(sim)

