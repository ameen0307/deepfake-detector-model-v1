import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os

# ===================== CONFIG =====================
DEVICE = "cpu"
WEIGHTS_PATH = "weights/xception_df.pth"
IMG_SIZE = 299
MAX_FRAMES = 40
# =================================================

# ----------------- Model Definition -----------------
# Xception-style CNN (simplified, inference-only)
import torch.nn as nn
import torchvision.models as models

class DeepfakeXception(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.xception(weights=None) if hasattr(models, "xception") else models.resnet50(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ----------------- Load Model -----------------
_model = None

def load_model():
    global _model
    if _model is None:
        model = DeepfakeXception()
        state = torch.load(WEIGHTS_PATH, map_location=DEVICE)

        # supports both raw state_dict and wrapped checkpoints
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)

        model.to(DEVICE)
        model.eval()
        _model = model
    return _model

# ----------------- Transforms -----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------- Face Extraction -----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    # take the largest face
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face = frame[y:y+h, x:x+w]
    return face

# ----------------- Video Prediction -----------------
def predict_video(video_path):
    model = load_model()

    cap = cv2.VideoCapture(video_path)
    scores = []
    frame_count = 0

    while cap.isOpened() and frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        face = extract_face(frame)
        if face is None:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(face_rgb)
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            prob_fake = torch.sigmoid(logits).item()
            scores.append(prob_fake)

        frame_count += 1

    cap.release()

    if len(scores) < 5:
        return "UNCERTAIN", 0.0

    mean_fake = float(np.mean(scores))
    std_dev = float(np.std(scores))

    # Conservative decision thresholds
    if mean_fake > 0.65:
        return "FAKE", mean_fake
    elif mean_fake < 0.35:
        return "REAL", 1 - mean_fake
    else:
        return "UNCERTAIN", max(mean_fake, 1 - mean_fake)
