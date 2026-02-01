import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image

MODEL_PATH = "./"   # current folder

processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = SiglipForImageClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    fake_prob = probs[0].item()
    real_prob = probs[1].item()

    label = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = max(fake_prob, real_prob)

    print("\nImage:", image_path)
    print("Prediction:", label)
    print(f"Confidence: {confidence:.2%}")
    print(f"Fake: {fake_prob:.3f} | Real: {real_prob:.3f}")

# CHANGE THESE PATHS
predict("sample.jpg")
predict("real.jpg")
