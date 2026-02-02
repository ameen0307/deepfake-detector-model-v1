import streamlit as st
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image

MODEL_ID = "prithivMLmods/deepfake-detector-model-v1"

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ Deepfake Image Detector")
st.write("Upload an image and check whether it is **REAL**, **FAKE**, or **UNCERTAIN**.")

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        use_fast=False
    )
    model = SiglipForImageClassification.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        fake_prob = probs[0].item()
        real_prob = probs[1].item()
        confidence = max(fake_prob, real_prob)

        if confidence < 0.6:
            label = "UNCERTAIN"
            color = "🟡"
        elif fake_prob > real_prob:
            label = "FAKE"
            color = "🔴"
        else:
            label = "REAL"
            color = "🟢"

    st.markdown("---")
    st.subheader(f"{color} Prediction: **{label}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    st.markdown("### Probabilities")
    st.write(f"Fake: {fake_prob:.3f}")
    st.write(f"Real: {real_prob:.3f}")

    st.markdown("---")
    st.caption(
        "Low confidence means the model is unsure. "
        "Results improve with clear, well-lit, front-facing faces."
    )
