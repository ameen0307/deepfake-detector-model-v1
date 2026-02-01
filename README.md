# Deepfake Detection 🕵️‍♂️

A deepfake image detection web application built using a pretrained vision model.

## Features
- Upload an image via a web UI
- Detect REAL, FAKE, or UNCERTAIN
- Confidence-based predictions
- No model training required

## Tech Stack
- Python 3.11
- PyTorch
- Hugging Face Transformers
- Streamlit

## Setup

```bash
git clone https://github.com/mayjary/deepfake-detection.git
cd deepfake-detection
python -m venv hf_env
source hf_env/bin/activate
pip install -r requirements.txt
