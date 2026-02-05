import streamlit as st
import torch
import cv2
import numpy as np
import sqlite3
import bcrypt
import os
from datetime import datetime
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ================= CONFIG =================
MODEL_ID = "prithivMLmods/deepfake-detector-model-v1"
DB_PATH = "deepfake.db"
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password BLOB
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        filepath TEXT,
        prediction TEXT,
        confidence REAL,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ================= AUTH =================
def register_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        c.execute("INSERT INTO users VALUES (NULL, ?, ?)", (username, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row and bcrypt.checkpw(password.encode(), row[0]):
        return True
    return False

# ================= MODEL =================
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    model.eval()
    return processor, model

processor, model = load_model()

# ================= PREDICTION =================
def classify_image(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    fake_prob = probs[0].item()
    real_prob = probs[1].item()
    confidence = max(fake_prob, real_prob)

    if confidence < 0.6:
        label = "UNCERTAIN"
    elif fake_prob > real_prob:
        label = "FAKE"
    else:
        label = "REAL"

    return label, confidence

def save_history(username, filepath, prediction, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO history VALUES (NULL, ?, ?, ?, ?, ?)
    """, (
        username,
        filepath,
        prediction,
        confidence,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def load_history(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT filepath, prediction, confidence, timestamp
        FROM history
        WHERE username=?
        ORDER BY id ASC
    """, (username,))
    rows = c.fetchall()
    conn.close()
    return rows

# ================= UI =================
st.set_page_config("Deepfake Detector", "🕵️", layout="centered")

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("🔐 Login / Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(u, p):
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")
        if st.button("Register"):
            if register_user(u, p):
                st.success("Account created. Please login.")
            else:
                st.error("Username already exists")

    st.stop()

# ================= MAIN APP =================
username = st.session_state.user
user_dir = os.path.join(UPLOAD_DIR, username)
os.makedirs(user_dir, exist_ok=True)

st.title("🕵️ Deepfake Detection Assistant")
st.caption("Upload images. Scroll to see your full visual history.")

# ---- CHAT HISTORY (GPT STYLE) ----
history = load_history(username)

for filepath, prediction, confidence, timestamp in history:
    with st.chat_message("user"):
        st.image(filepath, use_column_width=True)

    with st.chat_message("assistant"):
        if prediction == "REAL":
            st.success("🟢 REAL")
        elif prediction == "FAKE":
            st.error("🔴 FAKE")
        else:
            st.warning("🟡 UNCERTAIN")

        st.write(f"Confidence: **{confidence:.2%}**")
        st.caption(timestamp)

# ---- UPLOAD (CHAT INPUT STYLE) ----
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and uploaded_file.name:
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
    filepath = os.path.join(user_dir, filename)

    with open(filepath, "wb") as f:
        f.write(uploaded_file.read())

    image = Image.open(filepath).convert("RGB")
    label, confidence = classify_image(image)

    save_history(username, filepath, label, confidence)
    st.rerun()

# ---- LOGOUT ----
st.markdown("---")
if st.button("Logout"):
    st.session_state.user = None
    st.rerun()
