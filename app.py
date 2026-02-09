import streamlit as st
import torch
import sqlite3
import bcrypt
import os
import cv2
import hashlib
import numpy as np
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
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
        filepath TEXT UNIQUE,
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
    return row and bcrypt.checkpw(password.encode(), row[0])

# ================= MODEL =================
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model

processor, model = load_model()

# ================= HELPERS =================
def file_hash(data: bytes):
    return hashlib.md5(data).hexdigest()

def classify_image(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    # Correct label mapping for this model
    real_prob = probs[0].item()
    fake_prob = probs[1].item()

    confidence = max(real_prob, fake_prob)

    if confidence < 0.6:
        label = "UNCERTAIN"
    elif fake_prob > real_prob:
        label = "FAKE"
    else:
        label = "REAL"

    return label, confidence, fake_prob

# ================= VIDEO LOGIC (FIXED) =================
def classify_video(video_path, frame_interval=20, max_frames=40):
    cap = cv2.VideoCapture(video_path)
    fake_scores = []
    real_scores = []
    timeline = []

    frame_idx = 0
    used = 0

    while cap.isOpened() and used < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            label, conf, fake_prob = classify_image(image)

            timeline.append(fake_prob)

            if label == "FAKE":
                fake_scores.append(conf)
            elif label == "REAL":
                real_scores.append(conf)

            used += 1

        frame_idx += 1

    cap.release()

    if not fake_scores and not real_scores:
        return "UNCERTAIN", 0.0, []

    mean_fake = np.mean(fake_scores) if fake_scores else 0.0
    mean_real = np.mean(real_scores) if real_scores else 0.0
    confidence = max(mean_fake, mean_real)

    if confidence < 0.6:
        label = "UNCERTAIN"
    elif mean_fake > mean_real:
        label = "FAKE"
    else:
        label = "REAL"

    return label, confidence, timeline

# ================= HISTORY =================
def save_history(username, filepath, prediction, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
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
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()

def load_history(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, filepath, prediction, confidence, timestamp
        FROM history
        WHERE username=?
        ORDER BY id DESC
    """, (username,))
    rows = c.fetchall()
    conn.close()
    return rows

def delete_history_item(history_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT filepath FROM history WHERE id=?", (history_id,))
    row = c.fetchone()

    if row:
        path = row[0]
        if os.path.exists(path):
            os.remove(path)
        c.execute("DELETE FROM history WHERE id=?", (history_id,))
        conn.commit()

    conn.close()

def delete_all_history(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT filepath FROM history WHERE username=?", (username,))
    rows = c.fetchall()

    for (path,) in rows:
        if os.path.exists(path):
            os.remove(path)

    c.execute("DELETE FROM history WHERE username=?", (username,))
    conn.commit()
    conn.close()

# ================= UI STYLING =================
st.set_page_config("Deepfake Detector", "🕵️", layout="wide")

st.markdown("""
<style>
* {
    transition: background-color .2s ease,
                color .2s ease,
                box-shadow .18s ease,
                transform .15s ease;
}

html[data-theme="light"] {
    --card-bg: #ffffff;
    --card-border: #e5e7eb;
    --shadow-soft: rgba(15, 23, 42, 0.08);
    --shadow-hard: rgba(15, 23, 42, 0.15);
    --highlight: rgba(255,255,255,0.9);
}

html[data-theme="dark"] {
    --card-bg: #0f172a;
    --card-border: #1e293b;
    --shadow-soft: rgba(0, 0, 0, 0.35);
    --shadow-hard: rgba(0, 0, 0, 0.55);
    --highlight: rgba(255,255,255,0.04);
}

.card {
    background: linear-gradient(180deg,
        var(--highlight),
        var(--card-bg)
    );
    border-radius: 18px;
    padding: 1.6rem;
    border: 1px solid var(--card-border);
    box-shadow:
        0 2px 6px var(--shadow-soft),
        0 10px 26px var(--shadow-hard);
}

.card:hover {
    transform: translateY(-2px);
    box-shadow:
        0 6px 14px var(--shadow-soft),
        0 18px 42px var(--shadow-hard);
}

button[kind="primary"], button[kind="secondary"] {
    position: relative;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.35),
        0 2px 6px rgba(0,0,0,.18),
        0 6px 18px rgba(0,0,0,.12);
}

button[kind="primary"]:hover,
button[kind="secondary"]:hover {
    transform: translateY(-1px);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.45),
        0 4px 10px rgba(0,0,0,.22),
        0 10px 26px rgba(0,0,0,.18);
}

button[kind="primary"]:active,
button[kind="secondary"]:active {
    transform: translateY(1px);
    box-shadow:
        inset 0 2px 6px rgba(0,0,0,.25),
        0 2px 4px rgba(0,0,0,.15);
}

section[data-testid="stSidebar"] button:hover {
    transform: translateX(2px);
}

.fade-in {
    animation: fadeIn .45s ease forwards;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ================= SESSION =================
if "user" not in st.session_state:
    st.session_state.user = None
if "selected_id" not in st.session_state:
    st.session_state.selected_id = None
if "processed_hashes" not in st.session_state:
    st.session_state.processed_hashes = set()

# ================= LOGIN =================
if st.session_state.user is None:
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
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

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ================= MAIN APP =================
username = st.session_state.user
user_dir = os.path.join(UPLOAD_DIR, username)
os.makedirs(user_dir, exist_ok=True)

# -------- SIDEBAR --------
st.sidebar.title("📁 History")
history = load_history(username)

for hid, path, pred, conf, ts in history:
    name = os.path.basename(path)
    cols = st.sidebar.columns([4, 1])

    with cols[0]:
        if st.button(f"{name} · {ts.split()[0]}", key=f"select_{hid}"):
            st.session_state.selected_id = hid

    with cols[1]:
        if st.button("🗑️", key=f"del_{hid}"):
            delete_history_item(hid)
            st.session_state.selected_id = None
            st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Delete ALL history"):
    delete_all_history(username)
    st.session_state.selected_id = None
    st.rerun()

# -------- HEADER --------
st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
st.title("🕵️ Deepfake Detection Assistant")
st.caption("AI-powered forensic analysis for images and videos")
st.markdown("</div>", unsafe_allow_html=True)

# -------- UPLOAD CARD --------
st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📤 Upload image or video",
    ["jpg", "jpeg", "png", "mp4", "mov", "avi"]
)
st.markdown("</div>", unsafe_allow_html=True)

# -------- PROCESS --------
if uploaded_file:
    data = uploaded_file.getvalue()
    h = file_hash(data)

    if h not in st.session_state.processed_hashes:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
        filepath = os.path.join(user_dir, filename)

        with open(filepath, "wb") as f:
            f.write(data)

        ext = uploaded_file.name.lower().split(".")[-1]

        with st.spinner("🔍 Analyzing media..."):
            if ext in ["jpg", "jpeg", "png"]:
                image = Image.open(filepath).convert("RGB")
                label, confidence, _ = classify_image(image)
            else:
                label, confidence, _ = classify_video(filepath)

        save_history(username, filepath, label, confidence)
        st.session_state.processed_hashes.add(h)
        st.session_state.selected_id = None
        st.rerun()

# -------- DISPLAY --------
if st.session_state.selected_id:
    for hid, path, pred, conf, ts in history:
        if hid == st.session_state.selected_id:
            st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)

            if os.path.exists(path):
                if path.lower().endswith((".mp4", ".mov", ".avi")):
                    st.video(path)
                else:
                    st.image(path, use_column_width=True)
            else:
                st.warning("File not found")

            if pred == "REAL":
                st.success("🟢 REAL")
            elif pred == "FAKE":
                st.error("🔴 FAKE")
            else:
                st.warning("🟡 UNCERTAIN")

            st.write(f"Confidence: **{conf:.2%}**")
            st.caption(f"🕒 {ts}")

            if path.lower().endswith((".mp4", ".mov", ".avi")):
                _, _, timeline = classify_video(path)
                if timeline:
                    st.markdown("### 🎞 Frame-Level Fake Probability")
                    fig = plt.figure()
                    plt.plot(timeline)
                    plt.ylim(0, 1)
                    plt.xlabel("Sampled Frame")
                    plt.ylabel("Fake Probability")
                    plt.grid(alpha=0.3)
                    st.pyplot(fig)

            st.markdown("</div>", unsafe_allow_html=True)
            break
