import streamlit as st
import torch
import sqlite3
import bcrypt
import os
import cv2
import hashlib
import numpy as np
from datetime import datetime
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ================= CONFIG =================
MODEL_ID = "prithivMLmods/deepfake-detector-model-v1"
DB_PATH = "deepfake.db"
UPLOAD_DIR = "uploads"
MAX_FRAMES = 60
MIN_LAPLACIAN = 80.0
UNCERTAINTY_THRESHOLD = 0.52
TEMPERATURE = 0.5

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
        c.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed)
        )
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
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    model.eval()
    return processor, model

# ================= HELPERS =================
def file_hash(data: bytes):
    return hashlib.md5(data).hexdigest()

def frame_sharpness(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def frame_brightness(gray: np.ndarray) -> float:
    return gray.mean()

def apply_temperature(logits: torch.Tensor, temperature: float = TEMPERATURE):
    return torch.softmax(logits / temperature, dim=1)

def sample_frame_indices(total_frames: int, n: int):
    if total_frames <= n:
        return list(range(total_frames))
    step = total_frames / n
    return [int(i * step) for i in range(n)]

# ================= IMAGE CLASSIFICATION =================
def classify_image_tta(image: Image.Image):
    processor, model = load_model()

    variants = [
        image,
        ImageOps.mirror(image),
        image.crop((10, 10, image.width - 10, image.height - 10)).resize(image.size),
        image.crop((0, 0, image.width, image.height * 9 // 10)).resize(image.size),
        image.crop((0, image.height // 10, image.width, image.height)).resize(image.size),
    ]

    all_logits = []
    for variant in variants:
        inputs = processor(images=variant, return_tensors="pt")
        with torch.no_grad():
            all_logits.append(model(**inputs).logits)

    avg_logits = torch.stack(all_logits).mean(dim=0)
    probs = apply_temperature(avg_logits)[0]

    fake, real = probs[0].item(), probs[1].item()
    confidence = max(fake, real)

    if confidence < UNCERTAINTY_THRESHOLD:
        label = "UNCERTAIN"
    elif fake > real:
        label = "FAKE"
    else:
        label = "REAL"

    return label, confidence, fake

def classify_image(image: Image.Image):
    label, confidence, _ = classify_image_tta(image)
    return label, confidence

# ================= VIDEO CLASSIFICATION =================
def classify_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        total_frames = MAX_FRAMES * 30

    target_indices = set(sample_frame_indices(total_frames, MAX_FRAMES))
    frame_results = []
    prev_gray = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in target_indices:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = frame_sharpness(gray)
            brightness = frame_brightness(gray)

            if sharpness >= MIN_LAPLACIAN and brightness > 10:
                if prev_gray is not None and cv2.absdiff(gray, prev_gray).mean() < 1.5:
                    frame_idx += 1
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, _, fake_prob = classify_image_tta(Image.fromarray(frame_rgb))
                frame_results.append((fake_prob, sharpness))
                prev_gray = gray

        frame_idx += 1

    cap.release()

    if not frame_results:
        return "UNCERTAIN", 0.0

    fake_arr = np.array([r[0] for r in frame_results])
    sharp_w = np.array([r[1] for r in frame_results])
    sharp_w = sharp_w / sharp_w.sum()

    wmean_fake = np.dot(sharp_w, fake_arr)
    top_n = max(1, len(fake_arr) // 4)
    top_fake = np.sort(fake_arr)[-top_n:].mean()

    blended_fake = 0.6 * wmean_fake + 0.4 * top_fake
    blended_real = 1.0 - blended_fake
    confidence = max(blended_fake, blended_real)

    if confidence < UNCERTAINTY_THRESHOLD:
        label = "UNCERTAIN"
    elif blended_fake > blended_real:
        label = "FAKE"
    else:
        label = "REAL"

    return label, float(confidence)

# ================= DB HELPERS =================
def save_history(username, filepath, prediction, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        c.execute("""
            INSERT INTO history (username, filepath, prediction, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?)
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
        FROM history WHERE username=? ORDER BY id DESC
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
        if os.path.exists(row[0]):
            os.remove(row[0])
        c.execute("DELETE FROM history WHERE id=?", (history_id,))
        conn.commit()

    conn.close()

def delete_all_history(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT filepath FROM history WHERE username=?", (username,))

    for (path,) in c.fetchall():
        if os.path.exists(path):
            os.remove(path)

    c.execute("DELETE FROM history WHERE username=?", (username,))
    conn.commit()
    conn.close()

# ================= SESSION =================
st.set_page_config("Deepfake Detector", "🕵️", layout="wide")

for key, default in [
    ("user", None),
    ("selected_id", None),
    ("processed_hashes", set())
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ================= LOGIN =================
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

IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]

st.sidebar.title("📁 History")

if st.sidebar.button("🚪 Logout"):
    st.session_state.update({"user": None, "selected_id": None, "processed_hashes": set()})
    st.rerun()

st.sidebar.markdown(f"👤 **{username}**")
st.sidebar.markdown("---")

history = load_history(username)

if not history:
    st.sidebar.caption("No history yet.")

for hid, path, pred, conf, ts in history:
    name = os.path.basename(path)
    emoji = "🟢" if pred == "REAL" else ("🔴" if pred == "FAKE" else "🟡")
    ext = name.rsplit(".", 1)[-1].lower()
    icon = "🎬" if ext in VIDEO_TYPES else "🖼️"

    cols = st.sidebar.columns([4, 1])

    if cols[0].button(f"{icon}{emoji} {name[:16]} · {ts.split()[0]}", key=f"sel_{hid}"):
        st.session_state.selected_id = hid

    if cols[1].button("🗑️", key=f"del_{hid}"):
        delete_history_item(hid)
        st.session_state.selected_id = None
        st.rerun()

st.sidebar.markdown("---")

if history and st.sidebar.button("🗑️ Delete ALL"):
    delete_all_history(username)
    st.session_state.selected_id = None
    st.rerun()

st.title("🕵️ Deepfake Detection Assistant")
st.caption("Upload an image or video to check if it's real or AI-generated/manipulated.")

uploaded_file = st.file_uploader(
    "Upload image or video",
    IMAGE_TYPES + VIDEO_TYPES
)

if uploaded_file:
    data = uploaded_file.getvalue()
    h = file_hash(data)

    if h not in st.session_state.processed_hashes:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
        filepath = os.path.join(user_dir, filename)

        with open(filepath, "wb") as f:
            f.write(data)

        ext = uploaded_file.name.lower().rsplit(".", 1)[-1]

        with st.spinner("🔍 Analyzing... this may take a moment for videos."):
            if ext in IMAGE_TYPES:
                image = Image.open(filepath).convert("RGB")
                label, confidence = classify_image(image)
            else:
                label, confidence = classify_video(filepath)

        save_history(username, filepath, label, confidence)

        st.session_state.processed_hashes.add(h)
        st.session_state.selected_id = None
        st.rerun()

if st.session_state.selected_id:
    for hid, path, pred, conf, ts in history:
        if hid != st.session_state.selected_id:
            continue

        col1, col2 = st.columns([1, 1])

        with col1:
            if not os.path.exists(path):
                st.warning("⚠️ File not found on disk.")
            else:
                ext = path.lower().rsplit(".", 1)[-1]
                if ext in VIDEO_TYPES:
                    st.video(path)
                else:
                    st.image(Image.open(path), use_container_width=True)

        with col2:
            st.markdown("### Detection Result")

            conf = float(conf) if conf else 0.0

            if pred == "REAL":
                st.success(f"🟢 REAL — {conf:.2%} confidence")
            elif pred == "FAKE":
                st.error(f"🔴 FAKE — {conf:.2%} confidence")
            else:
                st.warning(f"🟡 UNCERTAIN — {conf:.2%} confidence")

            st.progress(min(conf, 1.0))
            st.caption(f"🕐 Analyzed at: {ts}")
            st.caption(f"📄 File: {os.path.basename(path)}")

        break

elif not uploaded_file:
    st.info("👆 Upload a file above, or select a past result from the sidebar.")
