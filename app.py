import streamlit as st
import torch
import sqlite3
import bcrypt
import os
import hashlib
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
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
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

    fake, real = probs[0].item(), probs[1].item()
    confidence = max(fake, real)

    if confidence < 0.6:
        label = "UNCERTAIN"
    elif fake > real:
        label = "FAKE"
    else:
        label = "REAL"

    return label, confidence

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
        pass  # duplicate prevented
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
        filepath = row[0]
        if os.path.exists(filepath):
            os.remove(filepath)

        c.execute("DELETE FROM history WHERE id=?", (history_id,))
        conn.commit()

    conn.close()

def delete_all_history(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT filepath FROM history WHERE username=?", (username,))
    rows = c.fetchall()

    for (filepath,) in rows:
        if os.path.exists(filepath):
            os.remove(filepath)

    c.execute("DELETE FROM history WHERE username=?", (username,))
    conn.commit()
    conn.close()


# ================= SESSION =================
st.set_page_config("Deepfake Detector", "🕵️", layout="wide")

if "user" not in st.session_state:
    st.session_state.user = None
if "selected_id" not in st.session_state:
    st.session_state.selected_id = None
if "processed_hashes" not in st.session_state:
    st.session_state.processed_hashes = set()

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

# -------- SIDEBAR (HISTORY) --------
st.sidebar.title("📁 History")

history = load_history(username)

for hid, path, pred, conf, ts in history:
    name = os.path.basename(path)
    cols = st.sidebar.columns([4, 1])

    if cols[0].button(f"{name} · {ts.split()[0]}", key=f"select_{hid}"):
        st.session_state.selected_id = hid

    if cols[1].button("🗑️", key=f"del_{hid}"):
        delete_history_item(hid)
        st.session_state.selected_id = None
        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Delete ALL history"):
    delete_all_history(username)
    st.session_state.selected_id = None
    st.rerun()


# -------- MAIN VIEW --------
st.title("🕵️ Deepfake Detection Assistant")

uploaded_file = st.file_uploader("Upload an image", ["jpg", "jpeg", "png"])

if uploaded_file is not None:
    data = uploaded_file.getvalue()
    h = file_hash(data)

    if h not in st.session_state.processed_hashes:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
        filepath = os.path.join(user_dir, filename)

        with open(filepath, "wb") as f:
            f.write(data)

        image = Image.open(filepath).convert("RGB")
        label, confidence = classify_image(image)

        save_history(username, filepath, label, confidence)
        st.session_state.processed_hashes.add(h)
        st.session_state.selected_id = None

        st.rerun()

# -------- DISPLAY SELECTED ITEM --------
if st.session_state.selected_id:
    for hid, path, pred, conf, ts in history:
        if hid == st.session_state.selected_id:
            if os.path.exists(path):
                st.image(path, use_column_width=True)
            else:
                st.warning("Image file not found")

            if pred == "REAL":
                st.success("🟢 REAL")
            elif pred == "FAKE":
                st.error("🔴 FAKE")
            else:
                st.warning("🟡 UNCERTAIN")

            st.write(f"Confidence: **{conf:.2%}**")
            st.caption(ts)
            break
