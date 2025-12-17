# app.py
import streamlit as st
import sqlite3
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import tempfile
import sys

# ---------------- Database Setup ----------------
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT)''')
    conn.commit()
    conn.close()

init_db()

def get_connection():
    return sqlite3.connect('users.db')

def login_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

def register_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

# ---------------- Load Model ----------------
try:
    model = tf.keras.models.load_model('rice_disease_model.h5')
    class_labels = sorted(os.listdir('train_data'))
except Exception as e:
    st.error(f"Error loading model or class labels: {e}")
    sys.exit()

# ---------------- Prediction Function ----------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index] * 100
    return class_labels[class_index], confidence

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="Rice Disease Detection", page_icon="🌾")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'predict_clicked' not in st.session_state:
    st.session_state.predict_clicked = False

# ----- Authentication -----
if not st.session_state.logged_in:
    st.title("🌾 Rice Disease Detection Login")
    menu = ["Login", "SignUp"]
    choice = st.selectbox("Select Action", menu)

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.success(f"Welcome {username}!")
                st.session_state.logged_in = True
                st.session_state.username = username
            else:
                st.error("Invalid username or password.")

    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type='password')
        if st.button("Sign Up"):
            if register_user(new_user, new_pass):
                st.success("Account created! Please login.")
            else:
                st.error("Username already exists.")

# ----- Upload & Prediction -----
else:
    st.sidebar.title(f"Welcome, {st.session_state.username}!")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.session_state.predict_clicked = False
        st.experimental_rerun()

    st.title("🌾 Rice Disease Detection")
    st.subheader("Upload Image for Prediction")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption='Uploaded Image', use_column_width=True)

        if st.button("Predict"):
           with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
              image_rgb = image_pil.convert("RGB")   # convert RGBA → RGB
              image_rgb.save(tmp_file.name)

              predicted_class, confidence = predict_image(tmp_file.name)

           st.success(f"Predicted Disease: **{predicted_class}**")
           st.info(f"Confidence: {confidence:.2f}%")

   
