import sys
import os

# Fix import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

from config.class_names import RICE_CLASSES, PULSES_CLASSES
from services.rice_predictor import RicePredictor
from services.pulses_predictor import PulsesPredictor
from services.chatbot import CropChatbot

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Crop Disease Detection System",
    page_icon="🌾",
    layout="wide"
)
st.markdown("""
<style>

/* MAIN APP BACKGROUND */
.stApp {
    background-color: #f6fff8;
}

/* SIDEBAR BACKGROUND */
section[data-testid="stSidebar"] {
    background-color: #5dade2;  /* sky blue */
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
    font-weight: 600;
}

/* RADIO BUTTON (NAVIGATION) */
div[role="radiogroup"] label {
    color: #ffffff !important;
    font-size: 15px;
}

/* PAGE TITLES */
h1 {
    color: #1b4f72;
}

h2, h3 {
    color: #2e86c1;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(90deg, #5dade2, #2e86c1);
    color: white;
    border-radius: 12px;
    padding: 0.6em 1.2em;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #2e86c1, #1b4f72);
}

</style>
""", unsafe_allow_html=True)




# --------------------------------------------------
# LOAD CHATBOT
# --------------------------------------------------
chatbot = CropChatbot()

# --------------------------------------------------
# CACHE MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():
    rice_model = load_model("models/rice_model.h5")
    pulses_model = load_model("models/pulses_disease_model.h5")

    rice_predictor = RicePredictor(rice_model, RICE_CLASSES)
    pulses_predictor = PulsesPredictor(pulses_model, PULSES_CLASSES)

    return rice_predictor, pulses_predictor

rice_predictor, pulses_predictor = load_models()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("🌾 Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Home", "Disease Prediction", "Chatbot", "About"]
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if menu == "Home":
    st.markdown(
        """
        <h1 style='text-align:center;'>🌾 Crop Disease Detection System</h1>
        <p style='text-align:center; font-size:18px;'>
        AI-powered system for detecting Rice and Pulses crop diseases
        </p>
        """,
        unsafe_allow_html=True
    )

    st.write("### 🚀 Features")
    st.markdown("""
    - Upload crop leaf images  
    - Detect diseases using CNN models  
    - Supports **Rice** and **Pulses**  
    - Integrated **Crop Assistant Chatbot**  
    """)

# --------------------------------------------------
# DISEASE PREDICTION PAGE
# --------------------------------------------------
elif menu == "Disease Prediction":
    st.header("🔬 Disease Prediction")

    uploaded_file = st.file_uploader(
        "Upload Leaf Image", type=["jpg", "jpeg", "png"]
    )

    crop_type = st.selectbox("Select Crop Type", ["Rice", "Pulses"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Disease"):
            if crop_type == "Rice":
                result = rice_predictor.predict(image)
            else:
                result = pulses_predictor.predict(image)

            st.success(f"🌱 Predicted Disease: **{result}**")

# --------------------------------------------------
# CHATBOT PAGE
# --------------------------------------------------
elif menu == "Chatbot":
    st.header("🤖 Crop Assistant Chatbot")

    user_question = st.text_input(
        "Ask about crop diseases, symptoms, or prevention:"
    )

    if user_question:
        response = chatbot.get_response(user_question)
        st.success(response)

# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif menu == "About":
    st.header("📌 About Project")

    st.markdown("""
    **Project Title:** Crop Disease Detection System  

    **Crops Supported:**  
    - Rice  
    - Pulses  

    **Technologies Used:**  
    - Python  
    - TensorFlow & Keras  
    - Streamlit  

    **Features:**  
    - CNN-based disease detection  
    - Separate models for rice & pulses  
    - Chatbot assistance  
    - SOLID principles based design  
    """)
