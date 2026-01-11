# -----------------------------
# SYSTEM & PATH CONFIGURATION
# -----------------------------
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# REQUIRED LIBRARIES
# -----------------------------
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import streamlit.components.v1 as components
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import pandas as pd

# -----------------------------
# PROJECT MODULE IMPORTS
# -----------------------------
from config.class_names import RICE_CLASSES, PULSES_CLASSES
from services.rice_predictor import RicePredictor
from services.pulses_predictor import PulsesPredictor
from services.chatbot import CropChatbot
from services.medicine_service import MedicineService

# Load medicine dataset
medicine_df = pd.read_csv("datasets/medicine.csv", quotechar='"')

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Crop Disease Detection System",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #f6fff8; }
section[data-testid="stSidebar"] { background-color: #5dade2; }
section[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: 600;
}
.stButton > button {
    background: linear-gradient(90deg, #5dade2, #2e86c1);
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOGIN/SIGNUP FUNCTIONALITY
# -----------------------------
# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# In-memory user database
users_db = {
    "admin": "admin123"
}

def verify_password(username, password):
    return users_db.get(username) == password

def login(username, password):
    if verify_password(username, password):
        st.session_state.logged_in = True
        st.session_state.username = username
        st.success(f"Welcome, {username}!")
    else:
        st.error("Incorrect username or password")

def sign_up(new_user, new_pass):
    if new_user in users_db:
        st.warning("Username already exists")
    else:
        users_db[new_user] = new_pass
        st.success("Account created successfully! Please login.")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("Logged out successfully!")

# -----------------------------
# PDF GENERATION FUNCTION
# -----------------------------
def generate_pdf(crop_type, image_file, result, medicine):
    if medicine is None:
        medicine = {}

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(width / 2, height - 40, "Crop Disease Detection Report")

    img = Image.open(image_file)
    pdf.drawImage(ImageReader(img), 150, height - 330, width=300, height=200)

    y = height - 360

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Prediction Summary")
    y -= 20
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Crop Type: {crop_type}")
    y -= 18
    pdf.drawString(50, y, f"Detected Disease: {result}")

    y -= 30
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Justification")
    y -= 18
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, medicine.get("justification", "N/A"))

    y -= 30
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Recommendation")
    y -= 18
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, medicine.get("recommendation", "N/A"))

    y -= 30
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Recommended Medicine")
    y -= 18
    pdf.setFont("Helvetica", 11)

    if medicine:
        pdf.drawString(50, y, f"Medicine Name: {medicine.get('medicine_name', 'N/A')}")
        y -= 15
        pdf.drawString(50, y, f"Type: {medicine.get('medicine_type', 'N/A')}")
        y -= 15
        pdf.drawString(50, y, f"Dosage: {medicine.get('dosage', 'N/A')}")
        y -= 15
        pdf.drawString(50, y, f"Application Method: {medicine.get('application_method', 'N/A')}")
        y -= 15
        pdf.drawString(50, y, f"Frequency: {medicine.get('frequency', 'N/A')}")
        y -= 15
        pdf.drawString(50, y, f"Precautions: {medicine.get('precautions', 'N/A')}")
        y -= 15
        pdf.drawString(50, y, f"Source: {medicine.get('source', 'N/A')}")
    else:
        pdf.drawString(50, y, "No medicine recommendation available.")

    pdf.save()
    buffer.seek(0)
    return buffer

# -----------------------------
# INITIALIZE SERVICES
# -----------------------------
chatbot = CropChatbot()
medicine_service = MedicineService()

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    rice_model = load_model("models/rice_model.h5")
    pulses_model = load_model("models/pulses_disease_model.h5")
    return (
        RicePredictor(rice_model, RICE_CLASSES),
        PulsesPredictor(pulses_model, PULSES_CLASSES)
    )

rice_predictor, pulses_predictor = load_models()

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("🌾 Navigation")

if not st.session_state.logged_in:
    action = st.sidebar.radio("Choose Action", ["Login", "Sign Up"])

    if action == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            login(username, password)

    elif action == "Sign Up":
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            sign_up(new_user, new_pass)

else:
    st.sidebar.write(f"👋 Hello, {st.session_state.username}!")
    if st.sidebar.button("Logout"):
        logout()

    menu = st.sidebar.radio(
        "Go to",
        ["Home", "Disease Prediction", "Chatbot", "Chatbot Botpress", "About"]
    )

    # -----------------------------
    # HOME PAGE
    # -----------------------------
    if menu == "Home":
        st.title("🌾 Crop Disease Detection System")
        st.markdown("This system helps farmers detect crop diseases early using AI.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🔬 Disease Detection")
            st.markdown("Detect diseases in Rice and Pulses using leaf images.")
        with col2:
            st.markdown("### 💊 Medicine Recommendation")
            st.markdown("Get recommended treatment, dosage, precautions, and justification.")
        with col3:
            st.markdown("### 📄 PDF Reports")
            st.markdown("Download a detailed report for record-keeping and offline reference.")

        st.markdown("---")

        st.subheader("🤖 AI Crop Assistant")
        st.write(
            "Chat with an AI-powered assistant to get instant help on crop diseases, "
            "treatments, prevention tips, and general farming guidance."
        )

        col4, col5 = st.columns(2)
        with col4:
            st.info("🌱 Ask questions about crop health and symptoms")
        with col5:
            st.success("💬 Get AI-based suggestions and farming advice")

        left_img = os.path.join("assets", "left_image.png")
        right_img = os.path.join("assets", "image.avif")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            if os.path.exists(left_img):
                st.image(left_img, caption="Crop monitoring", use_column_width=True)
        with img_col2:
            if os.path.exists(right_img):
                st.image(right_img, caption="Crop leaf disease detection", use_column_width=True)

    # -----------------------------
    # DISEASE PREDICTION
    # -----------------------------
    elif menu == "Disease Prediction":
        st.header("🔬 Disease Prediction")
        uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
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

                if "healthy" in result.lower():
                    st.subheader("✅ Plant Health Status")
                    st.success("The plant is healthy. No medicine is required.")
                    st.info("Maintain proper irrigation, nutrition, and regular monitoring.")
                    pdf_file = generate_pdf(crop_type, uploaded_file, result, {})
                else:
                    medicine = medicine_service.get_medicine(crop_type, result)
                    if medicine:
                        st.subheader("💊 Recommended Treatment")
                        st.subheader("🧠 Disease Justification")
                        st.write(medicine.get("justification", "N/A"))
                        st.subheader("✅ Recommended Action")
                        st.write(medicine.get("recommendation", "N/A"))
                        st.markdown(f"""
                        **Medicine Name:** {medicine.get('medicine_name', 'N/A')}  
                        **Type:** {medicine.get('medicine_type', 'N/A')}  
                        **Dosage:** {medicine.get('dosage', 'N/A')}  
                        **Application Method:** {medicine.get('application_method', 'N/A')}  
                        **Frequency:** {medicine.get('frequency', 'N/A')}  
                        **Precautions:** {medicine.get('precautions', 'N/A')}  
                        **Source:** {medicine.get('source', 'N/A')}
                        """)
                    else:
                        st.warning("No medicine data found for this disease.")
                    pdf_file = generate_pdf(crop_type, uploaded_file, result, medicine)

                st.download_button("📄 Download Prediction Report (PDF)", pdf_file, "crop_disease_report.pdf", "application/pdf")

    # -----------------------------
    # CHATBOT
    # -----------------------------
    elif menu == "Chatbot":
        st.header("🤖 Crop Assistant Chatbot")
        user_question = st.text_input("Ask your question")
        if user_question:
            st.success(chatbot.get_response(user_question))

    # -----------------------------
    # BOTPRESS CHATBOT
    # -----------------------------
    elif menu == "Chatbot Botpress":
        st.header("🤖 Crop Assistant Chatbot (Botpress)")
        components.html("""
        <script src="https://cdn.botpress.cloud/webchat/v3.5/inject.js"></script>
        <script src="https://files.bpcontent.cloud/2026/01/08/02/20260108021422-EKVCT9LA.js"></script>
        """, height=900)

    # -----------------------------
    # ABOUT
    # -----------------------------
    elif menu == "About":
        st.markdown("""
# 🌾 Crop Disease Detection System (ABOUT)

The Crop Disease Detection System is an **AI-powered smart agriculture application** designed to help farmers and agricultural professionals detect crop diseases at an early stage using leaf images. Early detection of diseases allows timely intervention, preventing major crop losses and improving yield quality.

---

## 🔹 Key Features

### 🟢 AI-based Disease Detection
Uses deep learning image classification models to accurately identify diseases in **Rice and Pulses crops** by analyzing leaf images.

### 🟢 Medicine Recommendations
After detecting the disease, the system provides recommended treatments, **dosage, application method, precautions, and justification** based on reliable agricultural data sources.

### 🟢 PDF Report Generation
Generates downloadable PDF reports containing **prediction results, recommended actions, and medicine details** for record-keeping and offline reference.

### 🟢 Interactive AI Crop Assistant (Chatbot)
Integrated **AI-powered chatbot** to provide instant guidance on crop health, disease prevention, and farming best practices.

### 🟢 User-Friendly Interface
Designed using **Streamlit**, the platform provides a simple, intuitive, and responsive interface for users with minimal technical knowledge.

### 🟢 Multi-Crop Support
Currently supports multiple crops like **Rice and Pulses**, with potential to extend to other crops in the future.

### 🟢 Data-Driven Insights
Helps farmers make informed decisions using **AI predictions and curated agricultural data**.

---

## 🔹 Technical Overview

- **Frontend:** Streamlit  
- **Backend:** Python, Keras/TensorFlow for AI models  
- **Models:** Pre-trained Convolutional Neural Networks (CNN) for image classification  
- **Data Sources:** Agriculture databases and medicinal references  
- **PDF Generation:** ReportLab library for automated report creation  

---

## 🔹 Benefits

- **Reduce Crop Loss:** Early detection prevents spread of diseases.  
- **Save Time & Cost:** Reduces guesswork and unnecessary use of chemicals.  
- **Promote Smart Farming:** Encourages technology adoption in agriculture.  
- **Educational Value:** Helps farmers learn about disease prevention and treatment.  

---

## 🔹 Future Enhancements

- Add support for more crops like **Wheat, Maize, and Pepper**.  
- Integrate **real-time disease detection** using drone or mobile camera feeds.  
- Provide **weather and soil-based disease prediction analytics**.  
- Enhance **chatbot with voice input** for easier accessibility.  
""", unsafe_allow_html=True)
    # Display About image
        about_img = os.path.join("assets", "about.avif")
        if os.path.exists(about_img):
          st.image(about_img, caption="Crop Disease Detection System", width=700)
