# -----------------------------
# SYSTEM & PATH CONFIGURATION
# -----------------------------
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# REQUIRED LIBRARIES
# -----------------------------
import streamlit as st                          # Streamlit UI framework
from PIL import Image                           # Image handling
from tensorflow.keras.models import load_model  # Load trained ML models
import streamlit.components.v1 as components    # Embed external HTML/JS
from reportlab.lib.pagesizes import A4           # PDF page size
from reportlab.pdfgen import canvas             # PDF generator
from reportlab.lib.utils import ImageReader     # Correct import for reportlab
import io                                       # In-memory file handling

# -----------------------------
# PROJECT MODULE IMPORTS
# -----------------------------
from config.class_names import RICE_CLASSES, PULSES_CLASSES
from services.rice_predictor import RicePredictor
from services.pulses_predictor import PulsesPredictor
from services.chatbot import CropChatbot

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Crop Disease Detection System",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS STYLING
# -----------------------------
st.markdown("""
<style>
/* MAIN APP BACKGROUND */
.stApp { background-color: #f6fff8; }

/* SIDEBAR BACKGROUND */
section[data-testid="stSidebar"] { background-color: #5dade2; }

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: 600;
}

/* BUTTON STYLING */
.stButton > button {
    background: linear-gradient(90deg, #5dade2, #2e86c1);
    color: white;
    border-radius: 10px;
    font-weight: bold;
    padding: 0.5em 1em;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# PDF GENERATION FUNCTION
# -----------------------------
def generate_pdf(crop_type, image_file, result):
    """
    Generates a PDF report with:
    - Title
    - Leaf Image
    - Prediction Summary
    - Justification
    - Recommended Actions
    """
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # --- Title ---
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(width / 2, height - 50, "Crop Disease Detection Report")

    # --- Leaf Image ---
    img = Image.open(image_file)
    img_width, img_height = img.size

    max_width = width - 100
    max_height = 250
    ratio = min(max_width / img_width, max_height / img_height)
    img_width_new = img_width * ratio
    img_height_new = img_height * ratio

    # Space between title and image
    image_y = height - 120 - img_height_new
    pdf.drawImage(ImageReader(img), (width - img_width_new) / 2, image_y, width=img_width_new, height=img_height_new)

    # --- Prediction Summary ---
    y = image_y - 20  # move text closer to image
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Prediction Summary:")
    pdf.setFont("Helvetica", 11)
    y -= 18
    pdf.drawString(50, y, f"Crop Type: {crop_type}")
    y -= 18
    pdf.drawString(50, y, f"Detected Disease: {result}")

    # --- Justification ---
    justification = "The leaf surface appears uniform and green with no visible disease symptoms."
    y -= 30  # reduce space before Justification
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Justification:")
    pdf.setFont("Helvetica", 11)
    y -= 18
    pdf.drawString(50, y, justification)

    # --- Recommended Actions ---
    recommendations = [
        "Remove infected leaves",
        "Avoid excess watering",
        "Apply recommended fungicide"
    ]
    y -= 30  # reduce space before Recommended Actions
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Recommended Actions:")
    pdf.setFont("Helvetica", 11)
    y -= 18
    for action in recommendations:
        pdf.drawString(50, y, f"- {action}")
        y -= 18

    pdf.save()
    buffer.seek(0)
    return buffer

# -----------------------------
# LOAD CHATBOT (RULE-BASED)
# -----------------------------
chatbot = CropChatbot()

# -----------------------------
# LOAD ML MODELS (CACHED)
# -----------------------------
@st.cache_resource
def load_models():
    rice_model = load_model("models/rice_model.h5")
    pulses_model = load_model("models/pulses_disease_model.h5")
    rice_predictor = RicePredictor(rice_model, RICE_CLASSES)
    pulses_predictor = PulsesPredictor(pulses_model, PULSES_CLASSES)
    return rice_predictor, pulses_predictor

rice_predictor, pulses_predictor = load_models()

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("🌾 Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Home", "Disease Prediction", "Chatbot", "Chatbot Botpress", "About"]
)

# -----------------------------
# HOME PAGE
# -----------------------------
if menu == "Home":
    st.title("🌾 Crop Disease Detection System")
    st.markdown("""
    ### 🚀 Features
    - Upload crop leaf images  
    - Detect diseases using AI  
    - Supports Rice and Pulses  
    - Download prediction report (PDF)  
    - Chatbot assistance  
    """)

# -----------------------------
# DISEASE PREDICTION PAGE
# -----------------------------
elif menu == "Disease Prediction":
    st.header("🔬 Disease Prediction")

    # Image upload
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

    # Crop selection
    crop_type = st.selectbox("Select Crop Type", ["Rice", "Pulses"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Prediction
        if st.button("Predict Disease"):
            if crop_type == "Rice":
                result = rice_predictor.predict(image)
            else:
                result = pulses_predictor.predict(image)

            st.success(f"🌱 Predicted Disease: **{result}**")

            # Generate PDF
            pdf_file = generate_pdf(crop_type, uploaded_file, result)

            # PDF Download Button
            st.download_button(
                label="📄 Download Prediction Report (PDF)",
                data=pdf_file,
                file_name="crop_disease_report.pdf",
                mime="application/pdf"
            )

# -----------------------------
# INTERNAL CHATBOT PAGE
# -----------------------------
elif menu == "Chatbot":
    st.header("🤖 Crop Assistant Chatbot")
    user_question = st.text_input("Ask about crop diseases, symptoms, or prevention:")
    if user_question:
        response = chatbot.get_response(user_question)
        st.success(response)

# -----------------------------
# BOTPRESS CHATBOT PAGE
# -----------------------------
elif menu == "Chatbot Botpress":
    st.header("🤖 Crop Assistant Chatbot (Botpress)")
    st.write("Ask about crop diseases, symptoms, prevention, and treatment.")
    components.html("""
        <script src="https://cdn.botpress.cloud/webchat/v3.5/inject.js"></script>
        <script src="https://files.bpcontent.cloud/2026/01/08/02/20260108021422-EKVCT9LA.js"></script>
        <style>
            iframe { width: 100% !important; height: 100vh !important; border: none; }
        </style>
    """, height=900)

# -----------------------------
# ABOUT PAGE
# -----------------------------
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
    - Botpress  

    **Features:**  
    - AI-based disease detection  
    - PDF report generation  
    - Chatbot support  
    """)
