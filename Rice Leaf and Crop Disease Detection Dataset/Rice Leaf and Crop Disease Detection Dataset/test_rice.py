import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ---------------- PATHS ----------------
model_path = os.path.join('models', 'rice_model.h5')
test_dir = os.path.join('test_images', 'rice')

# ⚠️ IMPORTANT FIX HERE
class_dir = os.path.join('datasets', 'rice', 'train_data')

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(model_path)

# ---------------- CLASS NAMES ----------------
class_names = sorted([
    d for d in os.listdir(class_dir)
    if os.path.isdir(os.path.join(class_dir, d))
])

print("\n🔍 Rice Prediction Started...")
print("📌 Classes:", class_names, "\n")

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx] * 100

    print(f"📸 Image: {os.path.basename(img_path)}")
    print(f"🧪 Predicted Disease: {class_names[class_idx]}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print("-" * 40)

# ---------------- RUN ----------------
for img_file in os.listdir(test_dir):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        predict_image(os.path.join(test_dir, img_file))
