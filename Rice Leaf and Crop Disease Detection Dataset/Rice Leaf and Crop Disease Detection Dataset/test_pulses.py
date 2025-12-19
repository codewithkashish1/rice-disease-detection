# test_pulses_multiple.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ----------------------------
# 1️⃣ Paths & Parameters
# ----------------------------
MODEL_PATH = 'models/pulses_disease_model.h5'
TEST_DIR = 'test_images/pulses'   # ✅ corrected path
IMG_HEIGHT, IMG_WIDTH = 224, 224

CLASS_NAMES = [
    'Anthracnose',
    'Bacterial_spot',
    'Healthy'
]

# ----------------------------
# 2️⃣ Load Model
# ----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Pulses model loaded")

# ----------------------------
# 3️⃣ Predict All Images in Folder
# ----------------------------
def predict_multiple_images(test_dir):
    for img_name in os.listdir(test_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(test_dir, img_name)

            img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array, verbose=0)
            predicted_class = CLASS_NAMES[np.argmax(preds)]
            confidence = np.max(preds) * 100

            print(f"\n📸 Image: {img_name}")
            print(f"🦠 Disease: {predicted_class}")
            print(f"🎯 Confidence: {confidence:.2f}%")

# ----------------------------
# 4️⃣ Run Prediction
# ----------------------------
if os.path.exists(TEST_DIR):
    predict_multiple_images(TEST_DIR)
else:
    print(f"❌ Folder not found: {TEST_DIR}")
