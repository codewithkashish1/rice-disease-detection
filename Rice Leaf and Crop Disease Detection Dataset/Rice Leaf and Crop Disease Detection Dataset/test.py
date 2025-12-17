import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model('rice_disease_model.h5')

# Path to test images
test_dir = os.path.join(os.getcwd(), 'test_images')

# Get class labels from training folder names
class_labels = sorted(os.listdir(os.path.join(os.getcwd(), 'train_data')))

# Function to predict a single image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index] * 100
    predicted_class = class_labels[class_index]

    print(f"📸 Image: {os.path.basename(img_path)}")
    print(f"🧪 Predicted Disease: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print("----------------------------------------")

# Predict all images in test folder
for img_file in os.listdir(test_dir):
    if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        predict_image(os.path.join(test_dir, img_file))
