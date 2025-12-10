import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = tf.keras.models.load_model("rice_disease_model.h5")

class_names = ['Bacterial Leaf Blight', 'Healthy_leaf', 'Rice', 'Rice Blast', 'Tungro']

test_folder = "test_images"

for img_file in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_file)
    
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    
    pred = model.predict(x)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx] * 100
    
    print(f"Image: {img_file}")
    print(f"Predicted Class: {class_names[class_idx]}")
    print(f"Confidence: {confidence:.2f}%\n")
