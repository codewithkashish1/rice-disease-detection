from tensorflow.keras.preprocessing import image  #SRP(SINGLE RESPONSIBILITY PRINCIPLE)
import numpy as np

class ImageLoader:
    def load(self, img_path, size=(224, 224)):
        img = image.load_img(img_path, target_size=size)
        arr = image.img_to_array(img) / 255.0
        return np.expand_dims(arr, axis=0)
