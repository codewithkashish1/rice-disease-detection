import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

class RicePredictor:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names

        self.img_height = model.input_shape[1]
        self.img_width = model.input_shape[2]

    def predict(self, pil_image):
        img = pil_image.resize((self.img_width, self.img_height))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = self.model.predict(img_array)
        return self.class_names[np.argmax(preds)]
