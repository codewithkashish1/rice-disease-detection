import numpy as np       #OCD(OPEN CLOSED PRINCIPLE)
from interfaces.predictor_interface import PredictorInterface

class PulsesPredictor(PredictorInterface):

    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names

    def predict(self, image_array):
        preds = self.model.predict(image_array)
        return self.class_names[np.argmax(preds)]
