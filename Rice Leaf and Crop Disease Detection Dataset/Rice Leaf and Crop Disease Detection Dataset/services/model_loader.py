import tensorflow as tf           #DIP(DEPENDENCY INVERSION PRINCIPLE)

class ModelLoader:
    def load(self, model_path):
        return tf.keras.models.load_model(model_path)
