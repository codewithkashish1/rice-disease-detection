from tensorflow.keras.models import load_model

class TensorFlowModelLoader:
    def load(self, model_path):
        return load_model(model_path)
