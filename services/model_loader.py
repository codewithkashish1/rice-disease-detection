from abc import ABC, abstractmethod

class ModelLoader(ABC):
    @abstractmethod
    def load(self, model_path):
        pass
