from abc import ABC, abstractmethod    #ISP+LSP(COMMON INTERFACE)

class PredictorInterface(ABC):

    @abstractmethod
    def predict(self, image_array):
        pass
