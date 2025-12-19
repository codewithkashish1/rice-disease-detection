from services.image_loader import ImageLoader
from services.model_loader import ModelLoader
from services.rice_predictor import RicePredictor
from services.pulses_predictor import PulsesPredictor
from config.class_names import RICE_CLASSES, PULSES_CLASSES

image_loader = ImageLoader()
model_loader = ModelLoader()

crop_type = "pulses"   # or "rice"
image_path = "test_images/pulses/img1.jpg"

if crop_type == "rice":
    model = model_loader.load("models/rice_model.h5")
    predictor = RicePredictor(model, RICE_CLASSES)

elif crop_type == "pulses":
    model = model_loader.load("models/pulses_model.h5")
    predictor = PulsesPredictor(model, PULSES_CLASSES)

image_array = image_loader.load(image_path)
result = predictor.predict(image_array)

print("Detected Disease:", result)
