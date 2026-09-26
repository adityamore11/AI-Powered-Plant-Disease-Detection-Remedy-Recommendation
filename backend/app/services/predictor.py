import json
from functools import lru_cache
import numpy as np
from PIL import Image
from ..config import get_settings
from .remedy_service import parse_label

class ModelUnavailableError(RuntimeError): pass

@lru_cache
def get_predictor():
    return Predictor()

class Predictor:
    def __init__(self):
        self.settings = get_settings()
        model_path = self.settings.resolve(self.settings.model_path)
        names_path = self.settings.resolve(self.settings.class_names_path)
        if not model_path.exists() or not names_path.exists():
            raise ModelUnavailableError("No trained model is available. Run training/train.py first.")
        import tensorflow as tf
        self.tf = tf
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = json.loads(names_path.read_text(encoding="utf-8"))

    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        size = self.settings.image_size
        array = np.asarray(image.resize((size, size)), dtype=np.float32)
        probabilities = self.model.predict(np.expand_dims(array, 0), verbose=0)[0]
        indices = np.argsort(probabilities)[::-1][:top_k]
        output = []
        for index in indices:
            label = self.class_names[int(index)]
            plant, disease = parse_label(label)
            output.append({"label": label, "plant": plant, "disease": disease, "confidence": float(probabilities[index])})
        return output
