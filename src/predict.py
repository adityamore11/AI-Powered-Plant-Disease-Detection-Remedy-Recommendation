"""Reusable local-model prediction and optional Grad-CAM functions."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from config import CLASS_NAMES_PATH, CONFIDENCE_THRESHOLD, MODEL_PATH
from src.preprocessing import prepare_image
from src.utils import friendly_label, load_class_names

def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists(): raise FileNotFoundError("Trained model not found. Run `python src/train.py` first.")
    try:
        import tensorflow as tf
    except ImportError as exc: raise RuntimeError("TensorFlow is not installed. Install requirements.txt first.") from exc
    return tf.keras.models.load_model(model_path)

def preprocess_image(image: Image.Image | str | Path) -> np.ndarray: return prepare_image(image)

def get_top_predictions(probabilities: np.ndarray, class_names: list[str], top_k: int = 3) -> list[dict]:
    scores = np.asarray(probabilities).reshape(-1)
    if len(scores) != len(class_names): raise ValueError("Model output does not match the saved class mapping.")
    indexes = np.argsort(scores)[::-1][:min(top_k, len(scores))]
    return [{"class_name": class_names[i], "plant": friendly_label(class_names[i])[0], "disease": friendly_label(class_names[i])[1], "confidence": float(scores[i])} for i in indexes]

def predict_disease(image: Image.Image | str | Path, model=None, class_names: list[str] | None = None, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    model = model or load_model()
    class_names = class_names or load_class_names(CLASS_NAMES_PATH)
    probabilities = np.asarray(model.predict(preprocess_image(image), verbose=0))[0]
    top = get_top_predictions(probabilities, class_names)
    best = top[0]
    return {**best, "confidence_threshold": threshold, "is_confident": best["confidence"] >= threshold, "top_predictions": top}

def generate_gradcam(model, image: Image.Image | str | Path, class_index: int | None = None) -> np.ndarray:
    """Return a 0-1 heatmap, or raise a clear error for unsupported architectures."""
    import tensorflow as tf
    batch = preprocess_image(image)

    # The app's EfficientNet is nested inside the outer model. Calling a model
    # made from the nested layer's symbolic output can lose the outer input
    # structure in newer Keras versions, so expose the inner activation and
    # backbone output, then run the outer head explicitly.
    backbone_index = next((i for i, layer in enumerate(model.layers)
                           if isinstance(layer, tf.keras.Model)
                           and len(layer.output.shape) == 4
                           and any(isinstance(inner, tf.keras.layers.Conv2D) for inner in layer.layers)), None)
    if backbone_index is not None:
        backbone = model.layers[backbone_index]
        last_conv = next((layer for layer in reversed(backbone.layers)
                          if len(getattr(layer, "output", tf.constant(0)).shape) == 4), None)
        if last_conv is None:
            raise ValueError("Grad-CAM is unavailable: no convolutional layer was found in the feature extractor.")
        # Use the singular input tensor here. In Keras 3, passing the nested
        # model's one-item `inputs` list can fail to match its graph tensor IDs.
        feature_model = tf.keras.Model(backbone.input, [last_conv.output, backbone.output])
        with tf.GradientTape() as tape:
            x = tf.convert_to_tensor(batch)
            for layer in model.layers[1:backbone_index]:
                x = layer(x, training=False)
            conv, x = feature_model(x, training=False)
            for layer in model.layers[backbone_index + 1:]:
                x = layer(x, training=False)
            preds = x
            index = class_index if class_index is not None else tf.argmax(preds[0])
            score = preds[:, index]
    else:
        last_conv = next((layer for layer in reversed(model.layers)
                          if len(getattr(layer, "output", tf.constant(0)).shape) == 4), None)
        if last_conv is None: raise ValueError("Grad-CAM is unavailable: no convolutional layer was found.")
        grad_model = tf.keras.Model(model.inputs, [last_conv.output, model.output])
        with tf.GradientTape() as tape:
            conv, preds = grad_model(tf.convert_to_tensor(batch), training=False)
            index = class_index if class_index is not None else tf.argmax(preds[0])
            score = preds[:, index]
    grads = tape.gradient(score, conv); weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv[0] * weights, axis=-1); heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    return (heatmap / (maximum + tf.keras.backend.epsilon())).numpy()

def create_gradcam_overlay(image: Image.Image | str | Path, heatmap: np.ndarray) -> Image.Image:
    """Blend an OpenCV colour heatmap with the uploaded leaf image."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed. Install requirements.txt first.") from exc
    original = np.asarray(prepare_image(image)[0], dtype=np.uint8)
    resized = cv2.resize(heatmap.astype(np.float32), (original.shape[1], original.shape[0]))
    colored_bgr = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(original, 0.60, colored_rgb, 0.40, 0))
