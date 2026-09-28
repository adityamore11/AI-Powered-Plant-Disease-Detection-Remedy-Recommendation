from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from src.preprocessing import prepare_image
from src.predict import create_gradcam_overlay, get_top_predictions
from src.remedy import get_disease_info
from src.utils import friendly_label, load_class_names, save_class_names

def test_prepares_rgb_image():
    batch = prepare_image(Image.new("L", (30, 20), 100))
    assert batch.shape == (1, 224, 224, 3) and batch.dtype == np.float32

def test_invalid_image_path_is_friendly(tmp_path):
    broken = tmp_path / "bad.jpg"; broken.write_text("not an image")
    with pytest.raises(ValueError): prepare_image(broken)

def test_class_mapping_round_trip(tmp_path):
    path = tmp_path / "classes.json"; save_class_names(["Tomato___Early_blight"], path)
    assert load_class_names(path) == ["Tomato___Early_blight"]

def test_label_and_prediction_structure():
    assert friendly_label("Tomato___Early_blight") == ("Tomato", "Early Blight")
    result = get_top_predictions(np.array([.1, .9]), ["Apple___healthy", "Tomato___Early_blight"])
    assert result[0]["class_name"] == "Tomato___Early_blight" and len(result) == 2

def test_missing_remedy_returns_safe_fallback(tmp_path):
    path = tmp_path / "empty.json"; path.write_text(json.dumps({}))
    info = get_disease_info("Apple___Apple_scab", path)
    assert "not available" in info["description"] and info["recommended_actions"]

def test_gradcam_overlay_uses_image_dimensions():
    pytest.importorskip("cv2")
    overlay = create_gradcam_overlay(Image.new("RGB", (32, 24)), np.ones((4, 4), dtype=np.float32))
    assert overlay.size == (224, 224)
