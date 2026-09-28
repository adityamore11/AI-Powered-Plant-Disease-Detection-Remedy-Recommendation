"""Consistent, inference-safe image preparation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from config import IMAGE_SIZE

def open_image(image: Image.Image | str | Path) -> Image.Image:
    try:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened: return opened.convert("RGB").copy()
        if not isinstance(image, Image.Image): raise TypeError("Expected a PIL image or image path.")
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid readable image.") from exc

def prepare_image(image: Image.Image | str | Path, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Return a float32 RGB batch. EfficientNet preprocessing is inside the model."""
    rgb = open_image(image).resize(image_size, Image.Resampling.LANCZOS)
    array = np.asarray(rgb, dtype=np.float32)
    return np.expand_dims(array, axis=0)
