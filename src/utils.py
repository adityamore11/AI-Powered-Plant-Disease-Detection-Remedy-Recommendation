"""Shared filesystem, labels, and dataset discovery helpers."""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Iterable
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass

def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths: path.mkdir(parents=True, exist_ok=True)

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS

def friendly_label(class_name: str) -> tuple[str, str]:
    """Turn PlantVillage labels into a display plant and disease name."""
    parts = class_name.split("___", 1)
    plant = parts[0].replace("_", " ").strip().title()
    disease = (parts[1] if len(parts) == 2 else "Unknown").strip().replace("_", " ").title()
    return plant, disease

def find_split_dirs(dataset_dir: Path) -> dict[str, Path]:
    """Find common train/validation/test directory names, if supplied."""
    aliases = {"train": ("train", "training"), "val": ("val", "valid", "validation"), "test": ("test", "testing")}
    found = {}
    for split, names in aliases.items():
        for name in names:
            candidate = dataset_dir / name
            if candidate.is_dir(): found[split] = candidate; break
    return found

def find_class_root(dataset_dir: Path) -> Path:
    """Find class folders inside flat, split, or Kaggle wrapper structures."""
    split_dirs = find_split_dirs(dataset_dir)
    source = split_dirs.get("train", dataset_dir)
    candidates = [source, *source.rglob("*")] if source.exists() else []
    candidates.sort(key=lambda p: (p.name != "color", len(p.parts)))
    for candidate in candidates:
        if candidate.is_dir() and sum(1 for p in candidate.iterdir() if p.is_dir() and "___" in p.name) >= 2:
            return candidate
    return source

def discover_classes(dataset_dir: Path) -> list[str]:
    """Discover class directories from either a flat or split PlantVillage layout."""
    source = find_class_root(dataset_dir)
    if not source.exists(): return []
    return sorted(p.name for p in source.iterdir() if p.is_dir() and not p.name.startswith("."))

def image_counts(dataset_dir: Path, class_names: list[str] | None = None) -> dict[str, int]:
    source = find_class_root(dataset_dir)
    classes = class_names or discover_classes(dataset_dir)
    return {name: sum(1 for p in (source / name).rglob("*") if is_image_file(p)) for name in classes}

def save_class_names(class_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

def load_class_names(path: Path) -> list[str]:
    if not path.exists(): raise FileNotFoundError(f"Class mapping was not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("Class mapping must be a JSON list of strings.")
    return data
