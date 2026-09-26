"""Dataset discovery and reproducible file-level stratified splitting."""
from collections import Counter
from pathlib import Path
import random
import tensorflow as tf

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_ALIASES = {"train": "train", "training": "train", "valid": "validation", "val": "validation", "validation": "validation", "test": "test"}

def _images(folder: Path): return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
def _class_dirs(root: Path): return sorted(p for p in root.iterdir() if p.is_dir())

def discover_splits(dataset_dir: Path, seed=42, validation_fraction=.15, test_fraction=.15):
    if not dataset_dir.exists(): raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    named = {SPLIT_ALIASES[p.name.lower()]: p for p in dataset_dir.iterdir() if p.is_dir() and p.name.lower() in SPLIT_ALIASES}
    if "train" in named and ("validation" in named or "test" in named):
        source = {key: {d.name: _images(d) for d in _class_dirs(folder)} for key, folder in named.items()}
        classes = sorted(set().union(*(group.keys() for group in source.values())))
        splits = {key: [(path, classes.index(label)) for label, paths in group.items() for path in paths] for key, group in source.items()}
        if "validation" not in splits: splits["validation"] = []
        if "test" not in splits: splits["test"] = []
        return classes, splits
    groups = {d.name: _images(d) for d in _class_dirs(dataset_dir)}
    if not groups or not any(groups.values()):
        raise ValueError("No class folders with images found. Extract the dataset ZIP so dataset/<class name>/<image> exists.")
    classes, splits = sorted(groups), {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for label in classes:
        files = groups[label][:]; rng.shuffle(files)
        n = len(files); n_test = max(1, round(n * test_fraction)); n_val = max(1, round(n * validation_fraction))
        if n - n_test - n_val < 1: raise ValueError(f"Class {label!r} has too few images to split.")
        index = classes.index(label)
        splits["test"] += [(p, index) for p in files[:n_test]]
        splits["validation"] += [(p, index) for p in files[n_test:n_test+n_val]]
        splits["train"] += [(p, index) for p in files[n_test+n_val:]]
    return classes, splits

def make_dataset(items, image_size, batch_size, training=False, seed=42):
    paths = [str(p) for p, _ in items]; labels = [y for _, y in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training: ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    def decode(path, label):
        data = tf.io.read_file(path); image = tf.io.decode_image(data, channels=3, expand_animations=False)
        image.set_shape([None, None, 3]); image = tf.image.resize(image, (image_size, image_size))
        return tf.cast(image, tf.float32), label
    return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def summary(classes, splits):
    print(f"Classes ({len(classes)}): {classes}")
    for name, items in splits.items(): print(f"{name}: {len(items)} images; distribution: {dict(Counter(classes[y] for _, y in items))}")
