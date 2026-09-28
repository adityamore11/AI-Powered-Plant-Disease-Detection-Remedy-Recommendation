"""Train and fine-tune EfficientNetB0 on an extracted PlantVillage-style dataset."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tensorflow as tf
from config import *
from src.utils import ensure_directories, save_class_names, set_seed

def resolve_image_root(dataset_dir: Path) -> Path:
    """Locate a directory whose immediate children are PlantVillage class folders."""
    candidates = [dataset_dir, *dataset_dir.rglob("*")]
    # If archive variants exist, the ordinary color images are the best default.
    candidates.sort(key=lambda p: (p.name != "color", len(p.parts)))
    for candidate in candidates:
        if candidate.is_dir() and sum(1 for p in candidate.iterdir() if p.is_dir() and "___" in p.name) >= 2:
            # Prefer normal color images when a Kaggle archive offers variants.
            return candidate
    raise FileNotFoundError(f"No class folders were found under {dataset_dir}. Extract the dataset first.")

def make_datasets(image_root: Path, batch_size: int, seed: int):
    common = dict(directory=image_root, labels="inferred", label_mode="int", image_size=IMAGE_SIZE,
                  batch_size=batch_size, validation_split=VALIDATION_SPLIT + TEST_SPLIT, subset="training", seed=seed)
    train = tf.keras.utils.image_dataset_from_directory(**common)
    held_out = tf.keras.utils.image_dataset_from_directory(directory=image_root, labels="inferred", label_mode="int", image_size=IMAGE_SIZE,
        batch_size=batch_size, validation_split=VALIDATION_SPLIT + TEST_SPLIT, subset="validation", seed=seed, shuffle=True)
    held_batches = tf.data.experimental.cardinality(held_out).numpy()
    if held_batches < 2: raise ValueError("Dataset is too small to create validation and test sets.")
    val_batches = max(1, round(held_batches * VALIDATION_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT)))
    val, test = held_out.take(val_batches), held_out.skip(val_batches)
    autotune = tf.data.AUTOTUNE
    return tuple(ds.prefetch(autotune) for ds in (train, val, test)), train.class_names

def build_model(num_classes: int):
    augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.08), tf.keras.layers.RandomZoom(0.1), tf.keras.layers.RandomTranslation(0.08, 0.08), tf.keras.layers.RandomContrast(0.1)], name="augmentation")
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs); x = base(x, training=False); x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x); outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="plant_disease_efficientnetb0")
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, base

def main(dataset_dir: Path = DATASET_DIR, head_epochs: int = HEAD_EPOCHS, fine_tune_epochs: int = FINE_TUNE_EPOCHS):
    ensure_directories([MODELS_DIR, RESULTS_DIR]); set_seed(SEED)
    image_root = resolve_image_root(dataset_dir); print(f"Using images from: {image_root}")
    datasets, class_names = make_datasets(image_root, BATCH_SIZE, SEED); train_ds, val_ds, test_ds = datasets
    print(f"Discovered {len(class_names)} classes:"); print("\n".join(class_names)); save_class_names(class_names, CLASS_NAMES_PATH)
    model, base = build_model(len(class_names))
    callbacks = [tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", mode="max", save_best_only=True), tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.2)]
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, callbacks=callbacks)
    base.trainable = True
    for layer in base.layers[:-30]: layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LEARNING_RATE), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history2 = model.fit(train_ds, validation_data=val_ds, initial_epoch=len(history1.history["loss"]), epochs=len(history1.history["loss"]) + fine_tune_epochs, callbacks=callbacks)
    history = {k: history1.history.get(k, []) + history2.history.get(k, []) for k in set(history1.history) | set(history2.history)}
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    loss, accuracy = model.evaluate(test_ds, verbose=0); print(f"Final held-out test accuracy: {accuracy:.4f}; loss: {loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path, default=DATASET_DIR); parser.add_argument("--head-epochs", type=int, default=HEAD_EPOCHS); parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    args = parser.parse_args(); main(args.dataset, args.head_epochs, args.fine_tune_epochs)
