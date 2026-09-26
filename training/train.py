import json, sys
from pathlib import Path
import tensorflow as tf
from .config import TrainingSettings
from .dataset import discover_splits, make_dataset, summary

def ensure_knowledge_base(classes, path):
    existing = json.loads(path.read_text()) if path.exists() else {}
    disclaimer = "AI-assisted screening only; confirm important decisions with a qualified agricultural professional. Follow product labels and local agricultural guidance."
    for label in classes:
        if label in existing: continue
        parts = label.split("___", 1); plant = parts[0].replace("_", " "); disease = (parts[1] if len(parts)>1 else "Unspecified condition").replace("_", " ")
        existing[label] = {"label": label, "plant_name": plant, "disease_name": disease,
          "description": f"Information record for the model class {plant} — {disease}.", "symptoms": ["Inspect leaves and compare with local extension guidance."],
          "recommended_actions": ["Isolate and monitor affected plants where practical.", "Remove severely affected plant material using clean tools.", "Use an appropriate registered treatment according to the product label and local agricultural guidance."],
          "prevention": ["Use clean planting material and sanitize tools.", "Promote airflow and avoid prolonged leaf wetness."],
          "precautions": ["Do not use this result as a definitive diagnosis.", "Do not apply chemicals without reading the product label."], "severity": "Consult local guidance", "disclaimer": disclaimer}
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

def make_model(num_classes, image_size):
    augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(.08), tf.keras.layers.RandomZoom(.1), tf.keras.layers.RandomTranslation(.05,.05), tf.keras.layers.RandomContrast(.1)])
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(image_size,image_size,3)); base.trainable = False
    inputs = tf.keras.Input((image_size,image_size,3)); x = augmentation(inputs); x = tf.keras.applications.efficientnet.preprocess_input(x); x = base(x, training=False); x = tf.keras.layers.GlobalAveragePooling2D()(x); x = tf.keras.layers.Dropout(.3)(x); outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs); model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, base

def main():
    s = TrainingSettings(); classes, splits = discover_splits(s.resolve(s.dataset_dir), s.seed, s.validation_fraction, s.test_fraction); summary(classes, splits)
    if not splits["train"] or not splits["validation"]: raise ValueError("Training and validation images are required.")
    train = make_dataset(splits["train"],s.image_size,s.batch_size,True,s.seed); val = make_dataset(splits["validation"],s.image_size,s.batch_size)
    model, base = make_model(len(classes), s.image_size); model_path=s.resolve(s.model_path); model_path.parent.mkdir(parents=True,exist_ok=True)
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor="val_accuracy", mode="max"),tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=.3)]
    histories=[model.fit(train, validation_data=val, epochs=s.head_epochs, callbacks=callbacks)]
    if s.finetune_epochs:
        base.trainable=True
        for layer in base.layers[:-30]: layer.trainable=False
        model.compile(tf.keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", metrics=["accuracy"])
        histories.append(model.fit(train, validation_data=val, epochs=s.finetune_epochs, callbacks=callbacks))
    s.resolve(s.class_names_path).write_text(json.dumps(classes, indent=2)); ensure_knowledge_base(classes,s.resolve(s.knowledge_base_path))
    merged={key: sum((h.history.get(key,[]) for h in histories),[]) for key in {k for h in histories for k in h.history}}
    result=s.resolve(s.results_dir); result.mkdir(parents=True,exist_ok=True); (result/"training_history.json").write_text(json.dumps(merged,indent=2))
    print(f"Saved model to {model_path}. Run: python -m training.evaluate")
if __name__ == "__main__": main()
