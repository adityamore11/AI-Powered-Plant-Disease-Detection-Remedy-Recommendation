"""Evaluate the saved model using a reproducible test partition and save plots."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from config import CLASS_NAMES_PATH, HISTORY_PATH, MODEL_PATH, RESULTS_DIR, DATASET_DIR
from src.predict import load_model
from src.train import make_datasets, resolve_image_root
from src.utils import load_class_names

def plot_history() -> None:
    if not HISTORY_PATH.exists(): return
    h = json.loads(HISTORY_PATH.read_text()); fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, label in (("accuracy", "Accuracy"), ("loss", "Loss")):
        axes[0 if key == "accuracy" else 1].plot(h.get(key, []), label="train"); axes[0 if key == "accuracy" else 1].plot(h.get("val_" + key, []), label="validation"); axes[0 if key == "accuracy" else 1].set_title(label); axes[0 if key == "accuracy" else 1].legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "training_curves.png", dpi=150); plt.close(fig)

def main(dataset_dir: Path = DATASET_DIR):
    RESULTS_DIR.mkdir(exist_ok=True); classes = load_class_names(CLASS_NAMES_PATH); (_, _, test_ds), discovered = make_datasets(resolve_image_root(dataset_dir), 32, 42)
    if classes != discovered: raise ValueError("Saved class mapping differs from this dataset; evaluate with the training dataset.")
    model = load_model(MODEL_PATH)
    # Predict each batch alongside its labels in one pass. This remains correct
    # even when TensorFlow shuffles the reproducible test partition.
    actual_batches, prediction_batches = [], []
    for images, labels_batch in test_ds:
        actual_batches.append(labels_batch.numpy())
        prediction_batches.append(model(images, training=False).numpy().argmax(axis=1))
    actual = np.concatenate(actual_batches)
    predicted = np.concatenate(prediction_batches)
    # A reproducible test split may not contain every rare class.  Keep the
    # report aligned to the 38-model-output mapping and report zero support.
    labels = list(range(len(classes)))
    report = classification_report(actual, predicted, labels=labels, target_names=classes, zero_division=0, output_dict=True)
    (RESULTS_DIR / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(16, 16)); ConfusionMatrixDisplay.from_predictions(actual, predicted, labels=labels, display_labels=classes, xticks_rotation=90, ax=ax, colorbar=False); fig.tight_layout(); fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150); plt.close(fig); plot_history()
    print(f"Accuracy: {report['accuracy']:.4f}. Results saved in {RESULTS_DIR}")
if __name__ == "__main__": main()
