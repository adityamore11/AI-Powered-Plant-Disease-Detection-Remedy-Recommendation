"""Central, machine-independent project settings."""
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.getenv("PLANT_DATASET_DIR", ROOT_DIR / "dataset"))
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = MODELS_DIR / "plant_disease.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
HISTORY_PATH = RESULTS_DIR / "training_history.json"
DISEASE_INFO_PATH = DATA_DIR / "disease_info.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 8
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
