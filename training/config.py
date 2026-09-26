from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
class TrainingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    dataset_dir: str = "dataset"
    model_path: str = "models/plant_disease.keras"
    class_names_path: str = "models/class_names.json"
    knowledge_base_path: str = "data/disease_knowledge.json"
    results_dir: str = "results"
    image_size: int = 224
    seed: int = 42
    batch_size: int = 32
    head_epochs: int = 8
    finetune_epochs: int = 5
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    def resolve(self, value: str) -> Path:
        p = Path(value); return p if p.is_absolute() else ROOT / p
