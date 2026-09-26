from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    dataset_dir: str = "dataset"
    model_path: str = "models/plant_disease.keras"
    class_names_path: str = "models/class_names.json"
    knowledge_base_path: str = "data/disease_knowledge.json"
    results_dir: str = "results"
    confidence_threshold: float = 0.80
    max_upload_mb: int = 10
    cors_origins: str = "http://localhost:5173"
    image_size: int = 224

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    @property
    def origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

@lru_cache
def get_settings() -> Settings:
    return Settings()
