import json
from functools import lru_cache
from pathlib import Path
from ..config import get_settings

def parse_label(label: str) -> tuple[str, str]:
    parts = label.replace("__", "_").split("___", 1)
    if len(parts) == 2:
        return tuple(part.replace("_", " ").strip() for part in parts)
    return label.replace("_", " ").strip(), "Unspecified condition"

@lru_cache
def _load(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)

def get_disease(label: str) -> dict | None:
    path = str(get_settings().resolve(get_settings().knowledge_base_path))
    return _load(path).get(label)

def list_diseases() -> list[dict]:
    path = str(get_settings().resolve(get_settings().knowledge_base_path))
    return list(_load(path).values())
