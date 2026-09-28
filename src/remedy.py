"""Safe, knowledge-base driven remedy lookup; it never changes model predictions."""
from __future__ import annotations
import json
from pathlib import Path
from config import DISEASE_INFO_PATH
from src.utils import friendly_label

def load_knowledge_base(path: Path = DISEASE_INFO_PATH) -> dict:
    if not path.exists(): return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError): return {}

def get_disease_info(class_name: str, path: Path = DISEASE_INFO_PATH) -> dict:
    info = load_knowledge_base(path).get(class_name)
    if info: return info
    plant, disease = friendly_label(class_name)
    return {
        "plant": plant,
        "disease": disease,
        "description": "Remedy information is not available for this class yet.",
        "symptoms": ["Disease-specific symptoms are not yet listed. Leaf damage can have several different causes, so seek a local diagnosis."],
        "recommended_actions": ["Please consult a local agricultural expert."],
        "prevention": [
            "Use clean tools and healthy planting material.",
            "Monitor plants regularly and follow crop-specific local guidance.",
            "Where suitable for this crop, maintain airflow and avoid unnecessary foliage wetting."
        ],
        "precautions": ["This AI-assisted screening result is not a professional diagnosis."],
        "severity": "Unknown",
        "disclaimer": "Use local agricultural guidance and product labels for any treatment."
    }
