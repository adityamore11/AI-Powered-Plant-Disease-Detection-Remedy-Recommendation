from fastapi import APIRouter
from ..config import get_settings

router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    settings = get_settings()
    return {"status": "ok", "model_ready": settings.resolve(settings.model_path).exists() and settings.resolve(settings.class_names_path).exists()}
