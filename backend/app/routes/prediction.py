from fastapi import APIRouter, File, HTTPException, UploadFile
from ..config import get_settings
from ..schemas import PredictionResponse
from ..services.preprocessing import load_image
from ..services.predictor import ModelUnavailableError, get_predictor
from ..services.remedy_service import get_disease, list_diseases

router = APIRouter(tags=["prediction"])

def _predict(image_bytes: bytes, content_type: str | None) -> dict:
    settings = get_settings()
    image = load_image(image_bytes, content_type, settings.max_upload_mb * 1024 * 1024)
    predictions = get_predictor().predict(image)
    best = predictions[0]
    low = best["confidence"] < settings.confidence_threshold
    return {**best, "is_low_confidence": low,
        "message": "Low confidence prediction. Please upload a clearer image or consult an agricultural expert." if low else None,
        "top_predictions": predictions, "disease_info": None if low else get_disease(best["label"]),
        "explanation_available": False, "heatmap_base64": None}

@router.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)):
    try:
        return _predict(await image.read(), image.content_type)
    except ValueError as error:
        raise HTTPException(400, str(error))
    except ModelUnavailableError as error:
        raise HTTPException(503, str(error))

@router.post("/predict/with-explanation", response_model=PredictionResponse)
async def predict_with_explanation(image: UploadFile = File(...)):
    return await predict(image)

@router.get("/diseases")
def diseases(): return list_diseases()

@router.get("/diseases/{label}")
def disease(label: str):
    item = get_disease(label)
    if not item: raise HTTPException(404, "Disease information was not found.")
    return item
