from pydantic import BaseModel, Field

class TopPrediction(BaseModel):
    label: str
    plant: str
    disease: str
    confidence: float = Field(ge=0, le=1)

class DiseaseInfo(BaseModel):
    label: str
    plant_name: str
    disease_name: str
    description: str
    symptoms: list[str]
    recommended_actions: list[str]
    prevention: list[str]
    precautions: list[str]
    severity: str
    disclaimer: str

class PredictionResponse(BaseModel):
    plant: str
    disease: str
    label: str
    confidence: float
    is_low_confidence: bool
    message: str | None = None
    top_predictions: list[TopPrediction]
    disease_info: DiseaseInfo | None = None
    explanation_available: bool = False
    heatmap_base64: str | None = None
