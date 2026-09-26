from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import get_settings
from .routes import health, prediction

settings = get_settings()
app = FastAPI(title="Plant Disease AI API", version="1.0.0", description="Local model inference and safe remedy lookup.")
app.add_middleware(CORSMiddleware, allow_origins=settings.origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(health.router)
app.include_router(prediction.router)
