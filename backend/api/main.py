import io
import os
import tempfile
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.models.inference import (
    load_image_model,
    load_video_model,
    predict_image_from_bytes,
    predict_video_from_path,
)


class HealthResponse(BaseModel):
    status: str


class PredictionResponse(BaseModel):
    label: str
    is_ai: bool
    confidence: float
    detail: Optional[str] = None


app = FastAPI(title="AI vs Real Media Detector", version="0.1.0")

# CORS for local dev & simple deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global model handles
image_model = None
video_model = None
model_warning = None


@app.on_event("startup")
def startup_event() -> None:
    global image_model, video_model, model_warning
    image_weights = os.getenv("IMAGE_WEIGHTS_PATH", "")
    video_weights = os.getenv("VIDEO_WEIGHTS_PATH", "")

    image_model, image_loaded = load_image_model(image_weights if image_weights else None)
    video_model, video_loaded = load_video_model(video_weights if video_weights else None)

    warnings = []
    if not image_loaded:
        warnings.append("Image model is untrained (no weights found). Predictions may be unreliable.")
    if not video_loaded:
        warnings.append("Video model is untrained (no weights found). Predictions may be unreliable.")
    model_warning = " ".join(warnings) if warnings else None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image type. Use JPEG/PNG/WebP.")

    content = await file.read()
    try:
        prob_ai = predict_image_from_bytes(image_model, content)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to process image: {exc}")

    label = "AI-generated" if prob_ai >= 0.5 else "Real"
    response = PredictionResponse(
        label=label,
        is_ai=bool(prob_ai >= 0.5),
        confidence=float(prob_ai if prob_ai >= 0.5 else 1.0 - prob_ai),
        detail=model_warning,
    )
    return response


@app.post("/predict/video", response_model=PredictionResponse)
async def predict_video(file: UploadFile = File(...), num_frames: int = 32) -> PredictionResponse:
    if file.content_type not in {"video/mp4", "video/x-matroska", "video/avi", "video/quicktime"}:
        raise HTTPException(status_code=400, detail="Unsupported video type. Use MP4/MKV/AVI/MOV.")

    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        try:
            prob_ai = predict_video_from_path(video_model, tmp.name, num_frames=num_frames)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to process video: {exc}")

    label = "AI-generated" if prob_ai >= 0.5 else "Real"
    response = PredictionResponse(
        label=label,
        is_ai=bool(prob_ai >= 0.5),
        confidence=float(prob_ai if prob_ai >= 0.5 else 1.0 - prob_ai),
        detail=model_warning,
    )
    return response