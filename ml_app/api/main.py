"""FastAPI skeleton for text emotion prediction."""

from __future__ import annotations

from fastapi import FastAPI

from .model import EMOTION_LABELS, WinningEmotionModel
from .schemas import HealthResponse, PredictRequest, PredictResponse, PredictionItem

app = FastAPI(
    title="Go Emotions Prediction API",
    description="API for multilabel emotion classification.",
    version="0.2.0",
)

prediction_model = WinningEmotionModel()


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {"message": "Go Emotions API is running. See /docs for Swagger UI."}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_name=prediction_model.model_name,
        model_version=prediction_model.model_version,
        available_labels=EMOTION_LABELS,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest) -> PredictResponse:
    predictions = []
    for text in payload.texts:
        result = prediction_model.predict(text)
        predictions.append(
            PredictionItem(
                text=text,
                predicted_labels=result.predicted_labels,
                scores=result.scores,
            )
        )

    return PredictResponse(
        model_name=prediction_model.model_name,
        model_version=prediction_model.model_version,
        predictions=predictions,
    )
