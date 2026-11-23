"""Pydantic schemas for request and response payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_items=1,
        description="Batch of input texts to classify.",
    )


class PredictionItem(BaseModel):
    text: str
    predicted_labels: list[str]
    scores: dict[str, float]


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    available_labels: list[str]
