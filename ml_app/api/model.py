"""Mock model implementation for the API skeleton."""

from __future__ import annotations

from dataclasses import dataclass


EMOTION_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


@dataclass(frozen=True)
class PredictionResult:
    predicted_labels: list[str]
    scores: dict[str, float]


class MockEmotionModel:
    """Deterministic mock predictor used while the production model is pending."""

    model_name = "go-emotions-mock"
    model_version = "0.1.0"

    def __init__(self) -> None:
        self._fixed_labels = ["neutral", "approval"]
        self._fixed_scores = {"neutral": 0.72, "approval": 0.28}

    def predict(self, _text: str) -> PredictionResult:
        return PredictionResult(
            predicted_labels=list(self._fixed_labels),
            scores=dict(self._fixed_scores),
        )
