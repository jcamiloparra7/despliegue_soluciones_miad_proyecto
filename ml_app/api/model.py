"""Winning model loading and prediction helpers for the API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from ml_app.scripts.p05_baseline_go_emotions import labels_to_multihot

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

WINNING_EXPERIMENT_NAME = "tfidf_calibrated_linearsvc_c_0.75"
WINNING_MODEL_VERSION = "p06-rerun-20261122"
WINNING_THRESHOLD = 0.30

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "go_emotions" / "simplified"
DEFAULT_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "ml_app"
    / "artifacts"
    / "p06"
    / "models"
    / "winning_tfidf_calibrated_linearsvc_c_0_75.joblib"
)


@dataclass(frozen=True)
class PredictionResult:
    predicted_labels: list[str]
    scores: dict[str, float]


def apply_threshold_with_fallback(proba: np.ndarray, threshold: float) -> np.ndarray:
    y_pred = (proba >= threshold).astype(np.int8)
    empty_mask = y_pred.sum(axis=1) == 0
    if np.any(empty_mask):
        best_idx = np.argmax(proba[empty_mask], axis=1)
        y_pred[empty_mask, best_idx] = 1
    return y_pred


def _resolve_artifact_path() -> Path:
    raw_path = os.getenv("EMOTION_MODEL_ARTIFACT_PATH")
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return DEFAULT_ARTIFACT_PATH


def _read_split(path: Path) -> tuple[list[str], list[list[int]]]:
    table = pq.read_table(path, columns=["text", "labels"])
    texts = table["text"].to_pylist()
    labels = [row or [] for row in table["labels"].to_pylist()]
    return texts, labels


def train_and_save_winning_model(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    artifact_path: Path | None = None,
) -> Path:
    target_path = artifact_path or _resolve_artifact_path()
    train_texts, train_labels = _read_split(data_root / "train-00000-of-00001.parquet")
    y_train = labels_to_multihot(train_labels, len(EMOTION_LABELS))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_features=50000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts).tocsr(copy=True)

    model = OneVsRestClassifier(
        CalibratedClassifierCV(
            estimator=LinearSVC(C=0.75, random_state=42),
            cv=3,
        ),
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    payload = {
        "vectorizer": vectorizer,
        "model": model,
        "emotion_names": list(EMOTION_LABELS),
        "threshold": WINNING_THRESHOLD,
        "model_name": WINNING_EXPERIMENT_NAME,
        "model_version": WINNING_MODEL_VERSION,
        "training_config": {
            "feature_type": "tfidf_unigram_bigram",
            "model_family": "calibrated_linear_svc",
            "C": 0.75,
            "cv": 3,
            "random_state": 42,
        },
    }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, target_path)
    return target_path


class WinningEmotionModel:
    """Loads and serves the winning multilabel model from P06 experiments."""

    def __init__(self, artifact_path: Path | None = None) -> None:
        self._artifact_path = artifact_path or _resolve_artifact_path()
        self._vectorizer: Any = None
        self._model: Any = None
        self.model_name = WINNING_EXPERIMENT_NAME
        self.model_version = WINNING_MODEL_VERSION
        self.threshold = WINNING_THRESHOLD
        self._load()

    def _load(self) -> None:
        if not self._artifact_path.exists():
            raise FileNotFoundError(
                "Winning model artifact not found at "
                f"'{self._artifact_path}'. Run "
                "'python -m ml_app.api.train_winning_model' first."
            )

        payload = joblib.load(self._artifact_path)

        required_keys = {"vectorizer", "model", "emotion_names", "threshold"}
        missing = required_keys.difference(payload.keys())
        if missing:
            raise ValueError(
                "Invalid winning model artifact. Missing keys: "
                + ", ".join(sorted(missing))
            )

        emotion_names = payload["emotion_names"]
        if list(emotion_names) != EMOTION_LABELS:
            raise ValueError(
                "The model artifact emotion labels do not match API labels."
            )

        self._vectorizer = payload["vectorizer"]
        self._model = payload["model"]
        self.threshold = float(payload["threshold"])
        self.model_name = str(payload.get("model_name", WINNING_EXPERIMENT_NAME))
        self.model_version = str(payload.get("model_version", WINNING_MODEL_VERSION))

    def predict(self, text: str) -> PredictionResult:
        X = self._vectorizer.transform([text])
        proba = self._model.predict_proba(X)
        y_pred = apply_threshold_with_fallback(proba, self.threshold)[0]
        row_scores = proba[0]

        selected_indices = np.where(y_pred == 1)[0]
        selected_indices = sorted(
            selected_indices.tolist(),
            key=lambda idx: float(row_scores[idx]),
            reverse=True,
        )

        predicted_labels = [EMOTION_LABELS[idx] for idx in selected_indices]
        scores = {
            label: round(float(score), 6)
            for label, score in zip(EMOTION_LABELS, row_scores)
        }
        return PredictionResult(predicted_labels=predicted_labels, scores=scores)
