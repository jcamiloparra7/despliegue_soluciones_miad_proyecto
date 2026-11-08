#!/usr/bin/env python3
"""Train and evaluate a baseline multilabel model for P05."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier


EMOTION_NAMES = [
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


def read_split(path: Path) -> tuple[list[str], list[list[int]]]:
    table = pq.read_table(path, columns=["text", "labels"])
    texts = table["text"].to_pylist()
    labels = [row or [] for row in table["labels"].to_pylist()]
    return texts, labels


def labels_to_multihot(labels: list[list[int]], num_classes: int) -> np.ndarray:
    y = np.zeros((len(labels), num_classes), dtype=np.int8)
    for i, row in enumerate(labels):
        for idx in row:
            if 0 <= idx < num_classes:
                y[i, idx] = 1
    return y


def apply_threshold_with_fallback(proba: np.ndarray, threshold: float) -> np.ndarray:
    y_pred = (proba >= threshold).astype(np.int8)
    empty_mask = y_pred.sum(axis=1) == 0
    if np.any(empty_mask):
        best_idx = np.argmax(proba[empty_mask], axis=1)
        y_pred[empty_mask, best_idx] = 1
    return y_pred


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "f1_micro": round(f1_score(y_true, y_pred, average="micro", zero_division=0), 6),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 6),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 6),
        "precision_micro": round(
            precision_score(y_true, y_pred, average="micro", zero_division=0), 6
        ),
        "recall_micro": round(recall_score(y_true, y_pred, average="micro", zero_division=0), 6),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data" / "go_emotions" / "simplified"
    out_dir = repo_root / "entrega_1" / "artifacts" / "p05"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_texts, train_labels = read_split(data_root / "train-00000-of-00001.parquet")
    val_texts, val_labels = read_split(data_root / "validation-00000-of-00001.parquet")
    test_texts, test_labels = read_split(data_root / "test-00000-of-00001.parquet")

    y_train = labels_to_multihot(train_labels, len(EMOTION_NAMES))
    y_val = labels_to_multihot(val_labels, len(EMOTION_NAMES))
    y_test = labels_to_multihot(test_labels, len(EMOTION_NAMES))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_features=50000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    estimator = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=30,
        tol=1e-3,
        random_state=42,
    )
    model = OneVsRestClassifier(estimator, n_jobs=-1)
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)
    test_proba = model.predict_proba(X_test)

    threshold_candidates = np.arange(0.10, 0.91, 0.05)
    scan_rows = []
    best_threshold = None
    best_micro_f1 = -1.0
    for t in threshold_candidates:
        y_val_pred = apply_threshold_with_fallback(val_proba, float(t))
        micro_f1 = f1_score(y_val, y_val_pred, average="micro", zero_division=0)
        scan_rows.append({"threshold": round(float(t), 2), "validation_f1_micro": round(float(micro_f1), 6)})
        if micro_f1 > best_micro_f1:
            best_micro_f1 = float(micro_f1)
            best_threshold = float(t)

    y_val_pred = apply_threshold_with_fallback(val_proba, best_threshold)
    y_test_pred = apply_threshold_with_fallback(test_proba, best_threshold)

    val_metrics = metric_block(y_val, y_val_pred)
    test_metrics = metric_block(y_test, y_test_pred)

    # Per-class metrics on test.
    per_class_rows = []
    for i, emotion in enumerate(EMOTION_NAMES):
        f1 = f1_score(y_test[:, i], y_test_pred[:, i], zero_division=0)
        support = int(y_test[:, i].sum())
        predicted_positive = int(y_test_pred[:, i].sum())
        per_class_rows.append(
            {
                "emotion": emotion,
                "f1_test": round(float(f1), 6),
                "support_test": support,
                "predicted_positive_test": predicted_positive,
            }
        )
    per_class_rows.sort(key=lambda r: r["f1_test"], reverse=True)

    results = {
        "model": {
            "type": "OneVsRest(SGDClassifier log_loss) + TF-IDF",
            "vectorizer": {
                "ngram_range": [1, 2],
                "min_df": 3,
                "max_features": 50000,
                "sublinear_tf": True,
            },
            "estimator": {
                "loss": "log_loss",
                "alpha": 1e-5,
                "max_iter": 30,
                "tol": 1e-3,
                "random_state": 42,
            },
        },
        "data": {
            "train_rows": int(y_train.shape[0]),
            "validation_rows": int(y_val.shape[0]),
            "test_rows": int(y_test.shape[0]),
            "num_classes": len(EMOTION_NAMES),
        },
        "threshold_selection": {
            "selected_threshold": best_threshold,
            "selection_metric": "validation f1_micro",
            "best_validation_f1_micro": round(best_micro_f1, 6),
        },
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
        "notes": [
            "Threshold selected on validation split only.",
            "Predictions force at least one positive class via argmax fallback when no class crosses threshold.",
        ],
    }

    (out_dir / "p05_baseline_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(
        out_dir / "p05_threshold_scan.csv",
        scan_rows,
        ["threshold", "validation_f1_micro"],
    )
    write_csv(
        out_dir / "p05_per_class_f1_test.csv",
        per_class_rows,
        ["emotion", "f1_test", "support_test", "predicted_positive_test"],
    )

    print(f"Generated: {out_dir / 'p05_baseline_results.json'}")
    print(f"Generated: {out_dir / 'p05_threshold_scan.csv'}")
    print(f"Generated: {out_dir / 'p05_per_class_f1_test.csv'}")


if __name__ == "__main__":
    main()
