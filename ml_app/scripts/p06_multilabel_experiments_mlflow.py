#!/usr/bin/env python3
"""Run fourteen multilabel experiments and log concise metrics to MLflow."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Callable

import mlflow
import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


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

MLFLOW_METRICS_TO_LOG = (
    "validation_f1_micro",
    "test_f1_micro",
    "test_f1_macro",
    "test_precision_micro",
    "test_recall_micro",
)
OVR_N_JOBS = 1


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


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_micro": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def find_best_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, candidates: np.ndarray
) -> tuple[float, float, list[dict[str, float]]]:
    best_threshold = float(candidates[0])
    best_score = -1.0
    scan_rows: list[dict[str, float]] = []

    for threshold in candidates:
        t = float(threshold)
        y_pred = apply_threshold_with_fallback(y_proba, t)
        score = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        scan_rows.append({"threshold": t, "validation_f1_micro": score})
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score, scan_rows


def run_experiment(
    *,
    name: str,
    feature_type: str,
    model_family: str,
    model_factory: Callable[[], OneVsRestClassifier],
    model_params: dict,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    threshold_candidates: np.ndarray,
) -> dict:
    model = model_factory()
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)
    test_proba = model.predict_proba(X_test)

    best_threshold, best_val_f1_micro, threshold_scan = find_best_threshold(
        y_val, val_proba, threshold_candidates
    )
    y_val_pred = apply_threshold_with_fallback(val_proba, best_threshold)
    y_test_pred = apply_threshold_with_fallback(test_proba, best_threshold)

    result = {
        "experiment_name": name,
        "feature_type": feature_type,
        "model_family": model_family,
        "model_params": model_params,
        "threshold_selection": {
            "selected_threshold": best_threshold,
            "selection_metric": "validation f1_micro",
            "best_validation_f1_micro": best_val_f1_micro,
            "candidates": [float(x) for x in threshold_candidates.tolist()],
            "scan": threshold_scan,
        },
        "metrics": {
            "validation": metric_block(y_val, y_val_pred),
            "test": metric_block(y_test, y_test_pred),
        },
    }
    return result


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def log_to_mlflow(result: dict) -> None:
    mlflow.log_param("feature_type", result["feature_type"])
    mlflow.log_param("model_family", result["model_family"])
    for key, value in result["model_params"].items():
        mlflow.log_param(key, value)

    threshold_info = result["threshold_selection"]
    mlflow.log_param("selected_threshold", threshold_info["selected_threshold"])
    mlflow.log_param("selection_metric", threshold_info["selection_metric"])

    mlflow.log_metric("validation_f1_micro", result["metrics"]["validation"]["f1_micro"])
    mlflow.log_metric("test_f1_micro", result["metrics"]["test"]["f1_micro"])
    mlflow.log_metric("test_f1_macro", result["metrics"]["test"]["f1_macro"])
    mlflow.log_metric("test_precision_micro", result["metrics"]["test"]["precision_micro"])
    mlflow.log_metric("test_recall_micro", result["metrics"]["test"]["recall_micro"])


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data" / "go_emotions" / "simplified"
    out_dir = repo_root / "ml_app" / "artifacts" / "p06"
    out_dir.mkdir(parents=True, exist_ok=True)

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8050")
    mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "go-emotions-multilabel-miad")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    try:
        mlflow.set_experiment(mlflow_experiment_name)
    except Exception as exc:
        raise RuntimeError(
            "MLflow tracking server is not reachable. "
            f"URI='{mlflow_tracking_uri}'. Verify MLFLOW_TRACKING_URI and network access "
            "(EC2 security group/firewall/port 8050)."
        ) from exc

    train_texts, train_labels = read_split(data_root / "train-00000-of-00001.parquet")
    val_texts, val_labels = read_split(data_root / "validation-00000-of-00001.parquet")
    test_texts, test_labels = read_split(data_root / "test-00000-of-00001.parquet")

    y_train = labels_to_multihot(train_labels, len(EMOTION_NAMES))
    y_val = labels_to_multihot(val_labels, len(EMOTION_NAMES))
    y_test = labels_to_multihot(test_labels, len(EMOTION_NAMES))
    threshold_candidates = np.arange(0.10, 0.91, 0.05)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_features=50000,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(train_texts).tocsr(copy=True)
    X_val_tfidf = vectorizer.transform(val_texts).tocsr(copy=True)
    X_test_tfidf = vectorizer.transform(test_texts).tocsr(copy=True)

    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(embedding_model_name)
    X_train_emb = embedding_model.encode(
        train_texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    X_val_emb = embedding_model.encode(
        val_texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    X_test_emb = embedding_model.encode(
        test_texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    model_specs = [
        {
            "name": "baseline_simple_sgd_defaults",
            "model_family": "sgd_log_loss",
            "model_params": {"loss": "log_loss", "random_state": 42},
            "model_factory": lambda: OneVsRestClassifier(
                SGDClassifier(
                    loss="log_loss",
                    random_state=42,
                ),
                n_jobs=OVR_N_JOBS,
            ),
        },
        {
            "name": "logreg_l2_c_1.0",
            "model_family": "logistic_regression_l2",
            "model_params": {
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 1000,
                "random_state": 42,
            },
            "model_factory": lambda: OneVsRestClassifier(
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                ),
                n_jobs=OVR_N_JOBS,
            ),
        },
        {
            "name": "logreg_l2_c_3.0",
            "model_family": "logistic_regression_l2",
            "model_params": {
                "C": 3.0,
                "solver": "liblinear",
                "max_iter": 1000,
                "random_state": 42,
            },
            "model_factory": lambda: OneVsRestClassifier(
                LogisticRegression(
                    C=3.0,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                ),
                n_jobs=OVR_N_JOBS,
            ),
        },
        {
            "name": "sgd_logloss_alpha_1e-4",
            "model_family": "sgd_log_loss",
            "model_params": {"alpha": 1e-4, "max_iter": 50, "tol": 1e-3, "random_state": 42},
            "model_factory": lambda: OneVsRestClassifier(
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    max_iter=50,
                    tol=1e-3,
                    random_state=42,
                ),
                n_jobs=OVR_N_JOBS,
            ),
        },
        {
            "name": "sgd_logloss_alpha_5e-5",
            "model_family": "sgd_log_loss",
            "model_params": {"alpha": 5e-5, "max_iter": 50, "tol": 1e-3, "random_state": 42},
            "model_factory": lambda: OneVsRestClassifier(
                SGDClassifier(
                    loss="log_loss",
                    alpha=5e-5,
                    max_iter=50,
                    tol=1e-3,
                    random_state=42,
                ),
                n_jobs=OVR_N_JOBS,
            ),
        },
        {
            "name": "calibrated_linearsvc_c_1.0",
            "model_family": "calibrated_linear_svc",
            "model_params": {"C": 1.0, "cv": 3, "random_state": 42},
            "model_factory": lambda: OneVsRestClassifier(
                CalibratedClassifierCV(
                    estimator=LinearSVC(C=1.0, random_state=42),
                    cv=3,
                ),
                n_jobs=OVR_N_JOBS,
            ),
        },
        {
            "name": "calibrated_linearsvc_c_2.0",
            "model_family": "calibrated_linear_svc",
            "model_params": {"C": 2.0, "cv": 3, "random_state": 42},
            "model_factory": lambda: OneVsRestClassifier(
                CalibratedClassifierCV(
                    estimator=LinearSVC(C=2.0, random_state=42),
                    cv=3,
                ),
                n_jobs=OVR_N_JOBS,
            ),
        },
    ]

    feature_sets = [
        {
            "feature_suffix": "tfidf",
            "feature_type": "tfidf_unigram_bigram",
            "X_train": X_train_tfidf,
            "X_val": X_val_tfidf,
            "X_test": X_test_tfidf,
        },
        {
            "feature_suffix": "embeddings",
            "feature_type": "all_minilm_l6_v2_embeddings",
            "X_train": X_train_emb,
            "X_val": X_val_emb,
            "X_test": X_test_emb,
        },
    ]

    experiment_specs = []
    for model_spec in model_specs:
        for feature_spec in feature_sets:
            experiment_specs.append(
                {
                    "name": f"{feature_spec['feature_suffix']}_{model_spec['name']}",
                    "feature_type": feature_spec["feature_type"],
                    "model_family": model_spec["model_family"],
                    "model_params": model_spec["model_params"],
                    "model_factory": model_spec["model_factory"],
                    "X_train": feature_spec["X_train"],
                    "X_val": feature_spec["X_val"],
                    "X_test": feature_spec["X_test"],
                }
            )

    all_results = []
    summary_rows = []

    for spec in experiment_specs:
        result = run_experiment(
            name=spec["name"],
            feature_type=spec["feature_type"],
            model_family=spec["model_family"],
            model_factory=spec["model_factory"],
            model_params=spec["model_params"],
            X_train=spec["X_train"],
            X_val=spec["X_val"],
            X_test=spec["X_test"],
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            threshold_candidates=threshold_candidates,
        )
        all_results.append(result)
        summary_rows.append(
            {
                "experiment_name": result["experiment_name"],
                "feature_type": result["feature_type"],
                "model_family": result["model_family"],
                "selected_threshold": round(
                    float(result["threshold_selection"]["selected_threshold"]), 4
                ),
                "validation_f1_micro": round(
                    float(result["metrics"]["validation"]["f1_micro"]), 6
                ),
                "validation_f1_macro": round(
                    float(result["metrics"]["validation"]["f1_macro"]), 6
                ),
                "test_f1_micro": round(float(result["metrics"]["test"]["f1_micro"]), 6),
                "test_f1_macro": round(float(result["metrics"]["test"]["f1_macro"]), 6),
                "test_f1_weighted": round(
                    float(result["metrics"]["test"]["f1_weighted"]), 6
                ),
                "test_precision_micro": round(
                    float(result["metrics"]["test"]["precision_micro"]), 6
                ),
                "test_recall_micro": round(float(result["metrics"]["test"]["recall_micro"]), 6),
            }
        )

        with mlflow.start_run(run_name=spec["name"]):
            log_to_mlflow(result)

    best_by_micro = max(summary_rows, key=lambda row: row["validation_f1_micro"])

    payload = {
        "tracking": {
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_experiment_name": mlflow_experiment_name,
        },
        "dataset": {
            "train_rows": int(y_train.shape[0]),
            "validation_rows": int(y_val.shape[0]),
            "test_rows": int(y_test.shape[0]),
            "num_classes": len(EMOTION_NAMES),
        },
        "embedding_model": embedding_model_name,
        "recommended_primary_metric": "f1_micro",
        "recommended_support_metric": "f1_macro",
        "selection_note": (
            "Use f1_micro as primary metric for model selection in this project, "
            "and monitor f1_macro to avoid regressions on minority labels."
        ),
        "mlflow_logged_metrics_per_experiment": list(MLFLOW_METRICS_TO_LOG),
        "best_experiment_by_validation_f1_micro": best_by_micro,
        "experiments": all_results,
    }

    (out_dir / "p06_experiment_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(
        out_dir / "p06_experiment_summary.csv",
        summary_rows,
        [
            "experiment_name",
            "feature_type",
            "model_family",
            "selected_threshold",
            "validation_f1_micro",
            "validation_f1_macro",
            "test_f1_micro",
            "test_f1_macro",
            "test_f1_weighted",
            "test_precision_micro",
            "test_recall_micro",
        ],
    )

    print(f"Generated: {out_dir / 'p06_experiment_results.json'}")
    print(f"Generated: {out_dir / 'p06_experiment_summary.csv'}")
    print(f"MLflow tracking URI: {mlflow_tracking_uri}")
    print(f"MLflow experiment: {mlflow_experiment_name}")


if __name__ == "__main__":
    main()
