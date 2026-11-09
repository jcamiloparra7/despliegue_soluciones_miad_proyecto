from pathlib import Path
import sys

import joblib
import numpy as np
import pyarrow.parquet as pq

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_app.scripts.p05_baseline_go_emotions import (
    EMOTION_NAMES,
    labels_to_multihot,
)

DATA_DIR = PROJECT_ROOT / "data" / "go_emotions" / "simplified"
MODEL_DIR = PROJECT_ROOT / "entrega_2" / "dashboard" / "models"


def cargar_entrenamiento():
    ruta = DATA_DIR / "train-00000-of-00001.parquet"

    tabla = pq.read_table(ruta, columns=["text", "labels"])

    textos = tabla["text"].to_pylist()
    etiquetas = [fila or [] for fila in tabla["labels"].to_pylist()]

    return textos, etiquetas


def main():
    print("Cargando datos de entrenamiento...")

    textos, etiquetas = cargar_entrenamiento()

    y_train = labels_to_multihot(
        etiquetas,
        len(EMOTION_NAMES)
    )

    print("Preparando vectorizador TF-IDF...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_features=50000,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(textos)

    print("Entrenando clasificador multietiqueta...")

    estimator = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=30,
        tol=1e-3,
        random_state=42,
    )

    model = OneVsRestClassifier(
        estimator,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    archivo_modelo = MODEL_DIR / "emotionai_baseline.joblib"

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": model,
            "emotion_names": EMOTION_NAMES,
            "threshold": 0.20,
        },
        archivo_modelo
    )

    print(f"Modelo guardado en: {archivo_modelo}")


if __name__ == "__main__":
    main()
