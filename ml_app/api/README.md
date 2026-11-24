# API skeleton (winning model for inference)

Este módulo expone un servicio FastAPI para clasificación multietiqueta de emociones usando el modelo ganador de P06:

- **Representación:** TF-IDF (unigramas + bigramas)
- **Modelo:** OneVsRest(Calibrated LinearSVC, C=0.75, cv=3)
- **Umbral de clasificación:** 0.30

## Instalación rápida

```bash
python3 -m pip install -r ml_app/api/requirements.txt
```

## Entrenar y guardar el artefacto del modelo ganador

```bash
python3 -m ml_app.api.train_winning_model
```

Por defecto, el artefacto se guarda en:

`ml_app/artifacts/p06/models/winning_tfidf_calibrated_linearsvc_c_0_75.joblib`

## Ejecución local

```bash
uvicorn ml_app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Ejecución con Docker

Desde la raíz del repositorio:

```bash
docker build -f ml_app/api/Dockerfile -t emotionai-api .
docker run --rm -p 8000:8000 emotionai-api
```

## Endpoints

- `GET /health`: estado del servicio y metadatos del modelo.
- `POST /predict`: inferencia por lote.

## Ejemplo de request

```json
{
  "texts": [
    "I am happy today",
    "This is frustrating"
  ]
}
```

La API retorna `predicted_labels` (clases positivas según umbral con fallback al argmax) y `scores` con probabilidad por emoción.
