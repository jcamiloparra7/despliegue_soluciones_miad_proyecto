# API skeleton (mock model)

Este módulo expone un esqueleto de API en FastAPI para clasificación de emociones con un modelo simulado.

## Instalación rápida

```bash
python3 -m pip install -r ml_app/api/requirements.txt
```

## Ejecución local

```bash
uvicorn ml_app.api.main:app --host 0.0.0.0 --port 8000 --reload
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

El modelo actual es *mock*: siempre retorna la misma predicción (`neutral`, `approval`) y los mismos puntajes.
