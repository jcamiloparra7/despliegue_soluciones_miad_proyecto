# P02 - Habilitacion y caracterizacion de datos (GoEmotions)

## Proposito

Documentar la disponibilidad del dataset y su caracterizacion tecnica inicial para sustentar el modelado y la evaluacion del prototipo.

## 1) Habilitacion de datos

Comando de descarga utilizado:

```bash
hf download google-research-datasets/go_emotions --repo-type=dataset
```

La descarga se valido mediante lectura de archivos parquet en:

- `data/go_emotions/raw/`
- `data/go_emotions/simplified/`

## 2) Estructura validada

| Dataset / split | Filas | Columnas | Tamano aproximado |
|---|---:|---:|---:|
| `raw/train` | 211,225 | 37 | 24 MB |
| `simplified/train` | 43,410 | 3 | 2.6 MB |
| `simplified/validation` | 5,426 | 3 | 342 KB |
| `simplified/test` | 5,427 | 3 | 339 KB |

## 3) Hallazgos de caracterizacion

Consolidado sobre `simplified`:

- muestras: **54,263**
- etiquetas asignadas: **63,812**
- promedio de etiquetas por muestra: **1.176**
- control de calidad basico de texto: sin nulos y sin vacios en los splits evaluados.

Etiquetas con mayor frecuencia:

1. `neutral` (17,772)
2. `admiration` (5,122)
3. `approval` (3,687)

Etiquetas de baja frecuencia:

- `grief`, `pride`, `relief`, `nervousness`.

## 4) Implicaciones para el proyecto

1. La tarea debe tratarse como clasificacion **multietiqueta**.
2. El desbalance entre clases exige metricas agregadas y por etiqueta.
3. El idioma del dataset (ingles) es consistente con el alcance definido para el MVP.

## 5) Evidencia reproducible

Script:

- `entrega_1/scripts/p02_profile_go_emotions.py`

Notebook integrado:

- `entrega_1/notebooks/analisis_integrado_go_emotions.ipynb`

Artefactos generados:

- `entrega_1/artifacts/p02/p02_dataset_profile.json`
- `entrega_1/artifacts/p02/p02_emotion_distribution.csv`
- `entrega_1/artifacts/p02/p02_labels_per_sample_distribution.csv`
