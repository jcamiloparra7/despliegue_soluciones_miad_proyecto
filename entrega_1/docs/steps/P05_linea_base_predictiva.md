# P05 - Linea base predictiva

## Proposito

Presentar una linea base reproducible de clasificacion multietiqueta para estimar emociones en texto y establecer referencia cuantitativa del MVP.

## 1) Configuracion metodologica

Modelo implementado:

- vectorizacion: TF-IDF (unigramas y bigramas);
- clasificador: One-vs-Rest con `SGDClassifier` (`loss=log_loss`);
- datos: `data/go_emotions/simplified` (`train`, `validation`, `test`);
- decision: umbral global con fallback por `argmax` para evitar predicciones vacias.

Hiperparametros principales:

- `ngram_range=(1,2)`
- `min_df=3`
- `max_features=50000`
- `alpha=1e-5`
- `max_iter=30`
- `random_state=42`

## 2) Seleccion de umbral

Se evaluo umbral global en validacion desde `0.10` hasta `0.90` con paso `0.05`, maximizando `f1_micro`.

- umbral seleccionado: **0.20**
- mejor `f1_micro` en validacion: **0.551555**

## 3) Resultados

### 3.1 Metricas agregadas

| Split | F1 micro | F1 macro | F1 weighted | Precision micro | Recall micro |
|---|---:|---:|---:|---:|---:|
| validation | 0.551555 | 0.406550 | 0.520298 | 0.523904 | 0.582288 |
| test | 0.555169 | 0.412434 | 0.525910 | 0.522133 | 0.592669 |

### 3.2 Desempeno por clase (resumen)

Etiquetas con mayor F1 en test: `gratitude`, `amusement`, `love`, `remorse`, `admiration`.

Etiquetas de menor F1 en test: `grief`, `nervousness`, `embarrassment`, `disappointment`, `relief`.

## 4) Interpretacion

1. El baseline muestra estabilidad entre validacion y prueba.
2. La diferencia entre `f1_micro` y `f1_macro` refleja el desbalance de clases observado en la exploracion.
3. Las clases de baja frecuencia presentan mayor dificultad de prediccion, por lo que se requieren ajustes posteriores por clase.

## 5) Trazabilidad con el caso de uso

La salida probabilistica del baseline habilita consumo por servicios externos (incluyendo chatbot como caso de uso) y alimenta las visualizaciones del dashboard definido en el alcance de P03.

## 6) Evidencia reproducible

Script:

- `entrega_1/scripts/p05_baseline_go_emotions.py`

Notebook integrado:

- `entrega_1/notebooks/analisis_integrado_go_emotions.ipynb`

Artefactos:

- `entrega_1/artifacts/p05/p05_baseline_results.json`
- `entrega_1/artifacts/p05/p05_threshold_scan.csv`
- `entrega_1/artifacts/p05/p05_per_class_f1_test.csv`
