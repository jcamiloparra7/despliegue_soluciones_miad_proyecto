# P04 - Exploracion de datos (EDA)

## Proposito

Presentar el analisis descriptivo de GoEmotions que sustenta decisiones de modelado y diseno del servicio de inferencia emocional.

## 1) Alcance del analisis

Datos analizados: `data/go_emotions/simplified` en sus tres particiones (`train`, `validation`, `test`).

Analisis realizados:

1. integridad de particiones por ID;
2. perfil de longitud de texto;
3. comportamiento multietiqueta;
4. distribucion de clases;
5. co-ocurrencia de emociones.

## 2) Resultados

### 2.1 Integridad de particiones

- filas por split: `train=43,410`, `validation=5,426`, `test=5,427`;
- duplicados internos por ID: 0 en los tres splits;
- solapamiento entre splits: 0.

Resultado: no se observa leakage por identificador entre entrenamiento, validacion y prueba.

### 2.2 Estadisticas del texto

| Split | Char mean | Char mediana | Char p95 | Words mean | Words mediana | Words p95 |
|---|---:|---:|---:|---:|---:|---:|
| train | 68.40 | 65 | 131 | 12.84 | 12 | 24 |
| validation | 68.24 | 64 | 133 | 12.79 | 12 | 24.75 |
| test | 67.82 | 65 | 132 | 12.73 | 12 | 24 |

Resultado: comentarios cortos y estables entre particiones.

### 2.3 Comportamiento multietiqueta

Consolidado (`train+validation+test`):

- muestras: 54,263;
- etiquetas asignadas: 63,812;
- promedio de etiquetas por muestra: 1.176.

Resultado: problema multietiqueta con predominio de casos de una etiqueta.

### 2.4 Distribucion de clases

Etiquetas de mayor frecuencia:

1. `neutral` (17,772)
2. `admiration` (5,122)
3. `approval` (3,687)
4. `gratitude` (3,372)
5. `annoyance` (3,093)

Etiquetas de baja frecuencia: `grief`, `pride`, `relief`, `nervousness`.

Resultado: desbalance relevante entre clases.

### 2.5 Co-ocurrencias destacadas

Pares frecuentes:

1. `anger` + `annoyance` (348)
2. `admiration` + `gratitude` (341)
3. `admiration` + `approval` (298)
4. `approval` + `neutral` (258)
5. `confusion` + `curiosity` (253)

Resultado: asociaciones semanticas consistentes con el enfoque multietiqueta.

## 3) Implicaciones para modelado y API

1. la prediccion debe permitir mas de una etiqueta por muestra;
2. la evaluacion debe incluir metrica global y por clase;
3. se recomienda salida probabilistica para habilitar umbrales operativos.

## 4) Evidencia reproducible

Script:

- `entrega_1/scripts/p04_eda_go_emotions.py`

Notebook integrado:

- `entrega_1/notebooks/analisis_integrado_go_emotions.ipynb`

Artefactos:

- `entrega_1/artifacts/p04/p04_eda_summary.json`
- `entrega_1/artifacts/p04/p04_text_length_stats.csv`
- `entrega_1/artifacts/p04/p04_emotion_counts_by_split.csv`
- `entrega_1/artifacts/p04/p04_top_label_cooccurrence_pairs.csv`
