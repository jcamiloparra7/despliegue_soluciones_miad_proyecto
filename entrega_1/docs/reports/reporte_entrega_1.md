# Entrega 1 - Proyecto de despliegue de solucion analitica

## 1. Problema y contexto

Los sistemas conversacionales modernos suelen responder correctamente al contenido de una consulta, pero no siempre ajustan su tono segun el estado emocional del usuario. En escenarios de soporte, atencion o acompanamiento, esto puede traducirse en respuestas percibidas como frias, poco empaticas o inadecuadas.

Este proyecto propone abordar esa brecha mediante una API de identificacion de emociones en texto y un dashboard de exploracion de metricas, orientados al caso de uso de calibracion de respuesta en chatbots en idioma ingles.

La propuesta se alinea con el objetivo del curso de construir un prototipo funcional de producto basado en datos, integrando componente descriptivo (exploracion), componente predictivo (modelo supervisado) y diseno de servicio (API + mockup de tablero).

## 2. Pregunta de negocio y alcance

### 2.1 Pregunta principal

**¿Como puede un chatbot en ingles ajustar su tono y estrategia de respuesta usando una API de identificacion de emociones sobre el texto de entrada para mejorar la adecuacion emocional de la interaccion?**

### 2.2 Alcance del MVP

Incluye:

- inferencia emocional sobre texto en ingles,
- salida multietiqueta con puntajes de confianza,
- dashboard para visualizacion de metricas y distribuciones.

No incluye:

- multilenguaje,
- modalidades no textuales (audio/video),
- implementacion de chatbot completo,
- diagnostico psicologico/clinico,
- garantias de desempeno fuera del dominio de entrenamiento.

## 3. Conjuntos de datos a emplear

Dataset base: **GoEmotions**.

- Configuracion `simplified` (principal): `train`, `validation`, `test`.
- Configuracion `raw` (complementaria): analisis de estructura y contexto de anotacion.

### 3.1 Resumen estructural

| Dataset / split | Filas | Columnas | Tamano aprox. |
|---|---:|---:|---:|
| `raw/train` | 211,225 | 37 | 24 MB |
| `simplified/train` | 43,410 | 3 | 2.6 MB |
| `simplified/validation` | 5,426 | 3 | 342 KB |
| `simplified/test` | 5,427 | 3 | 339 KB |

Hallazgos clave:

- muestras consolidadas (`simplified`): 54,263
- etiquetas asignadas: 63,812
- promedio etiquetas por muestra: 1.176
- integridad textual basica: sin nulos y sin vacios en los splits analizados.

## 4. Exploracion de datos

### 4.1 Integridad de particiones

- sin duplicados por ID dentro de cada split,
- sin solapamiento de IDs entre train/validation/test.

### 4.2 Perfil de texto

Comentarios cortos con comportamiento estable entre splits:

- longitud media en caracteres ~68,
- longitud media en palabras ~12.8.

### 4.3 Distribucion emocional

Emociones mas frecuentes (global):

1. `neutral` (17,772)
2. `admiration` (5,122)
3. `approval` (3,687)
4. `gratitude` (3,372)
5. `annoyance` (3,093)

Emociones raras: `grief`, `pride`, `relief`, `nervousness`.

### 4.4 Co-ocurrencias relevantes

Pares frecuentes:

- `anger` + `annoyance`
- `admiration` + `gratitude`
- `admiration` + `approval`

Interpretacion: la tarea requiere enfoque multietiqueta real, no solo monoclase.

## 5. Baseline predictivo inicial

Se implemento un baseline reproducible:

- TF-IDF (unigramas y bigramas),
- One-vs-Rest con `SGDClassifier (log_loss)`,
- seleccion de umbral global sobre validacion.

### 5.1 Umbral seleccionado

- umbral optimo: **0.20**
- mejor `f1_micro` en validacion: **0.551555**

### 5.2 Metricas agregadas

| Split | F1 micro | F1 macro | F1 weighted | Precision micro | Recall micro |
|---|---:|---:|---:|---:|---:|
| validation | 0.551555 | 0.406550 | 0.520298 | 0.523904 | 0.582288 |
| test | 0.555169 | 0.412434 | 0.525910 | 0.522133 | 0.592669 |

Interpretacion:

- desempeño estable entre validacion y prueba,
- brecha micro vs macro consistente con desbalance de clases,
- margen claro de mejora para clases raras.

## 6. Maqueta (mockup) del prototipo

Se diseno un mockup funcional del flujo:

`Fuente de texto -> API de emociones -> Modelo -> Dashboard de metricas`

Documento asociado:

- `entrega_1/docs/mockups/mockup_api_dashboard_emociones.md`

Incluye:

- flujo de componentes API + dashboard,
- contrato de API (`POST /v1/emotions/predict`),
- esquema de tablero para visualizacion de resultados.

## 7. Repositorios y soportes

### 7.1 Codigo y datos (estado documental)

- Estructura de proyecto y entregables en: `entrega_1/`
- Scripts reproducibles en: `entrega_1/scripts/`
- Notebook integrado de analisis y modelado en: `entrega_1/notebooks/analisis_integrado_go_emotions.ipynb`
- Resultados de ejecucion en: `entrega_1/artifacts/`

### 7.2 Evidencias incluidas

- Caracterizacion de datos: `artifacts/p02/`
- Exploracion de datos: `artifacts/p04/`
- Baseline predictivo: `artifacts/p05/`

## 8. Riesgos y limitaciones

- sesgo de dominio (datos de Reddit),
- desbalance de clases,
- transferibilidad limitada a contextos distintos del entrenamiento,
- salida del modelo como señal auxiliar y no diagnostica.

## 9. Siguientes pasos

1. crear un mvp de la api de identificacion de emociones reutilizando el modelo baseline multietiqueta ya entrenado como motor de inferencia;
2. crear los mockups de dashboards para exploracion de metricas, distribucion de emociones y monitoreo de confianza de prediccion.
