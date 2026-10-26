# ENTREGA 1: Despliegue de Soluciones

## 1. PROBLEMA Y CONTEXTO

En la interacción humano-máquina a través de canales digitales, los sistemas conversacionales automatizados conocidos como chatbots, resuelven eficazmente consultas funcionales o transaccionales. Sin embargo, presentan una limitación crítica: no pueden comprender el estado emocional del interlocutor. La incapacidad de modular el tono, la urgencia y el estilo de respuesta ante la frustración, la confusión, el entusiasmo o la gratitud del usuario genera interacciones percibidas como impersonales, rígidas y desarticuladas del contexto humano.

En entornos de soporte técnico, gestión de clientes y servicios de asistencia, una respuesta desalineada emocionalmente genera consecuencias tangibles: incrementa la fricción y el descontento del usuario ante respuestas estándar en momentos de frustración, fuerza el escalamiento innecesario a canales humanos aumentando los costos operativos, y eleva la tasa de abandono debido a la falta de contención y empatía en puntos críticos de la interacción.

Los métodos tradicionales de análisis de texto basados en diccionarios de palabras clave o reglas heurísticas fracasan al intentar descifrar emociones sutiles en lenguaje informal; no consideran la ambigüedad, el contexto semántico, ni la coexistencia de estados anímicos superpuestos.

Para resolver esta brecha se requiere un modelo predictivo supervisado de clasificación multietiqueta. Este enfoque permite inferir probabilísticamente la presencia de un espectro granular de emociones a partir de representaciones densas del texto, entregando puntuaciones de confianza por categoría. Dicha salida estructurada es la única que permite a los motores de diálogo ejecutar reglas de decisión de manera robusta y en tiempo real.

---

## 2. PREGUNTA DE NEGOCIO Y ALCANCE DEL PROYECTO

> **¿De qué manera un servicio API de clasificación supervisada multietiqueta de emociones en textos en inglés puede identificar los estados emocionales de los usuarios y proporcionar información que facilite la adaptación del tono y la estrategia de respuesta de un chatbot?**

### Articulación de la solución:
* **Componente predictivo (API):** Resuelve la pregunta operacional clasificando el texto entrante en tiempo real y retornando las probabilidades por categoría para calibrar la respuesta del chatbot.
* **Componente descriptivo (Dashboard):** Resuelve la dimensión estratégica permitiendo auditar la distribución histórica de emociones, identificar fricciones recurrentes y monitorear la confiabilidad del modelo en producción.

### Componentes incluidos:
* Procesamiento de texto no estructurado en idioma inglés proveniente de mensajes directos de usuarios.
* Clasificador multietiqueta que estima probabilidades sobre un catálogo granular de emociones y la categoría neutral.
* Endpoint empaquetado que recibe cargas útiles en JSON y devuelve las emociones predichas con sus respectivos puntajes de confianza.
* Interfaz para visualizaciones descriptivas y un módulo de pruebas ad-hoc que consume la API en tiempo real.
* Contenedorización integral con Docker (API y tablero), control de versiones de código mediante Git y versionado de datos y artefactos con DVC.

### Componentes excluidos:
* Restringido exclusivamente al inglés, no se admiten otros idiomas.
* Sin soporte para audio, señales de voz, imágenes o expresiones faciales.
* Excluye la generación de lenguaje natural (NLG) y la gestión integral del diálogo del chatbot.
* No ofrece diagnósticos psicológicos ni aplicaciones en el ámbito de la salud mental.
* Sin garantías de desempeño ante léxico técnico o dominios cerrados ajenos al conjunto de entrenamiento.

---

## 3. CONJUNTOS DE DATOS A EMPLEAR

Para el desarrollo del prototipo se emplea el conjunto de datos de referencia **GoEmotions** (Google Research), el cual recopila comentarios de Reddit anotados manualmente sobre una taxonomía granular de 27 categorías emocionales más una etiqueta neutral. Los datos cuentan con disponibilidad inmediata a través del ecosistema de *Hugging Face Datasets* y repositorios abiertos, facilitando su gobierno y versionado mediante DVC. 

Se gestionan dos configuraciones del conjunto de datos:
1. **Configuración `simplified`:** Contiene las instancias filtradas donde existe consenso entre anotadores (al menos dos coincidencias). Es la fuente directa para el entrenamiento, validación y prueba de los modelos supervisados. 
2. **Configuración `raw`:** Registra las 211,225 anotaciones individuales desagregadas por evaluador, con metadatos contextuales y enlaces a identificadores de Reddit. Se emplea para auditoría de calidad, análisis de concordancia inter-anotador y evaluación de casos límite. 

El particionamiento oficial adoptado para el prototipo garantiza la reproducibilidad experimental:

| Configuración / Partición | Registros | Variables | Tamaño aprox. | Propósito en el proyecto |
| :--- | ---: | ---: | ---: | :--- |
| `raw/train` | 211,225 | 37 | 24.0 MB | Contexto de anotación y auditoría de consenso. |
| `simplified/train` | 43,410 | 3 | 2.6 MB | Entrenamiento del modelo supervisado. |
| `simplified/validation` | 5,426 | 3 | 342 KB | Calibración de umbral y selección de hiperparámetros. |
| `simplified/test` | 5,427 | 3 | 339 KB | Evaluación final ciega de rendimiento. |

**Atributos clave del conjunto simplificado:**
* `text` (string): Contenido textual del mensaje del usuario (en inglés).
* `labels` (list[int]): Vector de índices de las emociones asignadas por consenso.
* `id` (string): Identificador unívoco del comentario original de Reddit.

En la configuración `simplified`, el conjunto consolidado suma **54,263 muestras** que contienen **63,812 asignaciones de etiquetas**, lo que representa un promedio de 1.176 etiquetas por texto. Las tres particiones presentan integridad total: sin valores nulos, registros corruptos ni cadenas vacías. 

---

## 4. REPOSITORIO GIT EN USO PARA EL CÓDIGO

![Evidencia Repositorio Git](https://github.com/jcamiloparra7/despliegue_soluciones_miad_proyecto/assets/IMAGE_PLACEHOLDER_1)
*Enlace:* [https://github.com/jcamiloparra7/despliegue_soluciones_miad_proyecto/tree/main](https://github.com/jcamiloparra7/despliegue_soluciones_miad_proyecto/tree/main)

---

## 5. REPOSITORIO DVC EN USO PARA LOS DATOS

![Evidencia DVC 1](https://github.com/jcamiloparra7/despliegue_soluciones_miad_proyecto/assets/IMAGE_PLACEHOLDER_2)
![Evidencia DVC 2](https://github.com/jcamiloparra7/despliegue_soluciones_miad_proyecto/assets/IMAGE_PLACEHOLDER_3)
![Evidencia DVC 3](https://github.com/jcamiloparra7/despliegue_soluciones_miad_proyecto/assets/IMAGE_PLACEHOLDER_4)

---

## 6. EXPLORACIÓN DE LOS DATOS 

### Integridad de particiones y perfil del texto
La validación de consistencia confirmó la ausencia de identificadores duplicados al interior de cada conjunto y una intersección nula entre las particiones de entrenamiento, validación y prueba, garantizando la inexistencia de *data leakage*. Los textos presentan un comportamiento altamente homogéneo y estable entre divisiones, con una longitud media de 68 caracteres y 12.8 palabras. 

### Desbalance de clases y multietiquetas
La distribución de etiquetas evidencia un marcado desbalance natural en las interacciones cotidianas. La categoría predominante es `neutral` (17,772 instancias), seguida por expresiones positivas frecuentes como `admiration` (5,122), `approval` (3,687) y `gratitude` (3,372), y una presencia significativa de malestar mediante `annoyance` (3,093). En contraste, categorías críticas de cola larga como `grief`, `pride`, `relief` y `nervousness` cuentan con un volumen sustancialmente menor. Desde la perspectiva de negocio, aunque las emociones negativas asociadas a fricción representen un porcentaje inferior, constituyen los momentos decisivos donde el chatbot debe ajustar su tono o derivar el caso a un agente humano.

![Frecuencia de Emociones](https://github.com/jcamiloparra7/despliegue_soluciones_miad_proyecto/assets/IMAGE_PLACEHOLDER_5)

Por otra parte, el análisis de co-ocurrencias reveló solapamientos semánticos recurrentes en pares como `anger` + `annoyance`, `admiration` + `gratitude` y `admiration` + `approval`. Esta coexistencia comprueba que los usuarios proyectan estados anímicos superpuestos, lo que descarta el uso de esquemas monoclase mutuamente excluyentes y confirma la necesidad técnica de un enfoque de clasificación multietiqueta supervisada.

![Heatmap de Coocurrencia](https://github.com/jcamiloparra7/despliegue_soluciones_miad_proyecto/assets/IMAGE_PLACEHOLDER_6)

### Línea base predictiva inicial
Se implementó un pipeline supervisado lineal sobre representaciones léxicas antes de abordar arquitecturas neuronales más complejas. 
El modelo base utiliza una representación TF-IDF con extracción de unigramas y bigramas, acoplada a una estrategia *One-vs-Rest* con un clasificador lineal. Dado el desbalance de las clases, se ejecutó una búsqueda en cuadrícula sobre el conjunto de validación para calibrar un umbral de decisión global, encontrando el punto óptimo en **0.20**, el cual favorece la recuperación de categorías minoritarias frente al corte estándar de 0.50 y alcanza un F1 micro de 0.5516.

**Desempeño del Baseline Inicial (TF-IDF + SGDClassifier, Umbral 0.20):**

| Partición | F1 micro | F1 macro | F1 weighted | Precisión micro | Sensibilidad micro |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validación** | 0.5516 | 0.4066 | 0.5203 | 0.5239 | 0.5823 |
| **Prueba (Test)** | **0.5552** | **0.4124** | **0.5259** | **0.5221** | **0.5927** |

![F1 por emoción en test](https://github.com/jcamiloparra7/despliegue_soluciones_miad_proyecto/assets/IMAGE_PLACEHOLDER_7)

---

## 7. MAQUETA MOCKUP DEL PROTOTIPO 

El prototipo busca facilitar la interpretación del estado emocional del usuario y proporcionar información que permita adaptar el tono y la estrategia de respuesta de un chatbot.

### SERVICIO DE INFERENCIA (API REST)
El servicio ejecuta inferencia multietiqueta simultánea para todas las clases que superen el umbral óptimo de 0.20, en caso de que ninguna lo alcance, asigna la categoría neutral o activa una bandera de baja certeza, operando empaquetado en una imagen Docker independiente para garantizar portabilidad y reproducibilidad.
* **Rol:** Microservicio desacoplado que expone el modelo supervisado multietiqueta para clasificar texto en inglés en tiempo real y retornar probabilidades por emoción.
* **Endpoint:** `POST /v1/emotions/predict` (HTTP / JSON)
* **Entrada (Request):**
  ```json
  {
    "text": "I feel overwhelmed and I don't know what to do.", 
    "language": "en"
  }
  ```
* **Salida (Response):**
  ```json
  {
    "language": "en", 
    "predictions": [
      {"label": "nervousness", "score": 0.71}, 
      {"label": "fear", "score": 0.42}
    ], 
    "top_label": "nervousness", 
    "model_version": "baseline_v1", 
    "threshold_policy": "global_0.20_with_fallback"
  }
  ```

### DASHBOARD DE MONITOREO
La interfaz propuesta integra cuatro componentes principales:
1. **Exploración de datos:** presenta indicadores generales del conjunto de datos GoEmotions y visualizaciones descriptivas sobre la distribución, frecuencia y coexistencia de emociones en los mensajes.
2. **Clasificación de emociones:** permite ingresar un mensaje en inglés y enviarlo a una API que procesa el texto mediante el modelo predictivo.
3. **Resultados de predicción:** muestra las emociones identificadas y sus respectivos puntajes de confianza, permitiendo reconocer múltiples emociones en un mismo mensaje.
4. **Estrategia de respuesta:** propone orientaciones sobre el tono y el enfoque de la interacción a partir de reglas predefinidas asociadas con las emociones detectadas, sin generar automáticamente respuestas conversacionales.

![Mockup del Dashboard](https://github.com/jcamiloparra7/despliegue_soluciones_miad_proyecto/assets/IMAGE_PLACEHOLDER_8)

El diseño se plantea inicialmente como una interfaz de una sola pantalla, con componentes modulares que facilitan su posterior implementación y su integración con la API del modelo.
*Nota: Los valores presentados en las visualizaciones y los resultados predictivos del mockup son ilustrativos y no corresponden a resultados experimentales del modelo ni a estadísticas calculadas sobre el conjunto de datos.*

---

## 8. REPORTE DE TRABAJO EN EQUIPO

* **Distribución del esfuerzo:** Equitativa (33.3% por integrante)
* **Periodo:** Semanas 1 a 3

### Integrantes

**Juan Camilo Parra — Tech Lead**
* Definición y liderazgo de la arquitectura funcional del prototipo, integrando el flujo entre el modelo predictivo, la API de inferencia y el tablero de visualización.
* Toma de decisiones técnicas sobre la formulación multietiqueta supervisada, métricas de evaluación y especificaciones del servicio de inferencia.
* Supervisión y control de coherencia metodológica de la solución, y diseño funcional de la maqueta del sistema.

**Aleisy Johana Bareño — Project Coordinator & Documentation Lead**
* Planificación operativa, cronograma y coordinación de actividades a lo largo de las semanas 1, 2 y 3.
* Redacción y estructuración del planteamiento del problema, su justificación en entornos de CRM/soporte, y delimitación del alcance funcional del MVP.
* Consolidación documental y control de calidad, garantizando el apego a la rúbrica y el cumplimiento del límite de 10 páginas.

**Santiago Montaño — Data & Modeling Engineer**
* Descarga, verificación de integridad estructural y control de disponibilidad de las particiones del conjunto de datos GoEmotions (`raw` y `simplified`).
* Desarrollo del análisis cuantitativo y descriptivo sobre textos, longitud de comentarios, frecuencias y correlación/co-ocurrencia multietiqueta.
* Implementación y ejecución de scripts del baseline, optimización del umbral de decisión y análisis de resultados por clase y agregados (micro, macro y ponderado).

> *El equipo trabajó con separación de responsabilidades y revisiones cruzadas entre lo técnico y lo documental. La integración final se realizó con trazabilidad de archivos y resultados reproducibles, buscando consistencia entre problema de negocio, análisis descriptivo y baseline predictivo.*