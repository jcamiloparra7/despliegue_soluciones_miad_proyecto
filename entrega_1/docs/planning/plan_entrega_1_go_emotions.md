# Plan de construcción de entregables - Proyecto (GoEmotions)

Este plan se construyó a partir de:
- `entrega_1/docs/requirements/dsa_proy_s2.md` (instrucciones de la entrega)
- `data/go_emotions/README.md` (estructura de datos)
- `data/go_emotions/dataset_paper.pdf` (fundamento técnico)

## 1) Insumos leídos y hallazgos clave

| Fuente | Hallazgo clave | Implicación para la entrega |
|---|---|---|
| data/go_emotions/README.md | Versión simplified con splits fijos: train 43,410; validation 5,426; test 5,427. | Permite una EDA reproducible y un baseline inicial con particiones definidas. |
| data/go_emotions/README.md | Versión raw con 211,225 filas y etiquetas por columna (multietiqueta). | Útil para análisis más profundo; para semana 2-3 conviene iniciar con simplified. |
| data/go_emotions/dataset_paper.pdf | GoEmotions: 58k comentarios de Reddit, 27 emociones + neutral. | Sustenta el contexto del problema y la formulación de la pregunta de negocio. |
| data/go_emotions/dataset_paper.pdf | Baseline reportado: BERT con F1 promedio ~0.46 en taxonomía fina. | Define expectativa realista de desempeño y métrica de referencia. |
| data/go_emotions/dataset_paper.pdf | Anotación con 3 evaluadores (y 2 adicionales cuando no hay acuerdo). | Apoya discusión de calidad de etiquetas y confiabilidad de datos. |
| Descarga y validación local | Dataset descargado con `hf download` y splits validados en parquet. | Permite ejecutar caracterización, EDA y baseline de forma reproducible. |

## 2) Plan refactorizado a 6 pasos

| ID | Fase | Entregable asociado | Paso | Actividades concretas | Evidencia/soporte | Dependencias | Salida esperada | Tiempo estimado (horas) | Semana objetivo |
|---|---|---|---|---|---|---|---|---|---|
| P01 | Equipo y planeación | Organización del proyecto | Definir roles, acuerdos y cronograma interno. | Asignar roles por integrante, establecer tablero de tareas y cadencia de seguimiento. | Acta de alineación y tablero inicial. | Ninguna | Equipo alineado | 2 | Semana 2 |
| P02 | Datos y caracterización | Conjuntos de datos a emplear | Habilitar datos y caracterizar estructura/calidad inicial. | Descargar dataset, validar splits parquet, perfilar esquema, distribución y multietiqueta. | Documento técnico + artefactos reproducibles de perfil. | P01 | Datos listos y documentados | 6 | Semana 2 |
| P03 | Problema y alcance | Problema + pregunta de negocio + alcance | Consolidar problema, pregunta principal y límites del MVP. | Definir caso de uso de API, restricciones (idioma/dominio), riesgos y criterios de éxito. | Documento de formulación y alcance. | P01, P02 | Dirección del proyecto cerrada | 4 | Semana 2 |
| P04 | Exploración de datos | Exploración de los datos | Ejecutar EDA orientada al caso de uso. | Analizar longitudes de texto, desbalance, co-ocurrencias y consistencia de splits. | Informe EDA + artefactos reproducibles. | P02, P03 | Hallazgos descriptivos accionables | 6 | Semana 2-3 |
| P05 | Línea base predictiva | Fundamento predictivo del prototipo | Entrenar y evaluar baseline multietiqueta. | Entrenar baseline, seleccionar umbral en validación y reportar métricas globales/por clase. | Documento baseline + resultados reproducibles. | P04 | Referencia predictiva inicial | 5 | Semana 3 |
| P06 | Prototipo y cierre de entrega | Mockup + reportes + envío | Consolidar materiales finales y control de calidad. | Completar mockup, reporte de 10 páginas, reporte de equipo y checklist de envío. | Paquete final de entrega con soportes. | P01, P03, P04, P05 | Entrega 1 lista para envío | 8 | Semana 3 |

## 3) Nota operativa importante

- Si en `data/go_emotions/` aparecen punteros Git LFS en lugar de archivos completos, usar la descarga por Hugging Face (`hf download ...`) y volver a validar lectura parquet.
