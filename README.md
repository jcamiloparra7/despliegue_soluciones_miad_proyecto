# Proyecto de despliegue MIAD - Entrega 1

Este repositorio contiene la **Entrega 1** del proyecto, enfocada en un servicio de **identificación de emociones en texto en inglés** usando GoEmotions, con alcance de **API + dashboard**.

## Estructura del repositorio

```text
.
├── data/
│   └── go_emotions/                  # Dataset y paper base
└── entrega_1/
    ├── docs/
    │   ├── steps/                    # P01 ... P06
    │   ├── reports/                  # Reporte principal y reporte de equipo
    │   ├── mockups/                  # Mockup del prototipo API + dashboard
    │   ├── checklists/               # Matriz de verificación de entrega
    │   ├── requirements/             # Enunciado original de la entrega
    │   └── planning/                 # Plan consolidado (md/xlsx)
    ├── notebooks/
    │   └── analisis_integrado_go_emotions.ipynb
    ├── scripts/                      # Scripts locales de soporte (gitignored)
    └── artifacts/                    # Salidas generadas (gitignored)
```

## Documento principal de entrega

- `entrega_1/docs/reports/reporte_entrega_1.md`

## Notebook integrado

- `entrega_1/notebooks/analisis_integrado_go_emotions.ipynb`

Este notebook integra:

1. Caracterización de datos (P02)
2. EDA (P04)
3. Línea base predictiva (P05)

## Reproducción rápida (local)

1. Asegurar disponibilidad de datos en `data/go_emotions/`.
2. Abrir y ejecutar:
   - `entrega_1/notebooks/analisis_integrado_go_emotions.ipynb`
3. Verificar resultados en:
   - `entrega_1/artifacts/p02/`
   - `entrega_1/artifacts/p04/`
   - `entrega_1/artifacts/p05/`

## Dashboard EmotionAI — Integración con la API

El dashboard de Streamlit consume la API de clasificación de emociones
desplegada en AWS, utilizando el modelo ganador seleccionado mediante
los experimentos registrados en MLflow.

### Ejecución local

Desde la raíz del repositorio, con el entorno virtual activo:

```bash
pip install -r requirements.txt
```

Configurar la dirección de la API:

```bash
export EMOTIONAI_API_URL="http://54.210.54.7:8000"
```

Ejecutar el dashboard:

```bash
streamlit run entrega_2/dashboard/app.py
```

Acceder desde el navegador a http://localhost:8501.

La API debe estar disponible. Si cambia la dirección IP de AWS,
se debe actualizar la variable EMOTIONAI_API_URL.

### Funcionalidades

- Exploración del conjunto de datos GoEmotions.
- Clasificación de emociones mediante la API.
- Visualización del desempeño del modelo ganador.
- Monitoreo y auditoría de inferencias.

### Modelo desplegado y validado

- Modelo: TF-IDF + LinearSVC calibrado (C=0.75).
- Versión: p06-rerun-20261122.
- Clasificación multietiqueta de 28 emociones.
- F1 Micro en prueba: 0.5578.
- F1 Macro en prueba: 0.4252.

### Trazabilidad

Las inferencias exitosas se almacenan localmente en
`entrega_2/dashboard/data/historial_inferencias_api.csv`.

Este archivo está excluido del control de versiones para evitar
la publicación de los mensajes analizados.