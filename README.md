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

### Despliegue del dashboard con Docker en AWS

El dashboard se ejecuta en un contenedor Docker independiente,
alojado en la misma instancia EC2 que la API de clasificación.

Desde la raíz del repositorio, construir la imagen:

```bash
sudo docker build -f entrega_2/dashboard/Dockerfile -t emotionai-dashboard:1.0 .
```

Iniciar el contenedor:

```bash
sudo docker run -d \
  --name emotionai-dashboard \
  -p 8501:8501 \
  -e EMOTIONAI_API_URL=http://54.210.54.7:8000 \
  --restart unless-stopped \
  emotionai-dashboard:1.0
```

Verificar el estado del contenedor:

```bash
sudo docker ps --filter name=emotionai-dashboard
curl --fail http://localhost:8501/_stcore/health
```

Direcciones de acceso:

- Dashboard: http://54.210.54.7:8501
- API: http://54.210.54.7:8000
- Documentación de la API: http://54.210.54.7:8000/docs

El acceso al dashboard requiere autorización del puerto TCP 8501
en el grupo de seguridad de EC2. Actualmente está restringido
a la dirección IP autorizada para las pruebas.

La imagen utiliza Python 3.13 e incluye únicamente las dependencias
y los archivos de datos requeridos por el dashboard.

Se validaron satisfactoriamente los cuatro módulos en AWS,
incluyendo la clasificación mediante la API y el registro de
inferencias dentro del contenedor.

El historial se conserva mientras exista el contenedor.
Para garantizar su persistencia al reemplazarlo, se requiere
configurar un volumen de Docker.

Si cambia la IP pública de EC2, se deben actualizar las
direcciones de acceso y la variable EMOTIONAI_API_URL.
