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
