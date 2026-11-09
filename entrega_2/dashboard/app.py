
import json
import pandas as pd
import altair as alt
from pathlib import Path

import streamlit as st

# Configuración general
st.set_page_config(
    page_title="EmotionAI | Emotion Classification",
    page_icon="🧠",
    layout="wide"
)

# Ubicación de los archivos del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROFILE_PATH = (
    PROJECT_ROOT
    / "ml_app"
    / "artifacts"
    / "p02"
    / "p02_dataset_profile.json"
)

# Cargar los datos
@st.cache_data
def cargar_perfil():
    with open(PROFILE_PATH, "r", encoding="utf-8") as archivo:
        return json.load(archivo)
@st.cache_data

def cargar_distribucion():
    ruta = (
        PROJECT_ROOT
        / "ml_app"
        / "artifacts"
        / "p02"
        / "p02_emotion_distribution.csv"
    )
    return pd.read_csv(ruta)


@st.cache_data
def cargar_distribucion_por_particion():
    ruta = (
        PROJECT_ROOT
        / "ml_app"
        / "artifacts"
        / "p04"
        / "p04_emotion_counts_by_split.csv"
    )

    return pd.read_csv(ruta)


@st.cache_data
def cargar_resultados_modelo():
    ruta = (
        PROJECT_ROOT
        / "ml_app"
        / "artifacts"
        / "p05"
        / "p05_baseline_results.json"
    )

    with open(ruta, "r", encoding="utf-8") as archivo:
        return json.load(archivo)

    
@st.cache_data
def cargar_desempeno_por_emocion():
    ruta = (
        PROJECT_ROOT
        / "ml_app"
        / "artifacts"
        / "p05"
        / "p05_per_class_f1_test.csv"
    )

    return pd.read_csv(ruta)


@st.cache_resource
def cargar_modelo_emotionai():
    import joblib

    ruta = (
        PROJECT_ROOT
        / "entrega_2"
        / "dashboard"
        / "models"
        / "emotionai_baseline.joblib"
    )

    return joblib.load(ruta)


def registrar_inferencia(texto, emociones, probabilidades, umbral):
    import csv
    from datetime import datetime

    ruta = (
        PROJECT_ROOT
        / "entrega_2"
        / "dashboard"
        / "data"
        / "historial_inferencias.csv"
    )

    indices = [
        i for i, probabilidad in enumerate(probabilidades)
        if probabilidad >= umbral
    ]

    if not indices:
        indices = [int(probabilidades.argmax())]

    registro = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "texto": texto,
        "emociones_detectadas": ", ".join(
            emociones[i] for i in indices
        ),
        "probabilidades": json.dumps({
            emociones[i]: round(float(probabilidades[i]), 6)
            for i in range(len(emociones))
        }),
        "umbral": umbral
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    existe = ruta.exists()

    with ruta.open("a", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=registro.keys()
        )

        if not existe:
            escritor.writeheader()

        escritor.writerow(registro)


# Encabezado
st.title("🧠 EmotionAI")
st.subheader("Emotion Classification for Chatbots")

st.write(
    "Prototipo de análisis y clasificación multietiqueta "
    "de emociones en textos en inglés."
)

st.divider()

# Navegación
st.sidebar.title("EmotionAI")

seccion = st.sidebar.radio(
    "Selecciona un módulo",
    [
        "Exploración de emociones",
        "Clasificación de emociones",
        "Desempeño del modelo",
        "Monitoreo y auditoría"
    ]
)

st.header(seccion)

# Módulo de exploración
if seccion == "Exploración de emociones":

    perfil = cargar_perfil()

    datos = perfil["derived"]

    total_mensajes = datos["total_simplified_rows"]
    total_etiquetas = datos["total_assigned_labels"]
    promedio = datos["avg_labels_per_sample"]
    total_emociones = len(datos["label_counts_total"])

    st.subheader("Indicadores generales de GoEmotions")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Mensajes", f"{total_mensajes:,}")
    col2.metric("Emociones", total_emociones)
    col3.metric("Etiquetas asignadas", f"{total_etiquetas:,}")
    col4.metric("Etiquetas por mensaje", f"{promedio:.2f}")

    st.caption(
        "Fuente: GoEmotions, configuración simplified. "
        "Indicadores consolidados de train, validation y test."
    )


    st.divider()
    st.subheader("Comparación de emociones por partición")

    st.write(
        "Compara la distribución relativa de emociones entre "
        "entrenamiento, validación y prueba."
    )

    datos_particiones = cargar_distribucion_por_particion()

    columnas = [
        "train_count",
        "validation_count",
        "test_count"
    ]

    # Convertir las columnas de particiones en filas
    datos_largos = datos_particiones.melt(
        id_vars="emotion",
        value_vars=columnas,
        var_name="particion",
        value_name="frecuencia"
    )

    # Calcular porcentajes dentro de cada partición
    datos_largos["porcentaje"] = (
        datos_largos["frecuencia"]
        / datos_largos.groupby("particion")["frecuencia"].transform("sum")
        * 100
    )

    # Nombres más fáciles de interpretar
    nombres = {
        "train_count": "Entrenamiento",
        "validation_count": "Validación",
        "test_count": "Prueba"
    }

    datos_largos["particion"] = (
        datos_largos["particion"].map(nombres)
    )

    emociones_seleccionadas = st.multiselect(
        "Selecciona las emociones que deseas comparar",
        options=datos_particiones["emotion"].tolist(),
        default=["neutral", "grief", "nervousness"]
    )

    datos_filtrados = datos_largos[
        datos_largos["emotion"].isin(emociones_seleccionadas)
    ]

    if datos_filtrados.empty:
        st.info("Selecciona al menos una emoción.")
    else:
        grafica_particiones = (
            alt.Chart(datos_filtrados)
            .mark_bar()
            .encode(
                x=alt.X(
                    "particion:N",
                    title="Partición",
                    sort=["Entrenamiento", "Validación", "Prueba"]
                ),
                y=alt.Y(
                    "porcentaje:Q",
                    title="Porcentaje de etiquetas (%)"
                ),
                color=alt.Color(
                    "emotion:N",
                    title="Emoción"
                ),
                xOffset="emotion:N",
                tooltip=[
                    alt.Tooltip("emotion:N", title="Emoción"),
                    alt.Tooltip("particion:N", title="Partición"),
                    alt.Tooltip("frecuencia:Q", title="Etiquetas"),
                    alt.Tooltip(
                        "porcentaje:Q",
                        title="Porcentaje",
                        format=".2f"
                    )
                ]
            )
            .properties(height=420)
        )

        st.altair_chart(grafica_particiones, width="stretch")

    st.caption(
        "Los porcentajes se calculan sobre el total de etiquetas "
        "asignadas en cada partición, no sobre el número de mensajes. "
        "Esto permite comparar conjuntos de diferentes tamaños."
    )
    
    st.divider()
    st.subheader("Distribución de emociones")

    distribucion = cargar_distribucion()


    vista = st.selectbox(
        "Selecciona una vista de las emociones",
        [
            "Todas las emociones",
            "Sin neutral",
            "Emociones minoritarias"
        ]
    )

    if vista == "Sin neutral":
        distribucion = distribucion[
            distribucion["emotion"] != "neutral"
        ].copy()

    elif vista == "Emociones minoritarias":
        distribucion = distribucion.nsmallest(
            10, "count"
        ).copy()


    grafica = (
        alt.Chart(distribucion)
        .mark_bar(color="#4776D0")
        .encode(
            x=alt.X(
                "count:Q",
                title="Número de etiquetas"
            ),
            y=alt.Y(
                "emotion:N",
                sort="-x",
                title="Emoción"
            ),
            tooltip=[
                alt.Tooltip("emotion:N", title="Emoción"),
                alt.Tooltip("count:Q", title="Frecuencia"),
                alt.Tooltip(
                    "percentage_of_assigned_labels:Q",
                    title="% de etiquetas",
                    format=".2f"
                )
            ]
        )
        .properties(height=650)
    )

    st.altair_chart(grafica, width="stretch")

    st.caption(
        "Frecuencia de cada emoción en GoEmotions simplified. "
        "Un mensaje puede tener múltiples etiquetas, por lo que "
        "las frecuencias no representan mensajes únicos."
    )


elif seccion == "Clasificación de emociones":
    st.subheader("Clasificador de emociones")

    st.write(
        "Ingresa un mensaje en inglés para identificar "
        "las emociones presentes en el texto."
    )

    texto_usuario = st.text_area(
        "Mensaje para analizar",
        placeholder="Example: I am really happy with your help!",
        height=150
    )

    clasificar = st.button(
        "Analizar emociones",
        type="primary"
    )


    if clasificar:
        if not texto_usuario.strip():
            st.warning("Ingresa un mensaje antes de analizarlo.")
        else:
            import numpy as np

            modelo_emotionai = cargar_modelo_emotionai()

            X = modelo_emotionai["vectorizer"].transform(
                [texto_usuario.strip()]
            )

            probabilidades = (
                modelo_emotionai["model"].predict_proba(X)[0]
            )

            umbral = modelo_emotionai["threshold"]
            emociones = modelo_emotionai["emotion_names"]

            indices = np.where(probabilidades >= umbral)[0]

            if len(indices) == 0:
                indices = [int(np.argmax(probabilidades))]

            indices = sorted(
                indices,
                key=lambda i: probabilidades[i],
                reverse=True
            )


            registrar_inferencia(
                texto=texto_usuario.strip(),
                emociones=emociones,
                probabilidades=probabilidades,
                umbral=umbral
            )


            st.subheader("Emociones detectadas")


            for i in indices:
                st.write(
                    f"**{emociones[i]} — {probabilidades[i]:.2%}**"
                )

                st.progress(
                    float(probabilidades[i])
                )

            st.caption(
                "Predicciones del modelo baseline entrenado "
                "con GoEmotions. Los porcentajes representan "
                "probabilidades estimadas, no precisión garantizada."
            )


            st.divider()
            st.subheader("Cinco emociones más probables")

            st.write(
                "Estas son las cinco emociones con mayor "
                "probabilidad estimada, independientemente "
                "de si superan el umbral de clasificación."
            )

            top_indices = np.argsort(probabilidades)[-5:][::-1]

            for i in top_indices:
                supera_umbral = probabilidades[i] >= umbral

                estado = (
                    "Detectada"
                    if supera_umbral
                    else "Por debajo del umbral"
                )

                st.write(
                    f"**{emociones[i]} — "
                    f"{probabilidades[i]:.2%}** "
                    f"({estado})"
                )

                st.progress(float(probabilidades[i]))

            st.caption(
                "Las emociones por debajo del umbral se muestran "
                "únicamente como información complementaria. "
                "No constituyen clasificaciones positivas."
            )


elif seccion == "Desempeño del modelo":
    resultados = cargar_resultados_modelo()

    st.subheader("Indicadores de desempeño del baseline")

    particion = st.radio(
        "Selecciona el conjunto de evaluación",
        ["Prueba", "Validación"],
        horizontal=True
    )

    clave = "test" if particion == "Prueba" else "validation"
    metricas = resultados["metrics"][clave]

    col1, col2, col3 = st.columns(3)

    col1.metric("F1 Micro", f"{metricas['f1_micro']:.4f}")
    col2.metric("F1 Macro", f"{metricas['f1_macro']:.4f}")
    col3.metric("F1 Weighted", f"{metricas['f1_weighted']:.4f}")

    col4, col5 = st.columns(2)

    col4.metric(
        "Precisión Micro",
        f"{metricas['precision_micro']:.4f}"
    )
    col5.metric(
        "Recall Micro",
        f"{metricas['recall_micro']:.4f}"
    )

    st.caption(
        "Resultados del modelo baseline OneVsRest "
        "(SGDClassifier con TF-IDF). "
        "El umbral de clasificación fue seleccionado "
        "utilizando F1 Micro en validación."
    )

    st.divider()
    st.subheader("Desempeño individual por emoción")

    st.write(
        "F1 de cada una de las 28 emociones evaluadas "
        "en el conjunto de prueba."
    )

    desempeno = cargar_desempeno_por_emocion()

    grafica_f1 = (
        alt.Chart(desempeno)
        .mark_bar()
        .encode(
            x=alt.X(
                "f1_test:Q",
                title="F1 en prueba",
                scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y(
                "emotion:N",
                sort="-x",
                title="Emoción"
            ),
            color=alt.condition(
                alt.datum.f1_test == 0,
                alt.value("#D9534F"),
                alt.value("#4776D0")
            ),
            tooltip=[
                alt.Tooltip("emotion:N", title="Emoción"),
                alt.Tooltip(
                    "f1_test:Q",
                    title="F1",
                    format=".4f"
                ),
                alt.Tooltip(
                    "support_test:Q",
                    title="Casos reales"
                ),
                alt.Tooltip(
                    "predicted_positive_test:Q",
                    title="Predicciones positivas"
                )
            ]
        )
        .properties(height=700)
    )

    st.altair_chart(grafica_f1, width="stretch")

    emociones_f1_cero = desempeno[
        desempeno["f1_test"] == 0
    ]

    if not emociones_f1_cero.empty:
        nombres = ", ".join(
            emociones_f1_cero["emotion"].tolist()
        )

        st.warning(
            f"Emociones con F1 igual a cero: {nombres}. "
            "El modelo no logró identificar correctamente "
            "casos positivos de estas emociones en el conjunto "
            "de prueba. Este resultado requiere analizar "
            "la cantidad de ejemplos y las predicciones realizadas."
        )


    st.subheader("Análisis de emociones con F1 igual a cero")

    tabla_criticas = emociones_f1_cero[
        [
            "emotion",
            "support_test",
            "predicted_positive_test",
            "f1_test"
        ]
    ].rename(
        columns={
            "emotion": "Emoción",
            "support_test": "Casos reales",
            "predicted_positive_test": "Predicciones positivas",
            "f1_test": "F1"
        }
    )

    st.dataframe(
        tabla_criticas,
        hide_index=True,
        width="stretch"
    )


    st.subheader("Interpretación de resultados")

    st.markdown(
        """
        El modelo presenta diferencias importantes en su capacidad
        para identificar las distintas emociones.

        **F1 Micro:** resume el desempeño global considerando
        conjuntamente las predicciones de todas las clases.

        **F1 Macro:** calcula el promedio del F1 de las 28 emociones,
        asignando el mismo peso a cada una.

        **F1 Weighted:** promedia el F1 de las emociones ponderándolo
        por su número de casos reales.

        La diferencia entre F1 Micro y F1 Macro evidencia que
        el desempeño agregado no representa por igual todas
        las emociones.

        **Implicación para EmotionAI:** emociones como *grief*
        y *nervousness* requieren especial atención, porque
        las dificultades para identificarlas podrían limitar
        la utilidad del modelo en determinados escenarios
        de atención al cliente.

        Estos resultados corresponden a una evaluación
        académica sobre GoEmotions y no constituyen
        una validación del desempeño en conversaciones
        reales de servicio al cliente.
        """
    )


    st.caption(
        "Fuente: evaluación del baseline en el conjunto de prueba. "
        "Las emociones con F1 igual a cero se identifican en rojo. "
        "Los resultados no corresponden a monitoreo en producción."
    )

elif seccion == "Monitoreo y auditoría":
    st.subheader("Historial local de inferencias")

    st.write(
        "Consulta los mensajes analizados por EmotionAI "
        "y las emociones identificadas por el modelo."
    )

    ruta_historial = (
        PROJECT_ROOT
        / "entrega_2"
        / "dashboard"
        / "data"
        / "historial_inferencias.csv"
    )

    if ruta_historial.exists():
        historial = pd.read_csv(ruta_historial)

        st.metric(
            "Mensajes analizados",
            len(historial)
        )

        st.dataframe(
            historial[
                [
                    "fecha_hora",
                    "texto",
                    "emociones_detectadas",
                    "umbral"
                ]
            ],
            width="stretch",
            hide_index=True
        )
        st.subheader("Auditoría individual de inferencias")

        indice_registro = st.selectbox(
            "Selecciona un mensaje para revisar",
            options=list(range(len(historial))),
            format_func=lambda i: (
                f"{historial.iloc[i]['fecha_hora']} — "
                f"{historial.iloc[i]['texto'][:70]}"
            )
        )

        st.divider()
        st.subheader("Distribución de emociones detectadas")

        conteo_emociones = (
            historial["emociones_detectadas"]
            .dropna()
            .str.split(", ")
            .explode()
            .value_counts()
            .reset_index()
        )

        registro_seleccionado = historial.iloc[indice_registro]

        probabilidades_registro = json.loads(
            registro_seleccionado["probabilidades"]
        )

        detalle_probabilidades = pd.DataFrame(
            [
                {
                    "Emoción": emocion,
                    "Probabilidad": float(probabilidad)
                }
                for emocion, probabilidad
                in probabilidades_registro.items()
            ]
        )

        detalle_probabilidades = detalle_probabilidades.sort_values(
            "Probabilidad",
            ascending=False
        )
        st.markdown("#### Probabilidades por emoción del mensaje seleccionado")
        grafica_auditoria = (
            alt.Chart(detalle_probabilidades)
            .mark_bar(color="#4776D0")
            .encode(
                x=alt.X(
                    "Probabilidad:Q",
                    scale=alt.Scale(domain=[0, 1])
                ),
                y=alt.Y(
                    "Emoción:N",
                    sort="-x"
                ),
                tooltip=[
                    "Emoción:N",
                    alt.Tooltip(
                        "Probabilidad:Q",
                        format=".2%"
                    )
                ]
            )
            .properties(height=650)
        )

        linea_umbral = (
    alt.Chart(pd.DataFrame({"Umbral": [0.2]}))
    .mark_rule(
        color="#C0392B",
        strokeDash=[6, 4],
        strokeWidth=2
    )
    .encode(x="Umbral:Q")
)

        st.altair_chart(
            grafica_auditoria + linea_umbral,
            width="stretch"
        )

        st.caption("Línea roja punteada: umbral de clasificación del 20 %.")

        st.caption(
            "Probabilidades estimadas para las 28 emociones. "
            "La clasificación utiliza el umbral registrado "
            "para cada inferencia."
        )

        conteo_emociones.columns = ["Emoción", "Frecuencia"]

        st.bar_chart(
            conteo_emociones,
            x="Emoción",
            y="Frecuencia"
        )

        st.caption(
            "Frecuencia de las emociones detectadas en las "
            "inferencias locales. Un mensaje puede tener "
            "múltiples emociones."
        )

        st.caption(
            "Historial de pruebas locales del prototipo académico. "
            "No corresponde a monitoreo en producción."
        )
    else:
        st.info(
            "Todavía no existen inferencias registradas."
        )

else:
    st.info(
        "Módulo en desarrollo. Próximamente incorporaremos "
        "las visualizaciones y funcionalidades correspondientes."
    )

