
import json
import pandas as pd
import altair as alt
from pathlib import Path

import streamlit as st
import requests
from api_client import predecir_emociones

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
        / "p06"
        / "p06_experiment_results.json"
    )

    with open(ruta, "r", encoding="utf-8") as archivo:
        resultados = json.load(archivo)

    return next(
        experimento
        for experimento in resultados["experiments"]
        if experimento["experiment_name"]
        == "tfidf_calibrated_linearsvc_c_0.75"
    )

    
@st.cache_data
def cargar_desempeno_por_emocion():
    ruta = (
        PROJECT_ROOT
        / "ml_app"
        / "artifacts"
        / "p06"
        / "p06_winning_model_per_class_f1_test.csv"
    )

    return pd.read_csv(ruta)



def registrar_inferencia(texto, emociones, probabilidades, modelo, version):
    import csv
    from datetime import datetime

    ruta = (
        PROJECT_ROOT
        / "entrega_2"
        / "dashboard"
        / "data"
        / "historial_inferencias_api.csv"
    )

    registro = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "texto": texto,
        "emociones_detectadas": ", ".join(emociones),
        "probabilidades": json.dumps(probabilidades),
        "modelo": modelo,
        "version": version
    }

    ruta.parent.mkdir(parents=True, exist_ok=True)
    existe = ruta.exists() and ruta.stat().st_size > 0

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
            try:
                with st.spinner("Consultando la API de EmotionAI..."):
                    respuesta = predecir_emociones(
                        texto_usuario.strip()
                    )

                prediccion = respuesta["predictions"][0]
                emociones = prediccion["predicted_labels"]
                probabilidades = prediccion["scores"]
                modelo = respuesta["model_name"]
                version = respuesta["model_version"]

            except requests.exceptions.RequestException:
                st.error(
                    "No fue posible obtener una respuesta de la API. "
                    "Verifica la disponibilidad del servicio."
                )

            except (KeyError, IndexError, TypeError, ValueError):
                st.error(
                    "La API devolvió una respuesta con un formato inesperado."
                )

            else:
                st.subheader("Emociones detectadas")

                for emocion in emociones:
                    probabilidad = probabilidades[emocion]
                    st.write(
                        f"**{emocion} — {probabilidad:.2%}**"
                    )
                    st.progress(float(probabilidad))

                st.caption(
                    f"Modelo: {modelo} | Versión: {version}"
                )

                st.caption(
                    "Los porcentajes representan probabilidades "
                    "estimadas, no precisión garantizada."
                )

                st.divider()

                st.subheader("Cinco emociones más probables")

                st.write(
                    "Estas son las cinco emociones con mayor "
                    "probabilidad estimada por el modelo."
                )

                top_emociones = sorted(
                    probabilidades.items(),
                    key=lambda elemento: elemento[1],
                    reverse=True
                )[:5]

                for emocion, probabilidad in top_emociones:
                    estado = (
                        "Detectada"
                        if emocion in emociones
                        else "No detectada"
                    )

                    st.write(
                        f"**{emocion} — {probabilidad:.2%}** "
                        f"({estado})"
                    )

                    st.progress(float(probabilidad))

                st.caption(
                    "Las emociones no detectadas se muestran "
                    "únicamente como información complementaria."
                )

                try:
                    registrar_inferencia(
                        texto=texto_usuario.strip(),
                        emociones=emociones,
                        probabilidades=probabilidades,
                        modelo=modelo,
                        version=version
                    )
                except OSError:
                    st.warning(
                        "La predicción fue exitosa, pero no fue "
                        "posible guardar el registro de auditoría."
                    )


elif seccion == "Desempeño del modelo":
    resultados = cargar_resultados_modelo()

    st.subheader("Indicadores de desempeño del modelo ganador")

    st.caption(
        "Modelo: TF-IDF + LinearSVC calibrado (C=0.75) | "
        "Versión: p06-rerun-20261122 | "
        "Umbral de clasificación: 30 %"
    )

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
        "Resultados del modelo ganador OneVsRest "
        "(LinearSVC calibrado con TF-IDF y C=0.75). "
        "El modelo fue seleccionado entre 20 configuraciones "
        "evaluadas en MLflow, utilizando F1 Micro en validación "
        "como criterio principal."
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

    emociones_criticas = desempeno.nsmallest(5, "f1_test")

    if not emociones_criticas.empty:
        nombres = ", ".join(
            emociones_criticas["emotion"].tolist()
        )

        st.warning(
            f"Emociones con menor F1: {nombres}. "
            "Estas categorías presentan mayores dificultades "
            "de clasificación y requieren especial atención "
            "al interpretar las predicciones del modelo."
        )


    st.subheader("Análisis de las cinco emociones con menor F1")

    tabla_criticas = emociones_criticas[
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

        **Implicación para EmotionAI:** emociones como *disappointment*,
        *annoyance* y *disapproval* presentan un bajo desempeño.
        Esta limitación es especialmente relevante en escenarios
        de atención al cliente, donde identificar señales de
        insatisfacción puede contribuir a orientar la respuesta
        del chatbot.

        Estos resultados corresponden a una evaluación
        académica sobre GoEmotions y no constituyen
        una validación del desempeño en conversaciones
        reales de servicio al cliente.
        """
    )


    st.caption(
        "Fuente: evaluación del modelo ganador en el conjunto de prueba de GoEmotions. "
        "Ninguna de las 28 emociones presenta un F1 igual a cero. "
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
        / "historial_inferencias_api.csv"
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
                    "modelo",
                    "version"
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



        st.altair_chart(
            grafica_auditoria,
            width="stretch"
        )

        st.caption(
            "Probabilidades estimadas para las 28 emociones, "
            "obtenidas directamente de la API de EmotionAI. "
            "Las emociones detectadas corresponden a la "
            "clasificación realizada por el modelo desplegado."
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

