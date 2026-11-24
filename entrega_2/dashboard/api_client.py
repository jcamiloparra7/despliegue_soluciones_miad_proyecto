import os
import requests


API_BASE_URL = os.getenv(
    "EMOTIONAI_API_URL",
    "http://54.210.54.7:8000"
).rstrip("/")


def verificar_api():
    respuesta = requests.get(
        f"{API_BASE_URL}/health",
        timeout=10
    )
    respuesta.raise_for_status()
    return respuesta.json()


def predecir_emociones(texto):
    respuesta = requests.post(
        f"{API_BASE_URL}/predict",
        json={"texts": [texto]},
        timeout=30
    )
    respuesta.raise_for_status()
    return respuesta.json()