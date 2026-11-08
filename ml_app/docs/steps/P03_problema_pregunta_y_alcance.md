# P03 - Problema, pregunta de negocio y alcance

## Proposito

Definir el problema de negocio, la pregunta principal y los limites operativos del MVP que sera evaluado en la Entrega 1.

## 1) Formulacion del problema

En sistemas conversacionales, la respuesta semantica puede ser correcta pero emocionalmente inadecuada. Esta brecha reduce empatia percibida y calidad de interaccion.

El proyecto aborda esta necesidad mediante un servicio API capaz de identificar emociones en texto y entregar una señal util para calibrar la respuesta del chatbot.

## 2) Pregunta de negocio

**¿Como puede un chatbot en ingles ajustar su tono y estrategia de respuesta usando una API de identificacion de emociones sobre el texto de entrada para mejorar la adecuacion emocional de la interaccion?**

Subpreguntas:

1. Que emociones resultan mas utiles para ajustar estrategia de respuesta.
2. Que umbrales de confianza deben activar ajustes automaticos.
3. En que casos conviene mantener respuestas neutras o conservadoras.

## 3) Formulacion analitica

- **Entrada:** texto en ingles (`text`).
- **Salida:** una o varias emociones con puntajes de confianza.
- **Tipo de tarea:** clasificacion multietiqueta.
- **Taxonomia:** 27 emociones + neutral (GoEmotions).

## 4) Justificacion tecnica

- El dataset GoEmotions ofrece granularidad fina y esta alineado con el idioma del caso de uso.
- El paper reporta la dificultad de la tarea en taxonomia fina y valida que un MVP debe tratar la salida como señal probabilistica.
- Las limitaciones de sesgo y dominio reportadas en README/paper se incorporan como restriccion explicita de uso.

## 5) Alcance del MVP

### Incluye

- servicio API para inferencia emocional en ingles;
- dashboard para exploracion de metricas y distribuciones;
- evidencia descriptiva y predictiva (EDA + baseline).

### No incluye

- multilenguaje;
- modalidades no textuales;
- implementacion de chatbot o interfaz conversacional completa;
- uso diagnostico psicologico o clinico;
- garantias de rendimiento fuera del dominio de entrenamiento.

## 6) Restricciones y criterios de evaluacion

Restricciones:

1. idioma ingles;
2. dominio de entrenamiento basado en comentarios;
3. salida interpretada como apoyo a decision, no como verdad absoluta.

Criterios de evaluacion de este componente:

- claridad de problema y pregunta;
- consistencia entre alcance y datos disponibles;
- trazabilidad hacia diseno de EDA, baseline y mockup.
