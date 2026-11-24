FROM python:3.13-slim

WORKDIR /app

COPY entrega_2/dashboard/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY entrega_2/dashboard/app.py entrega_2/dashboard/app.py
COPY entrega_2/dashboard/api_client.py entrega_2/dashboard/api_client.py

COPY ml_app/artifacts/p02/p02_dataset_profile.json ml_app/artifacts/p02/
COPY ml_app/artifacts/p02/p02_emotion_distribution.csv ml_app/artifacts/p02/
COPY ml_app/artifacts/p04/p04_emotion_counts_by_split.csv ml_app/artifacts/p04/
COPY ml_app/artifacts/p06/p06_experiment_results.json ml_app/artifacts/p06/
COPY ml_app/artifacts/p06/p06_winning_model_per_class_f1_test.csv ml_app/artifacts/p06/

RUN mkdir -p entrega_2/dashboard/data

EXPOSE 8501

CMD ["streamlit", "run", "entrega_2/dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
