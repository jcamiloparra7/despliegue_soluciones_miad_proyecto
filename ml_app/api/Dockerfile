FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY ml_app/api/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY ml_app/api ml_app/api
COPY ml_app/scripts/p05_baseline_go_emotions.py ml_app/scripts/p05_baseline_go_emotions.py
COPY ml_app/artifacts/p06/models/winning_tfidf_calibrated_linearsvc_c_0_75.joblib ml_app/artifacts/p06/models/winning_tfidf_calibrated_linearsvc_c_0_75.joblib

EXPOSE 8000

CMD ["uvicorn", "ml_app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
