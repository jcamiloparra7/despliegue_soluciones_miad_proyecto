#!/usr/bin/env bash
set -euo pipefail

if ! command -v mlflow >/dev/null 2>&1; then
  echo "Error: mlflow is not installed or not available in PATH." >&2
  exit 1
fi

HOST="${MLFLOW_HOST:-0.0.0.0}"
PORT="${MLFLOW_PORT:-8050}"
BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///$(pwd)/mlflow.db}"
ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-$(pwd)/mlflow-artifacts}"
ALLOWED_HOSTS="${MLFLOW_ALLOWED_HOSTS:-*}"
CORS_ALLOWED_ORIGINS="${MLFLOW_CORS_ALLOWED_ORIGINS:-*}"

mkdir -p "${ARTIFACT_ROOT}"

exec mlflow server \
  --host "${HOST}" \
  --port "${PORT}" \
  --allowed-hosts "${ALLOWED_HOSTS}" \
  --cors-allowed-origins "${CORS_ALLOWED_ORIGINS}" \
  --backend-store-uri "${BACKEND_STORE_URI}" \
  --default-artifact-root "${ARTIFACT_ROOT}" \
  --serve-artifacts
