FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=10001:10001 src ./src
COPY --chown=10001:10001 models ./models

ENV PYTHONPATH=/app/src \
    MODEL_DIR=/app/models \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PORT=8000

USER 10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["sh", "-c", "uvicorn churn.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
