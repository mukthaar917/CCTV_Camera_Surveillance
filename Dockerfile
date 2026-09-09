FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_NAME="Camera Surveillance AI API" \
    APP_VERSION="0.1.0" \
    DEBUG="false" \
    MODEL_WEIGHTS_PATH="models/weights/best.pt" \
    CONFIDENCE_THRESHOLD="0.35" \
    IOU_THRESHOLD="0.50" \
    IMAGE_SIZE="640" \
    DEVICE="cpu" \
    MAX_UPLOAD_SIZE_MB="10"

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY api ./api
COPY src ./src
COPY events ./events
COPY config ./config

RUN mkdir -p \
    /app/models/weights \
    /app/reports/metrics \
    /app/reports/figures \
    /app/reports/training_reports

RUN useradd \
        --create-home \
        --shell /usr/sbin/nologin \
        appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD [
    "uvicorn",
    "api.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "--workers",
    "1"
]