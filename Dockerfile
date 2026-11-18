# Demo web server: one uvicorn process serves the API and the static UI.
# Built in the cloud, not locally — see docs/DEPLOY_AZURE.md (GitHub Actions -> GHCR -> Azure).
# Secrets (APP_PASSWORD, APP_USER) come from the environment at run time; nothing secret is
# baked into the image. Build context is the repo root.
#   docker build -t retail-forecast .   # (optional local build)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Fail closed: the container refuses to start without APP_PASSWORD.
    AUTH_REQUIRED=1

WORKDIR /app

# Runtime deps only (no torch/lightgbm/chronos — those are for the pipeline, not serving).
COPY server/requirements.txt ./server/requirements.txt
RUN pip install -r server/requirements.txt

# Application code. config.py resolves paths relative to src/m5demo, so the layout matters.
COPY server ./server
COPY src/m5demo ./src/m5demo
COPY web ./web

# Only the three files the server actually reads (~92 MB total), not the whole data/ dir.
COPY data/results/full/test_forecasts.parquet ./data/results/full/test_forecasts.parquet
COPY data/processed/panel.parquet ./data/processed/panel.parquet
COPY data/processed/calendar.parquet ./data/processed/calendar.parquet

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
# --proxy-headers: trust X-Forwarded-Proto/For from the platform's ingress (TLS ends there).
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
