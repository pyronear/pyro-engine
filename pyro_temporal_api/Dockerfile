FROM python:3.11.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/usr/src/app

# Layer 1: git for the temporal-model dependency, curl for the healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git \
    || { apt-get update && apt-get install -y --no-install-recommends --fix-missing curl git; } \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Layer 2: Pip deps (onnxruntime + temporal-model-core, no torch)
COPY --from=ghcr.io/astral-sh/uv:0.6.16 /uv /bin/uv
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache --system -r /tmp/requirements.txt && \
    rm /bin/uv /tmp/requirements.txt

# Layer 3: Source code
WORKDIR /usr/src/app
COPY pyproject.toml ./pyproject.toml
COPY pyro_temporal_api ./pyro_temporal_api

# host network mode: bind to loopback, only the engine on the same box talks to this service
CMD ["uvicorn", "pyro_temporal_api.main:app", "--host", "127.0.0.1", "--port", "8082"]
