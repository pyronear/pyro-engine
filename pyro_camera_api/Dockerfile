FROM python:3.9.16-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/usr/src/app

# Layer 1: System libs (~100MB, almost never changes)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Layer 2: Pip deps (~400MB, changes only on version bumps)
COPY --from=ghcr.io/astral-sh/uv:0.5.13 /uv /bin/uv
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache --system -r /tmp/requirements.txt && \
    rm /bin/uv /tmp/requirements.txt

# Layer 3: Source code (~few KB, changes on every deploy)
WORKDIR /usr/src/app
COPY pyproject.toml ./pyproject.toml
COPY pyro_camera_api ./pyro_camera_api

CMD ["uvicorn", "pyro_camera_api.main:app", "--host", "0.0.0.0", "--port", "8081"]
