FROM python:3.9.16-slim

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/usr/src/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        build-essential \
        libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.13 /uv /bin/uv

COPY requirements.txt /tmp/requirements.txt
# if you have a lock file uncomment this
# COPY uv.lock /tmp/uv.lock

RUN uv pip install --no-cache --system -r /tmp/requirements.txt

COPY pyproject.toml ./pyproject.toml
COPY pyro_camera_api ./pyro_camera_api

CMD ["uvicorn", "pyro_camera_api.main:app", "--host", "0.0.0.0", "--port", "8081"]
