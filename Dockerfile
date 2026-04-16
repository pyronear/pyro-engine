# ---- Builder: only used for git-based deps (needs git) ----
FROM python:3.11.13-slim-bullseye AS git-deps

RUN apt-get update && \
    apt-get install -y --no-install-recommends git \
    || { apt-get update && apt-get install -y --no-install-recommends --fix-missing git; } \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.13 /uv /bin/uv
COPY ./requirements-git.txt /tmp/requirements-git.txt
RUN uv pip install --no-cache --target=/tmp/git-packages -r /tmp/requirements-git.txt

# ---- Runtime ----
FROM python:3.11.13-slim-bullseye

ENV LANG="C.UTF-8" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Layer 1: System libs (~100MB, almost never changes)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libglib2.0-0 \
        libgl1 \
    || { apt-get update && apt-get install -y --no-install-recommends --fix-missing \
        ffmpeg libsm6 libxext6 libglib2.0-0 libgl1; } \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Layer 2: Stable deps (~400MB, changes only on version bumps)
COPY --from=ghcr.io/astral-sh/uv:0.5.13 /uv /bin/uv
COPY ./requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache --system -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

# Layer 3: Git-based deps (~5MB, changes when API clients are updated)
COPY --from=git-deps /tmp/git-packages /usr/local/lib/python3.11/site-packages/

# Layer 4: Local packages (~1MB, changes on every deploy)
WORKDIR /opt/pyroengine_src
COPY ./pyro-predictor ./pyro-predictor
COPY ./pyro_camera_api/client ./pyro_camera_api/client
COPY ./pyroengine ./pyroengine
COPY ./pyproject.toml ./pyproject.toml
COPY ./setup.py ./setup.py
RUN uv pip install --no-cache --system --no-deps ./pyro-predictor \
    && uv pip install --no-cache --system --no-deps ./pyro_camera_api/client \
    && uv pip install --no-cache --system --no-deps . \
    && rm -f /bin/uv

# Layer 5: Entrypoint scripts (~few KB, rarely changes)
WORKDIR /usr/src/app
COPY ./src/run.py /usr/src/app/run.py
COPY ./src/control_reolink_cam.py /usr/src/app/control_reolink_cam.py
