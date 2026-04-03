# ---- Builder: only used for git-based deps (needs git) ----
FROM python:3.11.13-slim-bullseye AS git-deps

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements-git.txt /tmp/requirements-git.txt
RUN pip install --no-cache-dir --default-timeout=500 --target=/tmp/git-packages -r /tmp/requirements-git.txt

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
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Layer 2: Stable pip deps (~400MB, changes only on version bumps)
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --default-timeout=500 -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

# Layer 3: Git-based deps (~5MB, changes when API clients are updated)
COPY --from=git-deps /tmp/git-packages /usr/local/lib/python3.11/site-packages/

# Layer 4: Local packages (~1MB, changes on every deploy)
WORKDIR /opt/pyroengine_src
COPY ./pyro-predictor ./pyro-predictor
COPY ./pyroengine ./pyroengine
COPY ./setup.py ./setup.py
RUN pip install --no-cache-dir ./pyro-predictor \
    && pip install --no-cache-dir .

# Layer 5: Entrypoint scripts (~few KB, rarely changes)
WORKDIR /usr/src/app
COPY ./src/run.py /usr/src/app/run.py
COPY ./src/control_reolink_cam.py /usr/src/app/control_reolink_cam.py
