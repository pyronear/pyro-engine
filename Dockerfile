# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.11.17-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS builder
ARG TARGETARCH
COPY --from=uv /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /build

# Git and build tooling stay in this stage. Cache apt downloads outside image layers.
RUN --mount=type=cache,id=apt-cache-${TARGETARCH},target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-${TARGETARCH},target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends git

# Resolve no versions at build time. Keep dependencies cached across source-only edits.
# NCNN's GUI OpenCV dependency is intentionally replaced by the locked headless wheel.
RUN --mount=type=cache,id=uv-${TARGETARCH},target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-default-groups --no-install-project \
        --no-install-package pyro-predictor --no-install-package pyro-camera-api-client \
        --no-install-package opencv-python

# Build non-editable local packages. Bind mounts keep source/build artifacts out of layers.
RUN --mount=type=cache,id=uv-${TARGETARCH},target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=setup.py,target=setup.py \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=pyroengine,target=pyroengine,rw \
    --mount=type=bind,source=pyro-predictor,target=pyro-predictor,rw \
    --mount=type=bind,source=pyro_camera_api/client,target=pyro_camera_api/client,rw \
    uv sync --locked --no-default-groups --no-editable --no-install-package opencv-python \
    && chmod -R a+rX /opt/venv

FROM ${PYTHON_IMAGE} AS runtime
ARG TARGETARCH
ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/home/pyro \
    XDG_CACHE_HOME=/usr/src/app/data/.cache \
    HF_HOME=/usr/src/app/data/.cache/huggingface

# Headless wheels bundle their video libraries; the engine needs no FFmpeg/GUI stack.
RUN --mount=type=cache,id=apt-cache-${TARGETARCH},target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-${TARGETARCH},target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 libgomp1 \
    && groupadd --gid 1000 pyro \
    && useradd --uid 1000 --gid 1000 --no-log-init --home-dir /home/pyro --shell /usr/sbin/nologin pyro \
    && install -d -o 1000 -g 1000 /home/pyro /usr/src/app/data \
    && rm -rf /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/ensurepip \
    && rm -f /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.11

# Only installed runtime packages cross the stage boundary; they remain root-owned.
COPY --from=builder /opt/venv /opt/venv
WORKDIR /usr/src/app
COPY --chmod=0444 src/run.py src/control_reolink_cam.py ./
USER 1000:1000
CMD ["python", "run.py"]
