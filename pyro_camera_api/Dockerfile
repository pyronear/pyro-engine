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

# This project is not packaged; only its locked runtime dependencies enter the venv.
# Skip NCNN's redundant GUI OpenCV wheel so cv2 always comes from headless OpenCV.
RUN --mount=type=cache,id=uv-${TARGETARCH},target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups --no-install-package opencv-python \
    && chmod -R a+rX /opt/venv

FROM ${PYTHON_IMAGE} AS runtime
ARG TARGETARCH
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH=/usr/src/app \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    XDG_CACHE_HOME=/usr/src/app/data/.cache

# FFmpeg is required by RTSP/SRT streaming. No curl, Git, uv or build environment ships.
RUN --mount=type=cache,id=apt-cache-${TARGETARCH},target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-${TARGETARCH},target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && install -d /usr/src/app/data /usr/src/app/focus_debug \
    && rm -rf /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/ensurepip \
    && rm -f /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.11

COPY --from=builder /opt/venv /opt/venv
WORKDIR /usr/src/app
COPY --chmod=u=rwX,go=rX pyro_camera_api ./pyro_camera_api
EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "from http.client import HTTPConnection; c = HTTPConnection('127.0.0.1', 8081, timeout=4); c.request('GET', '/health'); raise SystemExit(c.getresponse().status != 200)"]
CMD ["python", "-m", "uvicorn", "pyro_camera_api.main:app", "--host", "0.0.0.0", "--port", "8081"]
