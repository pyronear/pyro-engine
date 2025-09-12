FROM python:3.9.16-slim

WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/usr/src/app"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
    || apt-get install -y --fix-missing && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install uv
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.5.13 /uv /bin/uv

# copy requirements file
COPY requirements.txt /tmp/requirements.txt
# install dependencies
RUN uv pip install --no-cache --system -r /tmp/requirements.txt

# copy the source code
COPY src /usr/src/app