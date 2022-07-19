FROM python:3.8.1-slim

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1


COPY ./README.md /tmp/README.md
COPY ./setup.py /tmp/setup.py
COPY ./pyroengine /tmp/pyroengine
COPY ./requirements.txt /tmp/requirements.txt

RUN apt update \
    && apt install -y git \
    && apt install ffmpeg libsm6 libxext6  -y \
    && apt install -y gcc python3-dev \
    && pip install --upgrade pip setuptools wheel \
    && pip install -e /tmp/. \
    && pip cache purge \
    && rm -rf /root/.cache/pip
    && rm -rf /var/lib/apt/lists/*
