FROM python:3.9.16-slim

# set environment variables
ENV PATH="/usr/local/bin:$PATH" \
    LANG="C.UTF-8" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
# set work directory
WORKDIR /usr/src/app

# install git
RUN apt-get update && apt-get install git -y

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
    || apt-get install -y --fix-missing && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --default-timeout=500 -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

WORKDIR /opt/pyroengine_src
COPY ./pyroengine ./pyroengine
COPY ./setup.py ./setup.py

RUN pip install --no-cache-dir -e . \
    && rm -rf /root/.cache/pip

WORKDIR /usr/src/app

COPY ./src/run.py /usr/src/app/run.py
COPY ./src/control_reolink_cam.py /usr/src/app/control_reolink_cam.py