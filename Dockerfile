FROM python:3.8.16-slim

# set environment variables
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"
ENV PATH /usr/local/bin:$PATH
ENV LANG C.UTF-8
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# set work directory
WORKDIR /usr/src/app

COPY ./README.md /tmp/README.md
COPY ./setup.py /tmp/setup.py

# install git
RUN apt-get update && apt-get install git -y


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y\
    && pip install --upgrade pip setuptools wheel 

COPY ./src/requirements.txt /tmp/requirements.txt
RUN pip install --default-timeout=500 -r /tmp/requirements.txt \
    && pip cache purge \
    && rm -rf /root/.cache/pip

COPY ./pyroengine /tmp/pyroengine

RUN pip install -e /tmp/. \
    && pip cache purge \
    && rm -rf /root/.cache/pip

COPY ./src/run.py /usr/src/app/run.py
COPY ./src/control_reolink_cam.py /usr/src/app/control_reolink_cam.py