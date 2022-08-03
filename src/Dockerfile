FROM pyronear/pyro-engine:latest

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"
ENV PATH /usr/local/bin:$PATH
ENV LANG C.UTF-8

COPY ./src/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements.txt \
    && pip cache purge \
    && rm -rf /root/.cache/pip

COPY .src/run.py /usr/src/app/run.py 
