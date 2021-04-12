FROM nelnatsol/pyrovision:latest

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1


COPY ./requirements.txt /tmp/requirements.txt
COPY ./README.md /tmp/README.md
COPY ./setup.py /tmp/setup.py
COPY ./pyroengine /tmp/pyroengine

RUN pip install --upgrade pip setuptools wheel \
    && pip install -e /tmp/. \
    && pip cache purge \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip
