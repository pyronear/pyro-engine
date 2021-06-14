FROM pyronear/pyro-engine:latest

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app/server/src"

# copy requirements file
COPY ./server/src/requirements.txt /usr/src/app/requirements.txt

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# runtime dependencies
RUN set -eux \
	&& apt-get update \
	&& apt-get install -y --no-install-recommends \
		ca-certificates \
		netbase \
	&& rm -rf /var/lib/apt/lists/*

ENV GPG_KEY 3.8.1
ENV PYTHON_VERSION 3.8.1

EXPOSE 80

RUN set -ex \
	 \
	&& savedAptMark="$(apt-mark showmanual)" \
	&& apt-get update && apt-get install -y --no-install-recommends \
		gcc \
		python3-dev \
		libffi-dev \
		libssl-dev \
	&& pip install --upgrade pip setuptools wheel \
    && pip install -r /usr/src/app/requirements.txt \
    && rm -rf /root/.cache/pip

# copy project
COPY . /usr/src/app/

RUN git clone https://github.com/pyronear/pyro-api.git
RUN pip install -e pyro-api/client/.
