FROM pyro-engine

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"
ENV PATH /usr/local/bin:$PATH
ENV LANG C.UTF-8

# copy requirements file
COPY runner/src/requirements.txt /usr/src/app/requirements.txt

RUN pip install -r /usr/src/app/requirements.txt \
    && rm -rf /root/.cache/pip

COPY runner/src/runner.py /usr/src/app/runner.py 
