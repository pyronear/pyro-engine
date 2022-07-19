FROM pyronear/pyro-engine:latest

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"
ENV PATH /usr/local/bin:$PATH
ENV LANG C.UTF-8

COPY runner/src/runner.py /usr/src/app/runner.py 
