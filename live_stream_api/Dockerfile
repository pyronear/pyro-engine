# live_stream_api/Dockerfile

FROM python:3.9.16-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]
