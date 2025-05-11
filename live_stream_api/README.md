# 🎥 Reolink Camera Stream Controller

This FastAPI app lets you control Reolink cameras (PTZ, zoom) and **stream live video feeds directly using FFmpeg over SRT**.  
There is **no MediaMTX server** involved anymore.

It supports multiple cameras (`cam1`, `cam2`, etc.) and automatically stops inactive streams after 60 seconds.

---

## 🛠 Installation

### 1. Install FFmpeg

Make sure FFmpeg is installed:

```bash
sudo apt update
sudo apt install ffmpeg
```

---

### 2. Prepare the Configuration Files

You need:

- A `.env` file (you must create it)
- A `credentials.json` file (use your own, or the one from **pyro-engine**)

The `ffmpeg_config.yaml` is already provided in this repository.

---

#### 📄 `.env`

Create a `.env` file with:

```bash
CAM_USER=admin
CAM_PWD=@Pyronear
MEDIAMTX_SERVER_IP=YOUR_SERVER_PUBLIC_IP
```

Replace `YOUR_SERVER_PUBLIC_IP` with your real public or private server IP address.

---

#### 📄 `credentials.json`

Example structure:

```json
{
  "cameras": {
    "cam1": "169.254.40.1",
    "cam2": "169.254.40.2"
  }
}
```

You can create your own `credentials.json`,  
**or reuse the one from [pyro-engine](https://github.com/pyronear/pyro-engine)** if available.

---

#### 📄 `ffmpeg_config.yaml`

The `ffmpeg_config.yaml` file is already included in the repository.  
It contains all FFmpeg and SRT streaming parameters.  
You can easily edit it to change bitrate, framerate, ports, etc.

---

## 🚀 Run the App

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

(Optional: run it inside a `screen` session)

```bash
screen -S reolink-app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

(Detach from screen with `Ctrl+A`, then `D`)

---

## 📡 API Endpoints

### Stream Control

- `POST /start_stream/{camera_id}` – Start streaming from a camera
- `POST /stop_stream` – Stop any active stream
- `GET /status` – Check which stream (if any) is running

### Camera Control

- `POST /move/{camera_id}/{direction}/{speed}` – Move PTZ camera (`Up`, `Down`, `Left`, `Right`) at specified speed
- `POST /stop/{camera_id}` – Stop camera movement
- `POST /zoom/{camera_id}/{level}` – Zoom camera (0–64)

---

## 🔒 Notes

- The cameras use HTTPS. SSL certificate verification is disabled for Reolink API requests.
- Only **one stream runs at a time** — starting a new stream will stop the previous one.
- Streams are **automatically stopped after 60 seconds** of inactivity.


---

# 🛠️ Maintained By
[Pyronear](https://pyronear.org/)

---

