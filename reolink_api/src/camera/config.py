import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Paths
FFMPEG_CONFIG_PATH = Path(__file__).resolve().parent.parent / "ffmpeg_config.yaml"
CREDENTIALS_PATH = Path(__file__).resolve().parent.parent / "credentials.json"

# Env vars
CAM_USER = os.getenv("CAM_USER")
CAM_PWD = os.getenv("CAM_PWD")
MEDIAMTX_SERVER_IP = os.getenv("MEDIAMTX_SERVER_IP")

if not CAM_USER or not CAM_PWD:
    raise RuntimeError("Missing env variables: CAM_USER or CAM_PWD")

if not MEDIAMTX_SERVER_IP:
    raise RuntimeError("Missing env variable: MEDIAMTX_SERVER_IP")

# Load credentials.json
if not CREDENTIALS_PATH.exists():
    raise FileNotFoundError(f"Missing credentials file at {CREDENTIALS_PATH}")
with open(CREDENTIALS_PATH) as f:
    RAW_CONFIG = json.load(f)

# Load ffmpeg_config.yaml
if not FFMPEG_CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing ffmpeg_config.yaml at {FFMPEG_CONFIG_PATH}")
with open(FFMPEG_CONFIG_PATH, "r") as f:
    FFMPEG_CONFIG = yaml.safe_load(f)

# Extract sections
SRT_SETTINGS = FFMPEG_CONFIG["srt_settings"]
FFMPEG_PARAMS = FFMPEG_CONFIG["ffmpeg_params"]

# Choose a stream name (remove suffix if needed)
first_cam = next(iter(RAW_CONFIG.values()))
STREAM_NAME = first_cam.get("name", "stream").rsplit("-", 1)[0].lower()

# Build STREAMS
STREAMS = {
    ip: {
        "input_url": f"rtsp://{CAM_USER}:{CAM_PWD}@{ip}:554/h264Preview_01_sub",
        "output_url": (
            f"srt://{MEDIAMTX_SERVER_IP}:{SRT_SETTINGS['port_start']}?"
            f"pkt_size={SRT_SETTINGS['pkt_size']}&"
            f"mode={SRT_SETTINGS['mode']}&"
            f"latency={SRT_SETTINGS['latency']}&"
            f"streamid={SRT_SETTINGS['streamid_prefix']}:{STREAM_NAME}"
        ),
    }
    for ip in RAW_CONFIG
}
