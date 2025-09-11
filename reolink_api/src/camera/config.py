# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import json
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv


def normalize_stream_name(name: str) -> str:
    name = name.lower().replace("_", "-")
    return re.sub(r"-\d{1,2}$", "", name)


# Load environment variables from .env file
load_dotenv()

# Paths
FFMPEG_CONFIG_PATH = Path(__file__).resolve().parent.parent / "ffmpeg_config.yaml"
CREDENTIALS_PATH = Path(__file__).resolve().parent.parent / "data/credentials.json"

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
# Extract sections
SRT_SETTINGS = FFMPEG_CONFIG.get("srt_settings", {}) or {}
FFMPEG_PARAMS = FFMPEG_CONFIG.get("ffmpeg_params", {}) or {}

# Choose a stream name; remove numeric suffix if present; normalize
first_cam = next(iter(RAW_CONFIG.values()))
STREAM_NAME = normalize_stream_name(first_cam.get("name", "stream"))

# Helper to build the SRT output URL for one camera name
from urllib.parse import urlencode


def build_srt_output_url(mediatx_ip: str, cam_name: str) -> str:
    srt_port = int(SRT_SETTINGS.get("port_start", SRT_SETTINGS.get("port", 8890)))
    streamid_prefix = SRT_SETTINGS.get("streamid_prefix", "publish")
    clean = normalize_stream_name(cam_name)

    params = {
        "pkt_size": str(SRT_SETTINGS.get("pkt_size", 1316)),
        "mode": SRT_SETTINGS.get("mode", "caller"),
        "latency": str(SRT_SETTINGS.get("latency", 30)),
    }
    if SRT_SETTINGS.get("rcvlatency") is not None:
        params["rcvlatency"] = str(SRT_SETTINGS["rcvlatency"])
    if SRT_SETTINGS.get("peerlatency") is not None:
        params["peerlatency"] = str(SRT_SETTINGS["peerlatency"])
    if SRT_SETTINGS.get("tlpktdrop") is not None:
        params["tlpktdrop"] = str(SRT_SETTINGS["tlpktdrop"])

    q = urlencode(params)
    streamid = f"{streamid_prefix}:{clean}"  # keep the colon
    return f"srt://{mediatx_ip}:{srt_port}?{q}&streamid={streamid}"


# Build STREAMS from credentials.json
STREAMS = {
    ip: {
        "input_url": f"rtsp://{CAM_USER}:{CAM_PWD}@{ip}:554/h264Preview_01_sub",
        "output_url": build_srt_output_url(MEDIAMTX_SERVER_IP, config.get("name", "stream")),
        "width": 640,
        "height": 360,
    }
    for ip, config in RAW_CONFIG.items()
}
