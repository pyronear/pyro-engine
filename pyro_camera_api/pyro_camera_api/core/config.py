# pyro_camera_api/core/config.py

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import quote, urlencode

from dotenv import load_dotenv

# SRT parameters
SRT_PKT_SIZE = 1316
SRT_MODE = "caller"
SRT_LATENCY = 50
SRT_PORT_START = 8890
SRT_STREAMID_PREFIX = "publish"

# Default FFmpeg restream parameters
FFMPEG_PARAMS: dict[str, object] = {
    "discardcorrupt": True,
    "low_delay": True,
    "rtsp_transport": "tcp",
    "video_codec": "libx264",
    "b_frames": 0,
    "gop_size": 14,
    "bitrate": "700k",
    "framerate": 10,
    "preset": "veryfast",
    "tune": "zerolatency",
    "audio_disabled": True,
    "output_format": "mpegts",
}


def normalize_stream_name(name: str) -> str:
    """Normalize a camera stream name for SRT usage."""
    name = name.lower().replace("_", "-")
    return re.sub(r"-\d{1,2}$", "", name)


# Load environment variables from .env if present
load_dotenv()

# Project root and credentials path
ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = Path(os.getenv("CREDENTIALS_PATH") or (ROOT / "data" / "credentials.json"))

# Required environment variables
CAM_USER = os.getenv("CAM_USER")
CAM_PWD = os.getenv("CAM_PWD")
MEDIAMTX_SERVER_IP = os.getenv("MEDIAMTX_SERVER_IP")

if not CAM_USER or not CAM_PWD:
    raise RuntimeError("Missing environment variables CAM_USER or CAM_PWD")
if not MEDIAMTX_SERVER_IP:
    raise RuntimeError("Missing environment variable MEDIAMTX_SERVER_IP")

# Load camera configuration from credentials file
if not CREDENTIALS_PATH.exists():
    raise FileNotFoundError(f"Missing credentials file at {CREDENTIALS_PATH}")
with CREDENTIALS_PATH.open("r") as f:
    RAW_CONFIG: dict[str, dict] = json.load(f)

# Encode credentials for RTSP URL
USER_ENC = quote(CAM_USER, safe="")
PWD_ENC = quote(CAM_PWD, safe="")


def build_srt_output_url(name_or_id: str) -> str:
    """
    Build an SRT output URL for the given stream name or identifier.

    If value looks like a full SRT streamid already, pass as is.
    Otherwise prefix with publish and normalize.
    """
    if name_or_id.startswith("#!::") or name_or_id.startswith("publish:") or ":" in name_or_id:
        streamid = name_or_id
        safe_chars = ":,=/!"
    else:
        streamid = f"{SRT_STREAMID_PREFIX}:{normalize_stream_name(name_or_id)}"
        safe_chars = ":"

    query = urlencode(
        {
            "pkt_size": SRT_PKT_SIZE,
            "mode": SRT_MODE,
            "latency": SRT_LATENCY,
            "streamid": streamid,
        },
        safe=safe_chars,
    )
    return f"srt://{MEDIAMTX_SERVER_IP}:{SRT_PORT_START}?{query}"


# Final mapping used by the API layer for restreaming
STREAMS: dict[str, dict] = {}
for ip, cfg in RAW_CONFIG.items():
    id_or_name = cfg.get("streamid") or cfg.get("stream_name") or cfg.get("name", "stream")
    input_url = f"rtsp://{USER_ENC}:{PWD_ENC}@{ip}:554/h264Preview_01_sub"
    output_url = build_srt_output_url(id_or_name)

    STREAMS[ip] = {
        "input_url": input_url,
        "output_url": output_url,
    }
