# pyro_camera_api/core/config.py
# Copyright (C) 2022-2025, Pyronear.
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote, urlencode

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SRT and FFmpeg parameters
# ---------------------------------------------------------------------------

SRT_PKT_SIZE = 1316
SRT_MODE = "caller"
SRT_LATENCY = 50
SRT_PORT_START = 8890
SRT_STREAMID_PREFIX = "publish"

FFMPEG_PARAMS: Dict[str, object] = {
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
    name = name.lower().replace("_", "-")
    return re.sub(r"-\d{1,2}$", "", name)


# ---------------------------------------------------------------------------
# Paths and env
# ---------------------------------------------------------------------------

load_dotenv()

# project root, two levels above core
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR") or (PROJECT_ROOT / "data"))

CREDENTIALS_PATH = Path(os.getenv("CREDENTIALS_PATH") or (DATA_DIR / "credentials.json"))

CAM_USER = os.getenv("CAM_USER", "")
CAM_PWD = os.getenv("CAM_PWD", "")
MEDIAMTX_SERVER_IP = os.getenv("MEDIAMTX_SERVER_IP", "")

if not CAM_USER or not CAM_PWD:
    logger.warning("Environment variables CAM_USER or CAM_PWD are not set")
if not MEDIAMTX_SERVER_IP:
    logger.warning("Environment variable MEDIAMTX_SERVER_IP is not set, defaulting to 127.0.0.1")
    MEDIAMTX_SERVER_IP = MEDIAMTX_SERVER_IP or "127.0.0.1"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logger.warning("Configuration file not found at %s", path)
        return {}
    try:
        with path.open("r", encoding="utf8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load JSON from %s, %s", path, exc)
        return {}


# ---------------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------------

RAW_CONFIG: Dict[str, Dict[str, Any]] = _load_json(CREDENTIALS_PATH)

if not RAW_CONFIG:
    logger.warning("RAW_CONFIG is empty, no cameras configured")

# ---------------------------------------------------------------------------
# RTSP stream mapping
# ---------------------------------------------------------------------------

STREAMS: Dict[str, Dict[str, Any]] = {}

if CAM_USER and CAM_PWD and MEDIAMTX_SERVER_IP and RAW_CONFIG:
    user_enc = quote(CAM_USER, safe="")
    pwd_enc = quote(CAM_PWD, safe="")

    def build_srt_output_url(name_or_id: str) -> str:
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

    for ip, cfg in RAW_CONFIG.items():
        id_or_name = cfg.get("streamid") or cfg.get("stream_name") or cfg.get("name", "stream")
        input_url = f"rtsp://{user_enc}:{pwd_enc}@{ip}:554/h264Preview_01_sub"
        output_url = build_srt_output_url(id_or_name)

        STREAMS[ip] = {
            "input_url": input_url,
            "output_url": output_url,
        }
else:
    logger.warning("STREAMS not built because of missing credentials, server IP, or RAW_CONFIG")
