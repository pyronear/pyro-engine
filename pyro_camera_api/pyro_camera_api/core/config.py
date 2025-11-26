# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import quote, urlencode

from dotenv import load_dotenv

SRT_PKT_SIZE = 1316
SRT_MODE = "caller"
SRT_LATENCY = 50
SRT_PORT_START = 8890
SRT_STREAMID_PREFIX = "publish"

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
    name = name.lower().replace("_", "-")
    return re.sub(r"-\d{1,2}$", "", name)


load_dotenv()

CAM_USER = os.getenv("CAM_USER")
CAM_PWD = os.getenv("CAM_PWD")
MEDIAMTX_SERVER_IP = os.getenv("MEDIAMTX_SERVER_IP")

CREDENTIALS_PATH = Path("/usr/src/app/data/credentials.json")

RAW_CONFIG = {}
if CREDENTIALS_PATH.exists():
    try:
        with open(CREDENTIALS_PATH, "r", encoding="utf8") as f:
            RAW_CONFIG = json.load(f)
    except Exception:
        RAW_CONFIG = {}

USER_ENC = quote(CAM_USER, safe="")
PWD_ENC = quote(CAM_PWD, safe="")

STREAMS: dict[str, dict] = {}

if RAW_CONFIG:
    for ip, cfg in RAW_CONFIG.items():
        id_or_name = cfg.get("streamid") or cfg.get("stream_name") or cfg.get("name", "stream")
        input_url = f"rtsp://{USER_ENC}:{PWD_ENC}@{ip}:554/h264Preview_01_sub"

        if id_or_name.startswith("#!::") or id_or_name.startswith("publish:") or ":" in id_or_name:
            streamid = id_or_name
            safe_chars = ":,=/!"
        else:
            streamid = f"{SRT_STREAMID_PREFIX}:{normalize_stream_name(id_or_name)}"
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
        output_url = f"srt://{MEDIAMTX_SERVER_IP}:{SRT_PORT_START}?{query}"

        STREAMS[ip] = {
            "input_url": input_url,
            "output_url": output_url,
        }
