# Copyright (C) 2022-2026, Pyronear.

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
CAM_USER: str = os.getenv("CAM_USER", "")
CAM_PWD: str = os.getenv("CAM_PWD", "")
MEDIAMTX_SERVER_IP = os.getenv("MEDIAMTX_SERVER_IP")

CREDENTIALS_PATH = Path("/usr/src/app/data/credentials.json")

RAW_CONFIG = {}
if CREDENTIALS_PATH.exists():
    try:
        with Path(CREDENTIALS_PATH).open("r", encoding="utf8") as f:
            RAW_CONFIG = json.load(f)
    except Exception:
        RAW_CONFIG = {}

USER_ENC = quote(CAM_USER, safe="")
PWD_ENC = quote(CAM_PWD, safe="")


def build_rtsp_input_url(ip: str, cfg: dict) -> str:
    username = quote(str(cfg.get("username", CAM_USER)), safe="")
    password = quote(str(cfg.get("password", CAM_PWD)), safe="")
    adapter = (cfg.get("adapter") or cfg.get("brand") or "").lower()

    if "linovision" in adapter:
        channel = str(cfg.get("rtsp_channel", cfg.get("channel", "102")))
        path = cfg.get("rtsp_path", f"/Streaming/Channels/{channel}")
    else:
        path = cfg.get("rtsp_path", "/h264Preview_01_sub")

    return f"rtsp://{username}:{password}@{ip}:554/{str(path).lstrip('/')}"


STREAMS: dict[str, dict] = {}

if RAW_CONFIG:
    for ip, cfg in RAW_CONFIG.items():
        id_or_name = cfg.get("streamid") or cfg.get("stream_name") or cfg.get("name", "stream")
        input_url = build_rtsp_input_url(ip, cfg)

        if id_or_name.startswith(("#!::", "publish:")) or ":" in id_or_name:
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
