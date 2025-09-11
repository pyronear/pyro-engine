# Copyright (C) 2020-2025, Pyronear.

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import quote, urlencode

import yaml
from dotenv import load_dotenv


def normalize_stream_name(name: str) -> str:
    # lowercase, replace underscores by hyphens, drop trailing -nn
    name = name.lower().replace("_", "-")
    return re.sub(r"-\d{1,2}$", "", name)


# load env
load_dotenv()

# paths, allow override via env
ROOT = Path(__file__).resolve().parent.parent
FFMPEG_CONFIG_PATH = Path(os.getenv("FFMPEG_CONFIG_PATH") or (ROOT / "ffmpeg_config.yaml"))
CREDENTIALS_PATH = Path(os.getenv("CREDENTIALS_PATH") or (ROOT / "credentials.json"))

# required env vars
CAM_USER = os.getenv("CAM_USER")
CAM_PWD = os.getenv("CAM_PWD")
MEDIAMTX_SERVER_IP = os.getenv("MEDIAMTX_SERVER_IP")

if not CAM_USER or not CAM_PWD:
    raise RuntimeError("Missing env variables CAM_USER or CAM_PWD")
if not MEDIAMTX_SERVER_IP:
    raise RuntimeError("Missing env variable MEDIAMTX_SERVER_IP")

# load camera credentials
if not CREDENTIALS_PATH.exists():
    raise FileNotFoundError(f"Missing credentials file at {CREDENTIALS_PATH}")
with open(CREDENTIALS_PATH, "r") as f:
    RAW_CONFIG: dict[str, dict] = json.load(f)

# load ffmpeg settings
if not FFMPEG_CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing ffmpeg_config.yaml at {FFMPEG_CONFIG_PATH}")
with open(FFMPEG_CONFIG_PATH, "r") as f:
    FFMPEG_CONFIG = yaml.safe_load(f)

# sections that other modules import
SRT_SETTINGS = FFMPEG_CONFIG["srt_settings"]
FFMPEG_PARAMS = FFMPEG_CONFIG["ffmpeg_params"]

# encode credentials for RTSP URL
USER_ENC = quote(CAM_USER, safe="")
PWD_ENC = quote(CAM_PWD, safe="")

def build_srt_output_url(name_or_id: str) -> str:
    """
    If value looks like a full SRT streamid already, pass as is.
    Otherwise prefix with publish and normalize.
    """
    # full streamid override
    if name_or_id.startswith("#!::") or name_or_id.startswith("publish:") or ":" in name_or_id:
        streamid = name_or_id
        safe_chars = ":,=/!"  # keep separators as is in query
    else:
        streamid = f"{SRT_SETTINGS['streamid_prefix']}:{normalize_stream_name(name_or_id)}"
        safe_chars = ":"  # keep colon

    query = urlencode(
        {
            "pkt_size": SRT_SETTINGS["pkt_size"],
            "mode": SRT_SETTINGS["mode"],
            "latency": SRT_SETTINGS["latency"],
            "streamid": streamid,
        },
        safe=safe_chars,
    )
    return f"srt://{MEDIAMTX_SERVER_IP}:{SRT_SETTINGS['port_start']}?{query}"

# final mapping used by the API layer
STREAMS: dict[str, dict] = {}
for ip, cfg in RAW_CONFIG.items():
    # pick id or name for the streamid
    id_or_name = cfg.get("streamid") or cfg.get("stream_name") or cfg.get("name", "stream")
    input_url = f"rtsp://{USER_ENC}:{PWD_ENC}@{ip}:554/h264Preview_01_sub"
    output_url = build_srt_output_url(id_or_name)

    STREAMS[ip] = {
        "input_url": input_url,
        "output_url": output_url,
        # expose a normalized base name for logs or UI
        "stream_name": normalize_stream_name(cfg.get("stream_name") or cfg.get("name", "stream")),
    }
