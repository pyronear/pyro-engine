import json
import os
import re
from pathlib import Path
from urllib.parse import quote, urlencode

import yaml
from dotenv import load_dotenv


def normalize_stream_name(name: str) -> str:
    name = name.lower().replace("_", "-")
    return re.sub(r"-\d{1,2}$", "", name)


load_dotenv()

FFMPEG_CONFIG_PATH = Path(__file__).resolve().parent.parent / "ffmpeg_config.yaml"
CREDENTIALS_PATH = Path(__file__).resolve().parent.parent / "credentials.json"

CAM_USER = os.getenv("CAM_USER")
CAM_PWD = os.getenv("CAM_PWD")
MEDIAMTX_SERVER_IP = os.getenv("MEDIAMTX_SERVER_IP")

if not CAM_USER or not CAM_PWD:
    raise RuntimeError("Missing env variables: CAM_USER or CAM_PWD")
if not MEDIAMTX_SERVER_IP:
    raise RuntimeError("Missing env variable: MEDIAMTX_SERVER_IP")

if not CREDENTIALS_PATH.exists():
    raise FileNotFoundError(f"Missing credentials file at {CREDENTIALS_PATH}")
with open(CREDENTIALS_PATH) as f:
    RAW_CONFIG = json.load(f)

if not FFMPEG_CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing ffmpeg_config.yaml at {FFMPEG_CONFIG_PATH}")
with open(FFMPEG_CONFIG_PATH, "r") as f:
    FFMPEG_CONFIG = yaml.safe_load(f)

SRT_SETTINGS = FFMPEG_CONFIG["srt_settings"]
FFMPEG_PARAMS = FFMPEG_CONFIG["ffmpeg_params"]

# encode credentials for RTSP URL safety
user_enc = quote(CAM_USER, safe="")
pwd_enc = quote(CAM_PWD, safe="")

def build_srt_output_url(name_or_id: str) -> str:
    """
    Accept either a full explicit streamid (for example OME style '#!::r=app/stream,m=publish')
    or a bare name that we will prefix with publish:
    """
    if name_or_id.startswith("#!::"):
        streamid = name_or_id
        safe = ":,=/!"  # keep these characters unescaped for advanced formats
    else:
        streamid = f"{SRT_SETTINGS['streamid_prefix']}:{normalize_stream_name(name_or_id)}"
        safe = ""       # simple value, no special characters
    query = urlencode(
        {
            "pkt_size": SRT_SETTINGS["pkt_size"],
            "mode": SRT_SETTINGS["mode"],
            "latency": SRT_SETTINGS["latency"],
            "streamid": streamid,
        },
        safe=safe,
    )
    return f"srt://{MEDIAMTX_SERVER_IP}:{SRT_SETTINGS['port_start']}?{query}"

# Build STREAMS, with per camera overrides for stream naming
STREAMS = {}
for ip, cfg in RAW_CONFIG.items():
    # choose stream id or name in this order: explicit streamid, explicit stream_name, camera name, fallback
    explicit_streamid = cfg.get("streamid")
    stream_name = cfg.get("stream_name") or cfg.get("name", "stream")
    id_or_name = explicit_streamid or stream_name

    STREAMS[ip] = {
        "input_url": f"rtsp://{user_enc}:{pwd_enc}@{ip}:554/h264Preview_01_sub",
        "output_url": build_srt_output_url(id_or_name),
    }
