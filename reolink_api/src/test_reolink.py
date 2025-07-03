import json
import os

from reolink import ReolinkCamera
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CREDENTIALS_PATH = "../../data/credentials.json"
CAM_USER = os.getenv("CAM_USER")
CAM_PWD = os.getenv("CAM_PWD")

with open(CREDENTIALS_PATH) as f:
    RAW_CONFIG = json.load(f)

CAMERA_REGISTRY = {}

for ip, conf in RAW_CONFIG.items():
    cam = ReolinkCamera(
        ip_address=ip,
        username=CAM_USER or "",
        password=CAM_PWD or "",
        cam_type=conf.get("type", "ptz"),
        cam_poses=conf.get("poses"),
        cam_azimuths=conf.get("azimuths"),
        focus_position=conf.get("focus_position", 720),
    )
    CAMERA_REGISTRY[ip] = cam


for cam in CAMERA_REGISTRY.values():
    im = cam.capture()
    print(im.size)