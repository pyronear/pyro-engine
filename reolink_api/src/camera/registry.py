# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import threading
from reolink import ReolinkCamera
from remote_camera import RemoteCamera
from camera.config import CAM_PWD, CAM_USER, RAW_CONFIG

CAMERA_REGISTRY = {}
PATROL_THREADS: dict[str, threading.Thread] = {}
PATROL_FLAGS: dict[str, threading.Event] = {}

for key, conf in RAW_CONFIG.items():
    brand = conf.get("brand", "").lower()
    if "reolink" in brand:
        cam = ReolinkCamera(
            ip_address=key,
            username=CAM_USER or "",
            password=CAM_PWD or "",
            cam_type=conf.get("type", "static"),
            cam_poses=conf.get("poses"),
            cam_azimuths=conf.get("azimuths"),
            focus_position=conf.get("focus_position", None),
        )
    else:
        cam = RemoteCamera(url=key, timeout=5)

    CAMERA_REGISTRY[key] = cam
