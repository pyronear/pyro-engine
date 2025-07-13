# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import threading

from reolink import ReolinkCamera

from camera.config import CAM_PWD, CAM_USER, RAW_CONFIG

CAMERA_REGISTRY = {}
PATROL_THREADS: dict[str, threading.Thread] = {}
PATROL_FLAGS: dict[str, threading.Event] = {}

for ip, conf in RAW_CONFIG.items():
    cam = ReolinkCamera(
        ip_address=ip,
        username=CAM_USER or "",
        password=CAM_PWD or "",
        cam_type=conf.get("type", "static"),
        cam_poses=conf.get("poses"),
        cam_azimuths=conf.get("azimuths"),
        focus_position=conf.get("focus_position", 720),
    )
    CAMERA_REGISTRY[ip] = cam
