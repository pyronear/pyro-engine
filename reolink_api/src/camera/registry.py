# Copyright (C) 2025-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from reolink import ReolinkCamera

from camera.config import CAM_PWD, CAM_USER, RAW_CONFIG

CAMERA_REGISTRY = {}
PATROL_THREADS = {}  # {camera_ip: threading.Thread}
PATROL_FLAGS = {}  # {camera_ip: threading.Event}

for ip, conf in RAW_CONFIG.items():
    cam = ReolinkCamera(
        ip_address=ip,
        username=CAM_USER,
        password=CAM_PWD,
        cam_type=conf.get("type", "ptz"),
        cam_poses=conf.get("poses"),
        cam_azimuths=conf.get("azimuths"),
        focus_position=conf.get("focus_position", 720),
    )
    CAMERA_REGISTRY[ip] = cam
