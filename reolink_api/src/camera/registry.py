# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import threading
from typing import Dict

from camera.camera_reolink import ReolinkCamera
from camera.camera_rtsp import RTSPCamera
from camera.camera_url import URLCamera
from camera.config import CAM_PWD, CAM_USER, RAW_CONFIG

logger = logging.getLogger("CameraRegistry")
logger.setLevel(logging.INFO)

CAMERA_REGISTRY: Dict[str, object] = {}
PATROL_THREADS: Dict[str, threading.Thread] = {}
PATROL_FLAGS: Dict[str, threading.Event] = {}

for key, conf in RAW_CONFIG.items():
    """
    key is the camera identifier, usually unique label in RAW_CONFIG.
    conf is expected to include:
      - brand (ex. 'reolink-823S2', 'mobotix', etc.)
      - type ('ptz', 'rtsp', 'static', ...)
      - ip_address (optional; fallback to key)
      - rtsp_url (for rtsp cams)
      - url / snapshot_url (for static snapshot cams)
      - poses / azimuths / focus_position for PTZ patrol
    """

    brand = conf.get("brand", "").lower()
    cam_type = conf.get("type", "static").lower()
    ip_addr = conf.get("ip_address", key)

    # 1. Reolink PTZ control
    if "reolink" in brand and cam_type == "ptz":
        cam_obj = ReolinkCamera(
            ip_address=ip_addr,
            username=CAM_USER or "",
            password=CAM_PWD or "",
            cam_type=cam_type,
            cam_poses=conf.get("poses", []),
            cam_azimuths=conf.get("azimuths", []),
            focus_position=conf.get("focus_position", None),
        )
        logger.info("Registered Reolink camera %s (%s)", key, cam_type)

    # 2. RTSP video stream grabber
    elif cam_type in ("rtsp", "rtsp_tcp", "rtsp_udp"):
        rtsp_url = conf.get("rtsp_url", "")
        if not rtsp_url:
            logger.warning("Camera %s declared as RTSP but no rtsp_url found", key)
        cam_obj = RTSPCamera(
            rtsp_url=rtsp_url,
            ip_address=ip_addr,
            cam_type="static",
        )
        logger.info("Registered RTSP camera %s (%s)", key, cam_type)

    # 3. HTTP snapshot grabber
    else:
        snapshot_url = conf.get("url") or conf.get("snapshot_url") or conf.get("rtsp_url") or key
        cam_obj = URLCamera(
            url=snapshot_url,
            timeout=5,
            cam_type="static",
        )
        logger.info("Registered URL camera %s (%s)", key, cam_type)

    CAMERA_REGISTRY[key] = cam_obj
