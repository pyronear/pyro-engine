# Copyright (C) 2022-2025, Pyronear.
# Licensed under the Apache License 2.0
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import threading
from typing import Dict, Optional

from camera.camera_reolink import ReolinkCamera
from camera.camera_rtsp import RTSPCamera
from camera.camera_url import URLCamera
from camera.config import CAM_PWD, CAM_USER, RAW_CONFIG

logger = logging.getLogger("CameraRegistry")
logger.setLevel(logging.INFO)

CAMERA_REGISTRY: Dict[str, object] = {}
PATROL_THREADS: Dict[str, threading.Thread] = {}
PATROL_FLAGS: Dict[str, threading.Event] = {}


def build_camera_object(key: str, conf: dict) -> Optional[object]:
    """
    Build the appropriate camera object based on configuration.

    Expected keys:
        - brand: "reolink", "rtsp", or "http"
        - type: "ptz" or "static"
        - ip_address (optional)
        - rtsp_url (for RTSP cameras)
        - url (for HTTP cameras)
        - poses / azimuths / focus_position (for PTZ)
    """

    brand = conf.get("brand", "").lower()
    cam_type = conf.get("type", "static").lower()
    ip_addr = conf.get("ip_address", key)

    # 1. Reolink cameras (PTZ or static)
    if "reolink" in brand:
        cam_obj = ReolinkCamera(
            ip_address=ip_addr,
            username=CAM_USER or "",
            password=CAM_PWD or "",
            cam_type=cam_type,
            cam_poses=conf.get("poses", []),
            cam_azimuths=conf.get("azimuths", []),
            focus_position=conf.get("focus_position"),
        )
        logger.info("Registered Reolink camera %s (ip=%s, type=%s)", key, ip_addr, cam_type)
        return cam_obj

    # 2. RTSP cameras
    if brand == "rtsp":
        rtsp_url = conf.get("rtsp_url")
        if not rtsp_url:
            logger.error("Camera %s declared as RTSP but missing 'rtsp_url'. Skipping registration.", key)
            return None

        cam_obj = RTSPCamera(
            rtsp_url=rtsp_url,
            ip_address=ip_addr,
            cam_type="static",
        )
        logger.info("Registered RTSP camera %s (ip=%s)", key, ip_addr)
        return cam_obj

    # 3. HTTP snapshot cameras
    snapshot_url = conf.get("url")
    if not snapshot_url:
        logger.error("Camera %s declared as static but missing 'url'. Skipping registration.", key)
        return None

    cam_obj = URLCamera(
        url=snapshot_url,
        timeout=5,
        cam_type="static",
    )
    logger.info("Registered HTTP snapshot camera %s (ip=%s)", key, ip_addr)
    return cam_obj


# Build the global registry safely
for key, conf in RAW_CONFIG.items():
    try:
        cam_obj = build_camera_object(key, conf)
        if cam_obj is not None:
            CAMERA_REGISTRY[key] = cam_obj
            im = cam_obj.capture()
            print(f"Image captured from {key}, image size {im.size}")
        else:
            logger.warning("Camera %s was not registered due to configuration error.", key)
    except Exception as e:
        logger.error("Failed to initialize camera %s: %s", key, e)
