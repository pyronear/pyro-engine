# pyro_camera_api/camera/registry.py

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

from pyro_camera_api.camera.backends.reolink import ReolinkCamera
from pyro_camera_api.camera.backends.rtsp import RTSPCamera
from pyro_camera_api.camera.backends.url import URLCamera
from pyro_camera_api.camera.base import BaseCamera
from pyro_camera_api.core.config import CAM_PWD, CAM_USER, RAW_CONFIG

logger = logging.getLogger("CameraRegistry")
logger.setLevel(logging.INFO)

# Global registry of camera objects, keyed by camera id or ip
CAMERA_REGISTRY: Dict[str, BaseCamera] = {}

# Patrol threading state, we will later move the logic to camera.patrol
PATROL_THREADS: Dict[str, threading.Thread] = {}
PATROL_FLAGS: Dict[str, threading.Event] = {}


def build_camera_object(key: str, conf: dict) -> Optional[BaseCamera]:
    """
    Build the appropriate camera object based on configuration.

    Expected keys in conf
      brand   "reolink", "rtsp", "http" or "url"
      type    "ptz" or "static"
      ip_address   optional, default is key
      rtsp_url     for RTSP cameras
      url          for HTTP snapshot cameras
      poses        PTZ positions
      azimuths
      focus_position
    """
    brand = conf.get("brand", "").lower()
    cam_type = conf.get("type", "static").lower()
    ip_addr = conf.get("ip_address", key)

    # Reolink cameras
    if "reolink" in brand:
        cam = ReolinkCamera(
            camera_id=key,
            ip_address=ip_addr,
            username=CAM_USER or "",
            password=CAM_PWD or "",
            cam_type=cam_type,
            cam_poses=conf.get("poses", []),
            cam_azimuths=conf.get("azimuths", []),
            focus_position=conf.get("focus_position"),
        )
        logger.info("Registered Reolink camera %s", key)
        return cam

    # RTSP cameras
    if brand == "rtsp":
        rtsp_url = conf.get("rtsp_url")
        if not rtsp_url:
            logger.error("Camera %s declared as RTSP but missing 'rtsp_url'", key)
            return None

        cam = RTSPCamera(
            camera_id=key,
            rtsp_url=rtsp_url,
            ip_address=ip_addr,
            cam_type="static",
        )
        logger.info("Registered RTSP camera %s", key)
        return cam

    # HTTP or URL snapshot cameras
    if brand in ("http", "https", "url", ""):
        snapshot_url = conf.get("url")
        if not snapshot_url:
            logger.error("Camera %s declared as HTTP snapshot but missing 'url'", key)
            return None

        cam = URLCamera(
            camera_id=key,
            url=snapshot_url,
            cam_type="static",
        )
        logger.info("Registered HTTP snapshot camera %s", key)
        return cam

    logger.error("Unknown camera brand for %s, value was %s", key, brand)
    return None


# Build the global registry at import time
for key, conf in RAW_CONFIG.items():
    try:
        cam = build_camera_object(key, conf)
        if cam is not None:
            CAMERA_REGISTRY[key] = cam
        else:
            logger.warning("Camera %s was not registered due to configuration error", key)
    except Exception as exc:
        logger.error("Failed to initialize camera %s, %s", key, exc)
