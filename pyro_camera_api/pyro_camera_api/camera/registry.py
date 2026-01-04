# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

from pyro_camera_api.camera.adapters.linovision import LinovisionCamera
from pyro_camera_api.camera.adapters.mock import MockCamera
from pyro_camera_api.camera.adapters.reolink import ReolinkCamera
from pyro_camera_api.camera.adapters.rtsp import RTSPCamera
from pyro_camera_api.camera.adapters.url import URLCamera
from pyro_camera_api.camera.base import BaseCamera
from pyro_camera_api.core.config import CAM_PWD, CAM_USER, RAW_CONFIG

logger = logging.getLogger("CameraRegistry")
logger.setLevel(logging.INFO)

# Global registry of camera objects, keyed by camera id
CAMERA_REGISTRY: Dict[str, BaseCamera] = {}

# Patrol threading state, later managed in camera.patrol
PATROL_THREADS: Dict[str, threading.Thread] = {}
PATROL_FLAGS: Dict[str, threading.Event] = {}


def build_camera_object(key: str, conf: dict) -> Optional[BaseCamera]:
    """
    Build the appropriate camera object based on configuration.

    Expected keys in conf:
      adapter:  "reolink", "linovision", "rtsp", "url", "mock"
      type:     "ptz" or "static"
      ip_address
      rtsp_url (if adapter=rtsp)
      url (if adapter=url or adapter=mock)
      poses, azimuths, focus_position (PTZ adapters)
    """
    adapter = conf.get("adapter", "").lower()
    cam_type = conf.get("type", "static").lower()
    ip_addr = conf.get("ip_address", key)

    # Reolink camera
    if "reolink" in adapter:
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

    # Linovision / ISAPI cameras
    if "linovision" in adapter:
        cam = LinovisionCamera(
            camera_id=key,
            ip_address=ip_addr,
            username=conf.get("username", CAM_USER or ""),
            password=conf.get("password", CAM_PWD or ""),
            cam_type=cam_type,
            cam_poses=conf.get("poses", []),
            cam_azimuths=conf.get("azimuths", []),
            protocol=conf.get("protocol", "http"),
            verify_tls=conf.get("verify_tls", False),
            snapshot_channel=conf.get("snapshot_channel", "101"),
            ptz_channel=conf.get("ptz_channel", "1"),
            focus_position=conf.get("focus_position"),
            timeout=conf.get("timeout", 3.0),
            azimuth_offset_deg=conf.get("azimuth_offset_deg", conf.get("azimuth_offset", 0.0)),
            default_elevation_deg=conf.get("timeout", 0),
        )
        logger.info("Registered Linovision camera %s", key)
        return cam

    # RTSP camera (capture only)
    if adapter == "rtsp":
        rtsp_url = conf.get("rtsp_url")
        if not rtsp_url:
            logger.error("Camera %s declared as RTSP adapter but missing 'rtsp_url'", key)
            return None

        cam = RTSPCamera(
            camera_id=key,
            rtsp_url=rtsp_url,
            ip_address=ip_addr,
            cam_type="static",
        )
        logger.info("Registered RTSP camera %s", key)
        return cam

    # URL / HTTP snapshot camera (capture only)
    if adapter in ("url", "http", "https"):
        snapshot_url = conf.get("url")
        if not snapshot_url:
            logger.error("Camera %s declared as URL adapter but missing 'url'", key)
            return None

        cam = URLCamera(
            camera_id=key,
            url=snapshot_url,
            cam_type="static",
        )
        logger.info("Registered URL snapshot camera %s", key)
        return cam

    # Mock camera adapter for tests and demos
    if adapter in ("mock", "mock"):
        image_url = conf.get(
            "url",
            "https://github.com/pyronear/pyro-engine/releases/download/v0.1.1/fire_sample_image.jpg",
        )
        cam = MockCamera(
            camera_id=key,
            image_url=image_url,
            cam_type=cam_type,
            cam_poses=conf.get("poses", []),
            cam_azimuths=conf.get("azimuths", []),
            focus_position=conf.get("focus_position"),
        )
        logger.info(
            "Registered Mock camera %s with image %s and poses %s",
            key,
            image_url,
            conf.get("poses"),
        )
        return cam

    # adapter not recognized
    logger.error("Unknown adapter for %s, value was '%s'", key, adapter)
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
