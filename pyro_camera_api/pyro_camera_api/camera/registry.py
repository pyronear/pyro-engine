# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

from pyro_camera_api.camera.backends.fake import FakeCamera
from pyro_camera_api.camera.backends.reolink import ReolinkCamera
from pyro_camera_api.camera.backends.rtsp import RTSPCamera
from pyro_camera_api.camera.backends.url import URLCamera
from pyro_camera_api.camera.base import BaseCamera
from pyro_camera_api.core.config import CAM_PWD, CAM_USER, RAW_CONFIG

logger = logging.getLogger("CameraRegistry")
logger.setLevel(logging.INFO)

CAMERA_REGISTRY: Dict[str, BaseCamera] = {}
PATROL_THREADS: Dict[str, threading.Thread] = {}
PATROL_FLAGS: Dict[str, threading.Event] = {}


def build_camera_object(key: str, conf: dict) -> Optional[BaseCamera]:
    """
    Build the appropriate camera object based on configuration.

    Expected keys in conf:
      backend: "reolink", "rtsp", "url", "fake" (or legacy "brand")
      type: "ptz" or "static"
      ip_address: for Reolink or RTSP backends
      rtsp_url: if backend is rtsp
      url: if backend is url or fake
      poses, azimuths, focus_position: for Reolink only
    """
    backend = (conf.get("backend") or conf.get("brand") or "").lower()
    cam_type = conf.get("type", "static").lower()
    ip_addr = conf.get("ip_address", key)

    if "reolink" in backend:
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

    if backend == "rtsp":
        rtsp_url = conf.get("rtsp_url") or (key if str(key).startswith("rtsp://") else None)
        if not rtsp_url:
            logger.error("Camera %s declared as RTSP backend but missing 'rtsp_url'", key)
            return None

        cam = RTSPCamera(
            camera_id=key,
            rtsp_url=rtsp_url,
            ip_address=ip_addr,
            cam_type="static",
        )
        logger.info("Registered RTSP camera %s", key)
        return cam

    if backend in ("url", "http", "https"):
        snapshot_url = conf.get("url") or (key if str(key).startswith("http") else None)
        if not snapshot_url:
            logger.error("Camera %s declared as URL backend but missing 'url'", key)
            return None

        cam = URLCamera(
            camera_id=key,
            url=snapshot_url,
            cam_type="static",
        )
        logger.info("Registered URL snapshot camera %s", key)
        return cam

    if backend == "fake":
        image_url = conf.get(
            "url",
            "https://github.com/pyronear/pyro-engine/releases/download/v0.1.1/fire_sample_image.jpg",
        )
        cam = FakeCamera(
            camera_id=key,
            image_url=image_url,
            cam_type="static",
        )
        logger.info("Registered Fake camera %s with image %s", key, image_url)
        return cam

    logger.error("Unknown backend for %s, value was '%s'", key, backend)
    return None


for key, conf in RAW_CONFIG.items():
    try:
        cam = build_camera_object(key, conf)
        if cam is not None:
            CAMERA_REGISTRY[key] = cam
        else:
            logger.warning("Camera %s was not registered due to configuration error", key)
    except Exception as exc:
        logger.error("Failed to initialize camera %s, %s", key, exc)
