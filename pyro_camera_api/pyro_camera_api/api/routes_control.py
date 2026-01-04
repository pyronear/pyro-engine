# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from pyro_camera_api.camera.base import PTZMixin
from pyro_camera_api.camera.registry import CAMERA_REGISTRY
from pyro_camera_api.core.config import RAW_CONFIG
from pyro_camera_api.utils.time_utils import update_command_time

router = APIRouter()
logger = logging.getLogger(__name__)


PAN_SPEEDS = {
    "reolink-823S2": {1: 1.4723, 2: 2.7747, 3: 4.2481, 4: 5.6113, 5: 7.3217},
    "reolink-823A16": {1: 1.4403, 2: 2.714, 3: 4.1801, 4: 5.6259, 5: 7.2743},
}

TILT_SPEEDS = {
    "reolink-823S2": {1: 2.1392, 2: 3.9651, 3: 6.0554},
    "reolink-823A16": {1: 1.7998, 2: 3.6733, 3: 5.5243},
}


def get_pan_speed_per_sec(adapter: str, level: int) -> Optional[float]:
    return PAN_SPEEDS.get(adapter, {}).get(level)


def get_tilt_speed_per_sec(adapter: str, level: int) -> Optional[float]:
    return TILT_SPEEDS.get(adapter, {}).get(level)


@router.post("/move")
def move_camera(
    camera_ip: str,
    direction: Optional[str] = None,
    speed: int = 10,
    pose_id: Optional[int] = None,
    degrees: Optional[float] = None,
):
    """
    Move a PTZ camera for a short action.

    This endpoint supports three modes of operation based on the request
    parameters.

    If pose_id is provided the camera moves to the given preset pose.
    If degrees and direction are provided the camera moves for the
    duration needed to cover the requested angle using a model specific
    speed table.
    If only direction is provided the camera starts moving in that
    direction at the requested speed without a fixed duration.
    """
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera with IP '{camera_ip}' not found")

    if not isinstance(cam, PTZMixin):
        raise HTTPException(status_code=400, detail="Camera does not support PTZ controls")

    conf = RAW_CONFIG.get(camera_ip, {})
    adapter = conf.get("adapter", "unknown")

    try:
        if pose_id is not None:
            logger.info("[%s] Moving to preset pose %s at speed %s", camera_ip, pose_id, speed)
            cam.move_camera("ToPos", speed=speed, idx=pose_id)
            return {"status": "ok", "camera_ip": camera_ip, "pose_id": pose_id, "speed": speed}

        if degrees is not None and direction:
            if direction in ["Left", "Right"]:
                deg_per_sec = get_pan_speed_per_sec(adapter, speed)
            elif direction in ["Up", "Down"]:
                deg_per_sec = get_tilt_speed_per_sec(adapter, speed)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported direction '{direction}'")

            if deg_per_sec is None:
                # Fallback for adapters without calibrated speed tables (e.g., linovision)
                try:
                    deg_per_sec = max(0.1, float(speed))
                except Exception:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported adapter '{adapter}' or speed level {speed}",
                    )

            duration_sec = abs(degrees) / deg_per_sec
            logger.info(
                "[%s] Moving %s for %.2fs at speed %s (adapter=%s)",
                camera_ip,
                direction,
                duration_sec,
                speed,
                adapter,
            )

            cam.move_camera(direction, speed=speed)
            time.sleep(duration_sec)
            cam.move_camera("Stop")

            logger.info("[%s] Movement %s stopped after ~%.2fs", camera_ip, direction, duration_sec)

            return {
                "status": "ok",
                "camera_ip": camera_ip,
                "direction": direction,
                "degrees": degrees,
                "duration": round(duration_sec, 2),
                "speed": speed,
                "adapter": adapter,
            }

        if direction:
            logger.info("[%s] Moving %s at speed %s", camera_ip, direction, speed)
            cam.move_camera(direction, speed=speed)
            return {"status": "ok", "camera_ip": camera_ip, "direction": direction, "speed": speed}

        raise HTTPException(
            status_code=400,
            detail="Either pose_id, degrees plus direction, or direction must be specified",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[%s] Movement error: %s", camera_ip, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/stop/{camera_ip}")
def stop_camera(camera_ip: str):
    """
    Stop the current movement of a PTZ camera.

    This endpoint sends a Stop command to the camera PTZ control.
    If the camera is not registered or does not support PTZ controls
    an error is returned.
    """
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera with IP '{camera_ip}' not found")

    if not isinstance(cam, PTZMixin):
        raise HTTPException(status_code=400, detail="Camera does not support PTZ controls")

    try:
        cam.move_camera("Stop")
        logger.info("[%s] Movement stopped", camera_ip)
        return {"message": f"Camera {camera_ip} stopped moving"}
    except Exception as exc:
        logger.error("[%s] Failed to stop movement: %s", camera_ip, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/preset/list")
def list_presets(camera_ip: str):
    """
    List all configured PTZ presets for a camera.

    The camera must provide a get_ptz_preset method.
    The response contains the raw preset structure as returned by
    the camera adapter.
    """
    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not hasattr(cam, "get_ptz_preset"):
        raise HTTPException(status_code=400, detail="Camera does not support presets")

    presets = cam.get_ptz_preset()
    return {"camera_ip": camera_ip, "presets": presets}


@router.post("/preset/set")
def set_preset(camera_ip: str, idx: Optional[int] = None):
    """
    Create or update a PTZ preset on a camera.

    If idx is provided the preset is set or overwritten at this index.
    If idx is not provided the adapter chooses the first free preset
    slot if such a behavior is implemented on the camera side.
    """
    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not hasattr(cam, "set_ptz_preset"):
        raise HTTPException(status_code=400, detail="Camera does not support presets")

    cam.set_ptz_preset(idx=idx)
    return {"status": "preset_set", "camera_ip": camera_ip, "id": idx}


@router.post("/zoom/{camera_ip}/{level}")
def zoom_camera(camera_ip: str, level: int):
    """
    Adjust the optical zoom level of a camera between 0 and 64.

    The camera must implement a start_zoom_focus method.
    Level represents the target zoom position accepted by the camera
    which is usually an integer range specific to the adapter.
    """
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera with IP '{camera_ip}' not found")

    if not (0 <= level <= 64):
        raise HTTPException(status_code=400, detail="Zoom level must be between 0 and 64")

    if not hasattr(cam, "start_zoom_focus"):
        raise HTTPException(status_code=400, detail="Camera does not support zoom control")

    try:
        cam.start_zoom_focus(level)
        logger.info("[%s] Zoom set to %s", camera_ip, level)
        return {"message": f"Zoom set to {level}", "camera_ip": camera_ip}
    except Exception as exc:
        logger.error("[%s] Failed to set zoom: %s", camera_ip, exc)
        raise HTTPException(status_code=500, detail=str(exc))
