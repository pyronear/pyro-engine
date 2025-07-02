# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from camera.config import RAW_CONFIG
from camera.registry import CAMERA_REGISTRY
from camera.time_utils import update_command_time

router = APIRouter()


# Lookup dictionaries for pan and tilt speeds
PAN_SPEEDS = {
    "reolink-823S2": {1: 1.4723, 2: 2.7747, 3: 4.2481, 4: 5.6113, 5: 7.3217},
    "reolink-823A16": {1: 1.4403, 2: 2.714, 3: 4.1801, 4: 5.6259, 5: 7.2743},
}

TILT_SPEEDS = {
    "reolink-823S2": {1: 2.1392, 2: 3.9651, 3: 6.0554},
    "reolink-823A16": {1: 1.7998, 2: 3.6733, 3: 5.5243},
}


def get_pan_speed_per_sec(brand: str, level: int) -> Optional[float]:
    return PAN_SPEEDS.get(brand, {}).get(level)


def get_tilt_speed_per_sec(brand: str, level: int) -> Optional[float]:
    return TILT_SPEEDS.get(brand, {}).get(level)


@router.post("/move")
def move_camera(
    camera_ip: str,
    direction: Optional[str] = None,
    speed: int = 10,
    pose_id: Optional[int] = None,
    degrees: Optional[float] = None,
):
    """
    Moves the camera:
    - If 'pose_id' is provided, move to the preset pose.
    - If 'degrees' is provided, move that many degrees in the given direction.
    - Otherwise, move in the specified 'direction' (Up, Down, Left, Right).
    """
    update_command_time()

    if camera_ip not in CAMERA_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Camera with IP '{camera_ip}' not found")

    cam = CAMERA_REGISTRY[camera_ip]
    brand = RAW_CONFIG.get(camera_ip, {}).get("brand", "unknown")

    try:
        if pose_id is not None:
            logging.info(f"[{camera_ip}] Moving to preset pose {pose_id} at speed {speed}")
            cam.move_camera("ToPos", speed=speed, idx=pose_id)
            return {"status": "ok", "camera_ip": camera_ip, "pose_id": pose_id, "speed": speed}

        if degrees is not None and direction:
            if direction in ["Left", "Right"]:
                deg_per_sec = get_pan_speed_per_sec(brand, speed)
            elif direction in ["Up", "Down"]:
                deg_per_sec = get_tilt_speed_per_sec(brand, speed)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported direction '{direction}'")

            if deg_per_sec is None:
                raise HTTPException(status_code=400, detail=f"Unsupported brand '{brand}' or speed level {speed}")

            duration_sec = abs(degrees) / deg_per_sec
            logging.info(f"[{camera_ip}] Moving {direction} for {duration_sec:.2f}s at speed {speed} (brand={brand})")

            cam.move_camera(direction, speed=speed)
            time.sleep(duration_sec)
            cam.move_camera("Stop")

            logging.info(f"[{camera_ip}] Movement {direction} stopped after ~{duration_sec:.2f}s")

            return {
                "status": "ok",
                "camera_ip": camera_ip,
                "direction": direction,
                "degrees": degrees,
                "duration": round(duration_sec, 2),
                "speed": speed,
                "brand": brand,
            }

        if direction:
            logging.info(f"[{camera_ip}] Moving {direction} at speed {speed}")
            cam.move_camera(direction, speed=speed)
            return {"status": "ok", "camera_ip": camera_ip, "direction": direction, "speed": speed}

        raise HTTPException(
            status_code=400,
            detail="Either pose_id, degrees+direction, or direction must be specified",
        )

    except Exception as e:
        logging.error(f"[{camera_ip}] Movement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{camera_ip}")
def stop_camera(camera_ip: str):
    """Stops the camera movement."""
    update_command_time()
    if camera_ip not in CAMERA_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Camera with IP '{camera_ip}' not found")

    cam = CAMERA_REGISTRY[camera_ip]

    try:
        cam.move_camera("Stop")
        logging.info(f"[{camera_ip}] Movement stopped")
        return {"message": f"Camera {camera_ip} stopped moving"}
    except Exception as e:
        logging.error(f"[{camera_ip}] Failed to stop movement: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preset/list")
def list_presets(camera_ip: str):
    cam = CAMERA_REGISTRY[camera_ip]
    presets = cam.get_ptz_preset()
    return {"camera_ip": camera_ip, "presets": presets}


@router.post("/preset/set")
def set_preset(camera_ip: str, idx: Optional[int] = None):
    cam = CAMERA_REGISTRY[camera_ip]
    cam.set_ptz_preset(idx=idx)
    return {"status": "preset_set", "camera_ip": camera_ip, "id": idx}


@router.post("/zoom/{camera_ip}/{level}")
def zoom_camera(camera_ip: str, level: int):
    """Adjusts the camera zoom level (0 to 64)."""
    update_command_time()
    if camera_ip not in CAMERA_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Camera with IP '{camera_ip}' not found")

    if not (0 <= level <= 64):
        raise HTTPException(status_code=400, detail="Zoom level must be between 0 and 64")

    cam = CAMERA_REGISTRY[camera_ip]

    try:
        cam.start_zoom_focus(level)
        logging.info(f"[{camera_ip}] Zoom set to {level}")
        return {"message": f"Zoom set to {level}", "camera_ip": camera_ip}
    except Exception as e:
        logging.error(f"[{camera_ip}] Failed to set zoom: {e}")
        raise HTTPException(status_code=500, detail=str(e))
