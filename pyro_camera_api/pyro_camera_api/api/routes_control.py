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


# reolink-823A16 and reolink-823S2 pan+tilt calibrated 2026-03-31 via semi-manual impulse method (zoom=0, settle=3s, R²>0.928).
# Model: δ = ω·T + b  →  T_command = (target_deg - b) / ω
# Bias b represents coast distance after Stop (~1.2s of deceleration at speed ω).
PAN_SPEEDS = {
    "reolink-823S2": {1: 1.5988, 2: 2.7877, 3: 4.5222, 4: 5.7913, 5: 6.3122},
    "reolink-823A16": {1: 1.3748, 2: 2.8895, 3: 4.5352, 4: 6.6175, 5: 7.3933},
}

# Coast bias b (degrees) per speed level — distance added by deceleration after Stop.
# Add to target before dividing by ω: T = (target - PAN_BIAS[adapter][speed]) / ω
PAN_BIAS = {
    "reolink-823S2": {1: 0.6312, 2: 1.4915, 3: 1.748, 4: 2.9926, 5: 4.393},
    "reolink-823A16": {1: 2.0047, 2: 3.6327, 3: 5.5697, 4: 7.5964, 5: 10.176},
}

TILT_SPEEDS = {
    "reolink-823S2": {1: 1.583, 2: 4.0438, 3: 6.9627},
    "reolink-823A16": {1: 2.0749, 2: 4.0741, 3: 5.5923},
}

TILT_BIAS = {
    "reolink-823S2": {1: 1.462, 2: 2.1954, 3: 2.6174},
    "reolink-823A16": {1: 2.2971, 2: 4.5217, 3: 7.0047},
}


def get_pan_speed_per_sec(adapter: str, level: int) -> Optional[float]:
    return PAN_SPEEDS.get(adapter, {}).get(level)


def get_pan_bias(adapter: str, level: int) -> float:
    return PAN_BIAS.get(adapter, {}).get(level, 0.0)


def get_tilt_speed_per_sec(adapter: str, level: int) -> Optional[float]:
    return TILT_SPEEDS.get(adapter, {}).get(level)


def get_tilt_bias(adapter: str, level: int) -> float:
    return TILT_BIAS.get(adapter, {}).get(level, 0.0)


# Measured FOV lookup tables (degrees), zoom levels 0–41.
# Calibrated via QR-code chained-ratio method. Plateau at zoom 41 (optical max).
_H_FOV_TABLE = {
    "reolink-823S2": [
        54.2, 52.206, 50.405, 48.356, 46.167, 44.183, 42.63, 41.058, 39.117, 37.523,
        35.393, 33.804, 32.341, 30.742, 29.446, 27.829, 26.394, 24.992, 23.604, 22.136,
        20.948, 19.675, 18.652, 17.794, 16.352, 15.273, 14.278, 13.287, 12.577, 11.681,
        10.832, 9.992, 9.298, 8.644, 8.022, 7.411, 6.84, 6.323, 5.793, 5.303,
        4.787, 4.183,
    ],
    "reolink-823A16": [
        54.2, 52.029, 50.146, 47.986, 46.384, 44.431, 42.376, 40.915, 38.623, 37.135,
        35.303, 33.894, 32.273, 30.703, 29.167, 27.67, 26.181, 24.921, 23.489, 22.138,
        20.887, 19.701, 18.467, 17.618, 16.244, 15.203, 14.174, 13.242, 12.332, 11.606,
        10.771, 9.993, 9.283, 8.558, 7.914, 7.321, 6.777, 6.241, 5.744, 5.229,
        4.704, 4.118,
    ],
}
_V_FOV_TABLE = {
    "reolink-823S2": [
        41.7, 40.166, 38.78, 37.204, 35.52, 33.993, 32.799, 31.589, 30.096, 28.869,
        27.23, 26.008, 24.882, 23.652, 22.655, 21.411, 20.307, 19.229, 18.16, 17.031,
        16.117, 15.138, 14.351, 13.69, 12.581, 11.751, 10.985, 10.223, 9.676, 8.987,
        8.334, 7.687, 7.154, 6.651, 6.172, 5.702, 5.263, 4.865, 4.457, 4.08,
        3.683, 3.219,
    ],
    "reolink-823A16": [
        41.7, 40.03, 38.581, 36.919, 35.686, 34.184, 32.603, 31.479, 29.716, 28.571,
        27.161, 26.077, 24.83, 23.622, 22.44, 21.289, 20.143, 19.174, 18.072, 17.032,
        16.07, 15.157, 14.208, 13.555, 12.498, 11.697, 10.905, 10.188, 9.488, 8.929,
        8.287, 7.688, 7.142, 6.584, 6.089, 5.633, 5.214, 4.802, 4.42, 4.023,
        3.619, 3.169,
    ],
}

# Default adapter when model is unknown
_DEFAULT_ADAPTER = "reolink-823S2"


def fov_at_zoom(zoom: int, adapter: str | None = None) -> tuple[float, float]:
    """Return (h_fov, v_fov) in degrees using measured lookup table with linear interpolation."""
    key = adapter if adapter in _H_FOV_TABLE else _DEFAULT_ADAPTER
    h_table = _H_FOV_TABLE[key]
    v_table = _V_FOV_TABLE[key]
    z = max(0, min(zoom, 41))
    z0 = int(z)
    z1 = min(z0 + 1, 41)
    t = z - z0
    h = h_table[z0] + t * (h_table[z1] - h_table[z0])
    v = v_table[z0] + t * (v_table[z1] - v_table[z0])
    return h, v


def _pick_speed(
    target_deg: float,
    speeds: dict,
    bias: dict,
    min_duration: float = 0.3,
    max_duration: float = 4.0,
) -> Optional[int]:
    """Return the highest speed level where T = (target - b) / ω is within [min_duration, max_duration]."""
    best: Optional[int] = None
    for level in sorted(speeds.keys()):
        b = bias.get(level, 0.0)
        if target_deg <= b:
            continue
        duration = (target_deg - b) / speeds[level]
        if min_duration <= duration <= max_duration:
            best = level
    return best


@router.post("/click_to_move")
def click_to_move(
    camera_ip: str,
    click_x: int,
    click_y: int,
    image_width: int,
    image_height: int,
    zoom: int = 0,
    h_fov: Optional[float] = None,
    v_fov: Optional[float] = None,
):
    """
    Move a PTZ camera to center on a pixel coordinate clicked in the image.

    Computes the pan/tilt angles from the click offset relative to image center,
    selects the best speed level using calibrated tables, and executes the move.

    click_x / click_y are pixel coordinates in the full-resolution image.
    h_fov / v_fov are optional — if omitted they are derived from zoom using
    the built-in FOV table (Reolink RLC-823 series, linear interpolation).
    """
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera with IP '{camera_ip}' not found")

    if not isinstance(cam, PTZMixin):
        raise HTTPException(status_code=400, detail="Camera does not support PTZ controls")

    conf = RAW_CONFIG.get(camera_ip, {})
    adapter = conf.get("adapter", "unknown")

    if h_fov is None or v_fov is None:
        h_fov, v_fov = fov_at_zoom(zoom, adapter)

    pan_deg = (click_x - image_width / 2) * h_fov / image_width
    tilt_deg = (click_y - image_height / 2) * v_fov / image_height

    pan_speeds = PAN_SPEEDS.get(adapter, {})
    pan_bias = PAN_BIAS.get(adapter, {})
    tilt_speeds = TILT_SPEEDS.get(adapter, {})
    tilt_bias = TILT_BIAS.get(adapter, {})

    result: dict = {
        "status": "ok",
        "camera_ip": camera_ip,
        "adapter": adapter,
        "pan_deg": round(pan_deg, 3),
        "tilt_deg": round(tilt_deg, 3),
        "moves": [],
    }

    try:
        # Pan
        if abs(pan_deg) >= 0.5:
            pan_direction = "Right" if pan_deg > 0 else "Left"
            pan_speed = _pick_speed(abs(pan_deg), pan_speeds, pan_bias)
            if pan_speed is not None:
                b = pan_bias.get(pan_speed, 0.0)
                duration = (abs(pan_deg) - b) / pan_speeds[pan_speed]
                logger.info("[%s] click_to_move pan %s %.2f° speed=%s dur=%.2fs", camera_ip, pan_direction, abs(pan_deg), pan_speed, duration)
                cam.move_camera(pan_direction, speed=pan_speed)
                time.sleep(duration)
                cam.move_camera("Stop")
                result["moves"].append({"axis": "pan", "direction": pan_direction, "deg": round(abs(pan_deg), 3), "speed": pan_speed, "duration": round(duration, 2)})
            elif pan_speeds:
                # Angle too small for calibrated model — micro-impulse: move+stop back-to-back
                logger.info("[%s] click_to_move pan %s %.2f° micro-impulse speed=1", camera_ip, pan_direction, abs(pan_deg))
                cam.move_camera(pan_direction, speed=1)
                cam.move_camera("Stop")
                result["moves"].append({"axis": "pan", "direction": pan_direction, "deg": round(abs(pan_deg), 3), "speed": 1, "duration": 0, "micro": True})
            else:
                result["moves"].append({"axis": "pan", "skipped": True, "reason": "no speed table for adapter"})

        # Tilt
        if abs(tilt_deg) >= 0.5:
            tilt_direction = "Down" if tilt_deg > 0 else "Up"
            tilt_speed = _pick_speed(abs(tilt_deg), tilt_speeds, tilt_bias)
            if tilt_speed is not None:
                b = tilt_bias.get(tilt_speed, 0.0)
                duration = (abs(tilt_deg) - b) / tilt_speeds[tilt_speed]
                logger.info("[%s] click_to_move tilt %s %.2f° speed=%s dur=%.2fs", camera_ip, tilt_direction, abs(tilt_deg), tilt_speed, duration)
                cam.move_camera(tilt_direction, speed=tilt_speed)
                time.sleep(duration)
                cam.move_camera("Stop")
                result["moves"].append({"axis": "tilt", "direction": tilt_direction, "deg": round(abs(tilt_deg), 3), "speed": tilt_speed, "duration": round(duration, 2)})
            elif tilt_speeds:
                logger.info("[%s] click_to_move tilt %s %.2f° micro-impulse speed=1", camera_ip, tilt_direction, abs(tilt_deg))
                cam.move_camera(tilt_direction, speed=1)
                cam.move_camera("Stop")
                result["moves"].append({"axis": "tilt", "direction": tilt_direction, "deg": round(abs(tilt_deg), 3), "speed": 1, "duration": 0, "micro": True})
            else:
                result["moves"].append({"axis": "tilt", "skipped": True, "reason": "no speed table for adapter"})

    except Exception as exc:
        logger.error("[%s] click_to_move error: %s", camera_ip, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return result


@router.post("/move")
def move_camera(
    camera_ip: str,
    direction: Optional[str] = None,
    speed: int = 10,
    pose_id: Optional[int] = None,
    degrees: Optional[float] = None,
    duration: Optional[float] = None,
):
    """
    Move a PTZ camera for a short action.

    This endpoint supports four modes of operation based on the request
    parameters.

    If pose_id is provided the camera moves to the given preset pose.
    If duration and direction are provided the camera moves for exactly
    that duration (seconds) server-side, then stops. Use duration=0 for
    a micro-impulse (move+stop back-to-back).
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

        if duration is not None and direction:
            logger.info("[%s] Moving %s for %.3fs at speed %s (adapter=%s)", camera_ip, direction, duration, speed, adapter)
            cam.move_camera(direction, speed=speed)
            if duration > 0:
                time.sleep(duration)
            cam.move_camera("Stop")
            return {
                "status": "ok",
                "camera_ip": camera_ip,
                "direction": direction,
                "duration": round(duration, 3),
                "speed": speed,
                "adapter": adapter,
            }

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

            if direction in ["Left", "Right"]:
                bias = get_pan_bias(adapter, speed)
            else:  # Up / Down — already validated above
                bias = get_tilt_bias(adapter, speed)
            duration_sec = max(0.0, (abs(degrees) - bias) / deg_per_sec)

            # Micro-impulse fallback: requested angle is below bias at this speed
            micro = duration_sec == 0.0 and abs(degrees) > 0
            if micro:
                logger.info("[%s] Moving %s %.2f° micro-impulse speed=1 (adapter=%s)", camera_ip, direction, abs(degrees), adapter)
                cam.move_camera(direction, speed=1)
                cam.move_camera("Stop")
            else:
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

            logger.info("[%s] Movement %s stopped after ~%.2fs", camera_ip, direction, 0.0 if micro else duration_sec)

            return {
                "status": "ok",
                "camera_ip": camera_ip,
                "direction": direction,
                "degrees": degrees,
                "duration": 0 if micro else round(duration_sec, 2),
                "speed": 1 if micro else speed,
                "adapter": adapter,
                "micro": micro,
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
