# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from pyro_camera_api.camera.base import PTZMixin
from pyro_camera_api.camera.registry import CAMERA_REGISTRY
from pyro_camera_api.core.config import RAW_CONFIG
from pyro_camera_api.utils.time_utils import update_command_time

router = APIRouter()
logger = logging.getLogger(__name__)


# Calibrated 2026-04-02 via automated ORB keypoint matching, server-side timing (no VPN latency).
# Model: δ = ω·T + b  →  T_command = (target_deg - b) / ω
# Bias b represents coast distance after Stop (mechanical inertia).
#
# IMPORTANT: Reolink cameras internally limit PTZ speed at zoom > 0.
# At zoom > 0, all speed levels are capped to ~1.5 °/s (same as speed 1).
# Multi-speed tables are only valid at zoom 0. For zoom > 0, always use speed 1.
# See tools/ptz_zoom_speed_calibration_report.md for full research data.

# Pan speed ω (°/s) at zoom 0
PAN_SPEEDS = {
    "reolink-823S2": {1: 1.4034, 2: 2.5692, 3: 4.1081, 4: 5.7028, 5: 7.1806},
    "reolink-823A16": {1: 1.4782, 2: 2.9035, 3: 4.5721, 4: 6.1209, 5: 7.8310},
}

# Pan bias b (°) at zoom 0
PAN_BIAS = {
    "reolink-823S2": {1: 0.6098, 2: 1.3402, 3: 1.6064, 4: 1.8333, 5: 2.4465},
    "reolink-823A16": {1: 1.5604, 2: 3.1656, 3: 4.8206, 4: 6.5182, 5: 8.4503},
}

# Tilt speed ω (°/s) at zoom 0
TILT_SPEEDS = {
    "reolink-823S2": {1: 2.0094, 2: 3.7474, 3: 5.2022},
    "reolink-823A16": {1: 1.9432, 2: 3.7885, 3: 5.7655},
}

# Tilt bias b (°) at zoom 0
TILT_BIAS = {
    "reolink-823S2": {1: 0.8354, 2: 1.7726, 3: 2.9208},
    "reolink-823A16": {1: 2.1793, 2: 4.2829, 3: 6.3717},
}


def get_pan_speed_per_sec(adapter: str, level: int) -> Optional[float]:
    return PAN_SPEEDS.get(adapter, {}).get(level)


def get_pan_bias(adapter: str, level: int) -> float:
    return PAN_BIAS.get(adapter, {}).get(level, 0.0)


def get_tilt_speed_per_sec(adapter: str, level: int) -> Optional[float]:
    return TILT_SPEEDS.get(adapter, {}).get(level)


def get_tilt_bias(adapter: str, level: int) -> float:
    return TILT_BIAS.get(adapter, {}).get(level, 0.0)


# Measured FOV lookup tables (degrees), zoom levels 0-41.
# Calibrated via QR-code chained-ratio method. Plateau at zoom 41 (optical max).
# Loaded from fov_tables.json to keep this file concise.
_FOV_FILE = Path(__file__).parent / "fov_tables.json"
with _FOV_FILE.open() as _f:
    _FOV_DATA = json.load(_f)
_H_FOV_TABLE: dict[str, list[float]] = _FOV_DATA["h_fov"]
_V_FOV_TABLE: dict[str, list[float]] = _FOV_DATA["v_fov"]

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
    zoom: int = 0,
    min_duration: float = 0.3,
    max_duration: float = 4.0,
) -> Optional[int]:
    """Return the highest speed level where T = (target - b) / ω is within [min_duration, max_duration].

    For moves longer than max_duration at the highest available speed, that highest
    speed is still returned (we accept a longer duration rather than falling back to
    a micro-impulse on a large angle).

    When zoom > 0, only speed 1 is considered because Reolink cameras
    internally cap all speed levels to ~1.5 °/s at higher zoom.
    """
    best: Optional[int] = None
    allowed = {1: speeds[1]} if (zoom > 0 and 1 in speeds) else speeds
    sorted_levels = sorted(allowed.keys())
    for level in sorted_levels:
        b = bias.get(level, 0.0)
        if target_deg <= b:
            continue
        duration = (target_deg - b) / allowed[level]
        if min_duration <= duration <= max_duration:
            best = level

    # Large angle: no level fits under max_duration → use the top speed anyway.
    if best is None and sorted_levels:
        top = sorted_levels[-1]
        b = bias.get(top, 0.0)
        if target_deg > b and (target_deg - b) / allowed[top] > max_duration:
            best = top
    return best


@router.post("/click_to_move")
def click_to_move(
    camera_ip: str,
    click_x: float,
    click_y: float,
):
    """
    Move a PTZ camera to center on a click in the image.

    click_x / click_y are normalized coordinates in [0, 1] (0 = left/top,
    1 = right/bottom). The current zoom level is read from the camera and
    the FOV is looked up from the calibrated table for the camera's adapter.
    """
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera with IP '{camera_ip}' not found")

    if not isinstance(cam, PTZMixin):
        raise HTTPException(status_code=400, detail="Camera does not support PTZ controls")

    conf = RAW_CONFIG.get(camera_ip, {})
    adapter = conf.get("adapter", "unknown")

    zoom = 0
    if hasattr(cam, "get_focus_level"):
        try:
            info = cam.get_focus_level() or {}
            z = info.get("zoom")
            if z is not None:
                zoom = int(z)
        except Exception as exc:
            logger.warning("[%s] click_to_move: failed to read zoom, assuming 0: %s", camera_ip, exc)

    h_fov, v_fov = fov_at_zoom(zoom, adapter)
    logger.info(
        "[%s] click_to_move: click=(%.3f,%.3f) zoom=%s adapter=%s h_fov=%.2f v_fov=%.2f",
        camera_ip, click_x, click_y, zoom, adapter, h_fov, v_fov,
    )

    pan_deg = (click_x - 0.5) * h_fov
    tilt_deg = (click_y - 0.5) * v_fov

    pan_speeds = PAN_SPEEDS.get(adapter, {})
    pan_bias = PAN_BIAS.get(adapter, {})
    tilt_speeds = TILT_SPEEDS.get(adapter, {})
    tilt_bias = TILT_BIAS.get(adapter, {})

    if zoom > 0:
        logger.warning("[%s] click_to_move: zoom=%s > 0, speed limited to 1", camera_ip, zoom)

    result: dict = {
        "status": "ok",
        "camera_ip": camera_ip,
        "adapter": adapter,
        "zoom": zoom,
        "h_fov": round(h_fov, 3),
        "v_fov": round(v_fov, 3),
        "pan_deg": round(pan_deg, 3),
        "tilt_deg": round(tilt_deg, 3),
        "moves": [],
    }
    if zoom > 0:
        result["warning"] = f"speed limited to 1 (zoom={zoom} > 0)"

    def _execute_axis(axis: str, deg: float, direction: str, speeds: dict, bias: dict) -> None:
        """Execute a single-axis move: calibrated duration, or micro-pulse if below bias."""
        speed_level = _pick_speed(abs(deg), speeds, bias, zoom=zoom)
        if speed_level is not None:
            b = bias.get(speed_level, 0.0)
            dur = (abs(deg) - b) / speeds[speed_level]
            logger.info(
                "[%s] click_to_move %s %s %.2f° speed=%s dur=%.2fs",
                camera_ip,
                axis,
                direction,
                abs(deg),
                speed_level,
                dur,
            )
            cam.move_camera(direction, speed=speed_level)
            time.sleep(dur)
            cam.move_camera("Stop")
            result["moves"].append({
                "axis": axis,
                "direction": direction,
                "deg": round(abs(deg), 3),
                "speed": speed_level,
                "duration": round(dur, 2),
            })
        elif speeds:
            # Angle below bias — micro-impulse at speed 1
            logger.info(
                "[%s] click_to_move %s %s %.2f° micro-impulse speed=1", camera_ip, axis, direction, abs(deg)
            )
            cam.move_camera(direction, speed=1)
            cam.move_camera("Stop")
            result["moves"].append({
                "axis": axis,
                "direction": direction,
                "deg": round(abs(deg), 3),
                "speed": 1,
                "duration": 0,
                "micro": True,
            })
        else:
            result["moves"].append({"axis": axis, "skipped": True, "reason": "no speed table for adapter"})

    try:
        # Skip if angle < half the micro-pulse displacement (bias at speed 1)
        pan_min = pan_bias.get(1, 1.0) / 2
        tilt_min = tilt_bias.get(1, 1.0) / 2
        if abs(pan_deg) >= pan_min:
            _execute_axis("pan", pan_deg, "Right" if pan_deg > 0 else "Left", pan_speeds, pan_bias)
        if abs(tilt_deg) >= tilt_min:
            _execute_axis("tilt", tilt_deg, "Down" if tilt_deg > 0 else "Up", tilt_speeds, tilt_bias)

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
    zoom: int = 0,
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
    speed table. When zoom > 0, speed is forced to 1.
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
            logger.info(
                "[%s] Moving %s for %.3fs at speed %s (adapter=%s)", camera_ip, direction, duration, speed, adapter
            )
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
            # Reolink cameras cap all speed levels to ~1.5 °/s at zoom > 0
            speed_limited = zoom > 0 and speed != 1
            effective_speed = 1 if zoom > 0 else speed
            if speed_limited:
                logger.warning("[%s] zoom=%s > 0: speed forced to 1 (requested %s)", camera_ip, zoom, speed)

            if direction in ["Left", "Right"]:
                deg_per_sec = get_pan_speed_per_sec(adapter, effective_speed)
            elif direction in ["Up", "Down"]:
                deg_per_sec = get_tilt_speed_per_sec(adapter, effective_speed)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported direction '{direction}'")

            if deg_per_sec is None:
                # Fallback for adapters without calibrated speed tables (e.g., linovision)
                try:
                    deg_per_sec = max(0.1, float(effective_speed))
                except Exception:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported adapter '{adapter}' or speed level {effective_speed}",
                    )

            bias = (
                get_pan_bias(adapter, effective_speed)
                if direction in ["Left", "Right"]
                else get_tilt_bias(adapter, effective_speed)
            )
            duration_sec = max(0.0, (abs(degrees) - bias) / deg_per_sec)

            # Micro-impulse: angle below bias → move+stop at speed 1
            micro = duration_sec == 0.0 and abs(degrees) > 0
            if micro:
                logger.info(
                    "[%s] Moving %s %.2f° micro-impulse speed=1 (adapter=%s)",
                    camera_ip,
                    direction,
                    abs(degrees),
                    adapter,
                )
                cam.move_camera(direction, speed=1)
                cam.move_camera("Stop")
            else:
                logger.info(
                    "[%s] Moving %s for %.2fs at speed %s (adapter=%s)",
                    camera_ip,
                    direction,
                    duration_sec,
                    effective_speed,
                    adapter,
                )
                cam.move_camera(direction, speed=effective_speed)
                time.sleep(duration_sec)
                cam.move_camera("Stop")

            resp: dict = {
                "status": "ok",
                "camera_ip": camera_ip,
                "direction": direction,
                "degrees": degrees,
                "duration": 0 if micro else round(duration_sec, 2),
                "speed": 1 if micro else effective_speed,
                "adapter": adapter,
                "micro": micro,
            }
            if speed_limited:
                resp["warning"] = f"speed forced to 1 (zoom={zoom} > 0, requested speed={speed})"
            return resp

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


@router.get("/speed_tables")
def get_speed_tables(camera_ip: str):
    """Return the calibrated speed and bias tables for the given camera's adapter."""
    conf = RAW_CONFIG.get(camera_ip, {})
    adapter = conf.get("adapter", "unknown")
    return {
        "adapter": adapter,
        "pan_speeds": PAN_SPEEDS.get(adapter, {}),
        "pan_bias": PAN_BIAS.get(adapter, {}),
        "tilt_speeds": TILT_SPEEDS.get(adapter, {}),
        "tilt_bias": TILT_BIAS.get(adapter, {}),
    }


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
