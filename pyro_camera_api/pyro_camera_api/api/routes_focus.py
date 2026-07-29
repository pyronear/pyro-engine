# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from fastapi import APIRouter, HTTPException

from pyro_camera_api.camera.base import FocusMixin
from pyro_camera_api.camera.focus_manager import full_calibration, stream_is_active, supports_focus_search
from pyro_camera_api.camera.registry import CAMERA_REGISTRY, PATROL_FLAGS, PATROL_THREADS
from pyro_camera_api.utils.time_utils import update_command_time

router = APIRouter()


def _patrol_is_running(camera_ip: str) -> bool:
    thread = PATROL_THREADS.get(camera_ip)
    flag = PATROL_FLAGS.get(camera_ip)
    return bool(thread and thread.is_alive() and (flag is None or not flag.is_set()))


@router.post("/manual")
def manual_focus(camera_ip: str, position: int):
    """
    Set a manual focus level for a camera in [0,1000].

    The camera must support manual focus via FocusMixin.
    `position` is applied directly to the camera's focus motor.
    """
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not isinstance(cam, FocusMixin):
        raise HTTPException(status_code=400, detail="Camera does not support manual focus")

    result = cam.set_manual_focus(position)

    return {
        "status": "manual_focus",
        "camera_ip": camera_ip,
        "position": position,
        "result": result,
    }


@router.post("/set_autofocus")
def toggle_autofocus(camera_ip: str, disable: bool = True):
    """
    Enable or disable autofocus mode on a camera.

    When `disable` is True autofocus is turned off and manual control can be applied.
    When `disable` is False autofocus is activated if supported by the adapter.
    """
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not hasattr(cam, "set_auto_focus"):
        raise HTTPException(status_code=400, detail="Camera does not support autofocus control")

    result = cam.set_auto_focus(disable)

    return {
        "status": "autofocus",
        "camera_ip": camera_ip,
        "disabled": disable,
        "result": result,
    }


@router.get("/status")
def get_focus_status(camera_ip: str):
    """
    Return the current autofocus and zoom information exposed by the camera.

    The adapter must implement get_focus_level which typically returns
    the current focus position and zoom position encoded in a device specific structure.
    """
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not hasattr(cam, "get_focus_level"):
        raise HTTPException(status_code=400, detail="Camera does not expose focus status")

    data = cam.get_focus_level()
    if not data:
        raise HTTPException(status_code=500, detail="Could not retrieve focus level")

    return {"camera_ip": camera_ip, "focus_data": data}


@router.post("/focus_finder")
def run_focus_optimization(camera_ip: str, save_images: bool = False):
    """
    Run the autofocus search algorithm and return the optimal focus position.

    This operation is supported only on PTZ cameras implementing FocusMixin.
    The camera is moved to its calibration pose (second preset when available)
    before the optimization step, under the per-camera lock.
    The optional `save_images` parameter allows storing captured frames generated
    during the autofocus process.

    Returns 409 when the patrol is running for the camera (stop it first),
    when a stream is active, or when the camera is already busy with another
    blocking operation.
    """
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not supports_focus_search(cam):
        raise HTTPException(status_code=400, detail="Camera does not support the autofocus search")

    # The patrol loop moves the camera without taking the move lock, so a
    # focus search running alongside it would measure sharpness on a moving
    # scene. Require the patrol to be stopped first.
    thread = PATROL_THREADS.get(camera_ip)
    if thread is not None and thread.is_alive():
        flag = PATROL_FLAGS.get(camera_ip)
        if flag is not None and flag.is_set():
            raise HTTPException(status_code=409, detail="Patrol is stopping, retry in a few seconds")
        raise HTTPException(
            status_code=409,
            detail="Patrol running, stop the patrol before running focus optimization",
        )

    if stream_is_active(camera_ip):
        raise HTTPException(status_code=409, detail="Stream active, focus optimization refused")

    try:
        # A patrol started mid-search would move the camera under the sweep,
        # so it is also part of the abort predicate (TOCTOU mitigation).
        best_position = full_calibration(
            cam,
            save_images=save_images,
            should_abort=lambda: _patrol_is_running(camera_ip),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Focus search failed: {exc}")
    if best_position is None:
        raise HTTPException(status_code=409, detail="Camera busy, focus optimization refused")

    return {
        "camera_ip": camera_ip,
        "best_focus_position": best_position,
        "status": "focus_updated",
    }
