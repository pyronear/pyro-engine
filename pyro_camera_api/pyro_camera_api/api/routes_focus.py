# pyro_camera_api/api/routes_focus.py
# Copyright (C) 2020-2025, Pyronear.
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from pyro_camera_api.camera.base import FocusMixin, PTZMixin
from pyro_camera_api.camera.registry import CAMERA_REGISTRY
from pyro_camera_api.utils.time_utils import update_command_time

router = APIRouter()


@router.post("/manual")
def manual_focus(camera_ip: str, position: int):
    """Set manual focus to a specific position."""
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not isinstance(cam, FocusMixin):
        raise HTTPException(status_code=400, detail="Camera does not support manual focus")

    # Reolink implementation returns a dict, but we keep it opaque here
    result = cam.set_manual_focus(position)  # type: ignore[call-arg]

    return {
        "status": "manual_focus",
        "camera_ip": camera_ip,
        "position": position,
        "result": result,
    }


@router.post("/set_autofocus")
def toggle_autofocus(camera_ip: str, disable: bool = True):
    """
    Enable or disable camera autofocus if supported.

    Reolink cameras expose set_auto_focus(disable).
    """
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not hasattr(cam, "set_auto_focus"):
        raise HTTPException(status_code=400, detail="Camera does not support autofocus control")

    result = cam.set_auto_focus(disable)  # type: ignore[call-arg]

    return {
        "status": "autofocus",
        "camera_ip": camera_ip,
        "disabled": disable,
        "result": result,
    }


@router.get("/status")
def get_focus_status(camera_ip: str):
    """
    Return current focus and zoom level if supported.

    Reolink cameras expose get_focus_level().
    """
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not hasattr(cam, "get_focus_level"):
        raise HTTPException(status_code=400, detail="Camera does not expose focus status")

    data = cam.get_focus_level()  # type: ignore[call-arg]
    if not data:
        raise HTTPException(status_code=500, detail="Could not retrieve focus level")

    return {"camera_ip": camera_ip, "focus_data": data}


@router.post("/focus_finder")
def run_focus_optimization(camera_ip: str, save_images: bool = False):
    """
    Run the autofocus search algorithm on a PTZ camera that supports FocusMixin.

    Optionally move to pose index one first if cam_poses is defined.
    """
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not isinstance(cam, FocusMixin):
        raise HTTPException(status_code=400, detail="Camera does not support autofocus")

    # Static cameras are explicitly not supported
    if getattr(cam, "cam_type", "static") == "static":
        raise HTTPException(status_code=400, detail="Autofocus is not supported for static cameras")

    # Optional move to pose one if PTZ and poses are defined
    if isinstance(cam, PTZMixin):
        cam_poses = getattr(cam, "cam_poses", None)
        if cam_poses and len(cam_poses) > 1:
            pose1 = cam_poses[1]
            try:
                cam.move_camera("ToPos", idx=pose1, speed=50)  # type: ignore[call-arg]
                time.sleep(1)
            except Exception:
                # Do not fail autofocus just because preset move failed
                pass

    best_position = cam.focus_finder(save_images=save_images)  # type: ignore[call-arg]

    return {
        "camera_ip": camera_ip,
        "best_focus_position": best_position,
        "status": "focus_updated",
    }
