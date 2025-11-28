# Copyright (C) 2022-2025, Pyronear.

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
    If the camera exposes PTZ presets the algorithm tries moving to the second
    preset before the optimization step when available.
    The optional `save_images` parameter allows storing captured frames generated
    during the autofocus process.
    """
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not isinstance(cam, FocusMixin):
        raise HTTPException(status_code=400, detail="Camera does not support autofocus")

    if getattr(cam, "cam_type", "static") == "static":
        raise HTTPException(status_code=400, detail="Autofocus is not supported for static cameras")

    if isinstance(cam, PTZMixin):
        cam_poses = getattr(cam, "cam_poses", None)
        if cam_poses and len(cam_poses) > 1:
            pose1 = cam_poses[1]
            try:
                cam.move_camera("ToPos", idx=pose1, speed=50)
                time.sleep(1)
            except Exception:
                pass

    best_position = cam.focus_finder(save_images=save_images)

    return {
        "camera_ip": camera_ip,
        "best_focus_position": best_position,
        "status": "focus_updated",
    }
