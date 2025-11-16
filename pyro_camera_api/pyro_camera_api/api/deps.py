# Copyright (C) 2022-2025, Pyronear.
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

from fastapi import HTTPException
from pyro_camera_api.camera.base import BaseCamera
from pyro_camera_api.camera.registry import CAMERA_REGISTRY


def get_camera(camera_ip: str) -> BaseCamera:
    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Unknown camera '{camera_ip}'")
    return cam
