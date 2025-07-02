# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from io import BytesIO
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from camera.registry import CAMERA_REGISTRY
from camera.time_utils import update_command_time

router = APIRouter()


@router.get("/capture")
def capture(camera_ip: str, pos_id: Optional[int] = Query(default=None)):
    update_command_time()
    cam = CAMERA_REGISTRY[camera_ip]
    img = cam.capture(pos_id=pos_id)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to capture image")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return Response(buffer.getvalue(), media_type="image/jpeg")


@router.get("/latest_image")
def get_latest_image(camera_ip: str, pose: int):
    cam = CAMERA_REGISTRY[camera_ip]

    if pose not in cam.last_images or cam.last_images[pose] is None:
        raise HTTPException(status_code=404, detail="No image available for this pose")

    buffer = BytesIO()
    cam.last_images[pose].save(buffer, format="JPEG")
    return Response(buffer.getvalue(), media_type="image/jpeg")
