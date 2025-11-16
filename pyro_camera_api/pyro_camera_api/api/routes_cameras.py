# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import logging
import time
from io import BytesIO
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from PIL import Image
from pyro_camera_api.api.deps import get_camera
from pyro_camera_api.camera.base import BaseCamera
from pyro_camera_api.core.config import RAW_CONFIG
from pyro_camera_api.services.anonymizer import paint_boxes_black, scale_and_clip_boxes
from pyro_camera_api.utils.time_utils import update_command_time

router = APIRouter()
logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Info endpoints
# ---------------------------------------------------------------------------


@router.get("/cameras")
def list_cameras():
    return {"camera_ids": list(RAW_CONFIG.keys())}


@router.get("/camera_infos")
def get_camera_infos():
    """Return list of cameras with their metadata."""
    camera_infos = []

    for cam_id, conf in RAW_CONFIG.items():
        camera_infos.append({
            "id": conf.get("id", cam_id),
            "camera_id": cam_id,
            "ip": conf.get("ip_address", cam_id),
            "backend": conf.get("backend", "unknown"),
            "type": conf.get("type", "Unknown"),
            "name": conf.get("name", cam_id),
            "azimuths": conf.get("azimuths", []),
            "poses": conf.get("poses", []),
        })

    return {"cameras": camera_infos}


# ---------------------------------------------------------------------------
# Capture endpoints
# ---------------------------------------------------------------------------


@router.get("/capture")
def capture(
    request: Request,
    camera_ip: str,
    cam: BaseCamera = Depends(get_camera),
    pos_id: Optional[int] = Query(default=None),
    anonymize: bool = Query(default=True, description="Apply anonymization using latest boxes"),
    max_age_ms: Optional[int] = Query(
        default=None,
        description="Only use boxes if not older than this many milliseconds",
    ),
    strict: bool = Query(
        default=False,
        description="If true and no recent boxes are available, return 503",
    ),
    width: Optional[int] = Query(
        default=None,
        description="Resize output image to this width while preserving aspect ratio. If not provided, no resize is applied.",
    ),
):
    update_command_time()

    # at this point cam is guaranteed to exist, or 404 would already be raised by get_camera
    img: Optional[Image.Image] = cam.capture(pos_id=pos_id)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to capture image")

    if anonymize:
        boxes_px_to_apply: List[Box] = []

        try:
            boxes_store = getattr(request.app.state, "boxes", None)
            frames_store = getattr(request.app.state, "frames", None)

            if boxes_store is None:
                logger.warning("boxes store not available on app.state")
            else:
                latest_boxes, ts_src = boxes_store.get()
                fresh = True
                if max_age_ms is not None:
                    fresh = (time.time() - ts_src) * 1000.0 <= max_age_ms

                if latest_boxes and fresh:
                    if frames_store is not None and frames_store.get() is not None:
                        pkt = frames_store.get()
                        src_h, src_w = pkt.array_bgr.shape[:2]
                        boxes_px_to_apply = scale_and_clip_boxes(latest_boxes, src_w, src_h, img.width, img.height)
                    else:
                        boxes_px_to_apply = scale_and_clip_boxes(
                            latest_boxes, img.width, img.height, img.width, img.height
                        )

        except Exception as exc:
            logger.warning("Using latest boxes failed: %s", exc)

        if boxes_px_to_apply:
            img = paint_boxes_black(img, boxes_px_to_apply)
        elif strict:
            raise HTTPException(status_code=503, detail="No recent boxes available for anonymization")

    if width is not None:
        try:
            aspect_ratio = img.height / img.width
            new_height = int(width * aspect_ratio)
            img = img.resize((width, new_height), Image.LANCZOS)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to resize image: {exc}")

    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return Response(buffer.getvalue(), media_type="image/jpeg")


@router.get("/latest_image")
def get_latest_image(
    camera_ip: str,
    pose: int,
    cam: BaseCamera = Depends(get_camera),
):
    # cam is guaranteed to exist thanks to get_camera
    if pose not in cam.last_images or cam.last_images[pose] is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    buffer = BytesIO()
    cam.last_images[pose].save(buffer, format="JPEG")
    return Response(buffer.getvalue(), media_type="image/jpeg")
