# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
import time
from io import BytesIO
from typing import List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from PIL import Image
from PIL.Image import Resampling

from pyro_camera_api.camera.registry import CAMERA_REGISTRY
from pyro_camera_api.core.config import RAW_CONFIG
from pyro_camera_api.services.anonymizer import paint_boxes_black, scale_and_clip_boxes
from pyro_camera_api.utils.time_utils import update_command_time

LANCZOS = Resampling.LANCZOS

router = APIRouter()
logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


@router.get("/cameras_list")
def list_cameras():
    """
    Return the list of configured camera identifiers.

    The identifiers come from the loaded RAW_CONFIG mapping and correspond
    to the keys used in CAMERA_REGISTRY and in the API endpoints.
    """
    return {"camera_ids": list(RAW_CONFIG.keys())}


@router.get("/camera_infos")
def get_camera_infos():
    """
    Return metadata for all configured cameras.

    Each entry includes the camera identifier, IP address, adapter name,
    camera type, human readable name, azimuths, and preset poses, based on
    the RAW_CONFIG content.
    """
    camera_infos = []
    for cam_id, conf in RAW_CONFIG.items():
        camera_infos.append({
            "id": conf.get("id", cam_id),
            "camera_id": cam_id,
            "ip": conf.get("ip_address", cam_id),
            "adapter": conf.get("adapter", "unknown"),
            "type": conf.get("type", "Unknown"),
            "name": conf.get("name", cam_id),
            "azimuths": conf.get("azimuths", []),
            "poses": conf.get("poses", []),
        })
    return {"cameras": camera_infos}


def _capture_impl(
    request: Request,
    camera_ip: str,
    pos_id: Optional[int],
    anonymize: bool,
    max_age_ms: Optional[int],
    strict: bool,
    width: Optional[int],
    quality: int,
) -> Response:
    """
    Internal helper that captures a frame and applies optional anonymization.

    The function retrieves the target camera from CAMERA_REGISTRY, captures
    a snapshot, optionally applies anonymization using the latest detected
    boxes stored on app.state, optionally resizes the image by width while
    preserving aspect ratio, and returns the result as a JPEG response.

    Parameters
    ----------
    request:
        The incoming FastAPI request, used to access shared state such
        as frames and detection boxes.
    camera_ip:
        Identifier of the camera, usually the IP address used as key
        in CAMERA_REGISTRY and RAW_CONFIG.
    pos_id:
        Optional preset pose to use before capturing, if supported by
        the camera implementation.
    anonymize:
        If true, black rectangles are drawn over regions defined by
        the latest detection boxes.
    max_age_ms:
        Maximum age in milliseconds allowed for the detection boxes.
        If the boxes are older than this value they are ignored.
    strict:
        If true and no recent boxes are available while anonymize is
        requested, the call fails with HTTP 503.
    width:
        Optional target width for the output image in pixels. Height
        is computed from the original aspect ratio.
    quality:
        JPEG quality parameter from 1 to 100. Higher means better quality
        and larger file size.

    Returns
    -------
    Response
        A FastAPI Response containing the JPEG encoded image.
    """
    update_command_time()

    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail="Unknown camera")

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
                        boxes_px_to_apply = scale_and_clip_boxes(
                            latest_boxes,
                            src_w,
                            src_h,
                            img.width,
                            img.height,
                        )
                    else:
                        boxes_px_to_apply = scale_and_clip_boxes(
                            latest_boxes,
                            img.width,
                            img.height,
                            img.width,
                            img.height,
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
            img = img.resize((width, new_height), LANCZOS)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to resize image: {exc}")

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return Response(buffer.getvalue(), media_type="image/jpeg")


@router.get("/capture")
def capture(
    request: Request,
    camera_ip: str,
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
    quality: int = Query(
        default=95,
        ge=1,
        le=100,
        description="JPEG quality that controls compression level. Higher means better quality and larger file size.",
    ),
):
    """
    Capture a snapshot from the requested camera.

    This endpoint captures a single image from the specified camera, optionally
    moves to a preset position, applies anonymization based on the latest
    detection boxes, and optionally resizes the image by width.

    Query parameters control anonymization, freshness of detection boxes,
    strict behavior when no boxes are available, target output width,
    and JPEG quality which controls compression.
    The response is always a JPEG image.
    """
    return _capture_impl(
        request=request,
        camera_ip=camera_ip,
        pos_id=pos_id,
        anonymize=anonymize,
        max_age_ms=max_age_ms,
        strict=strict,
        width=width,
        quality=quality,
    )


@router.get("/latest_image")
def get_latest_image(
    camera_ip: str,
    pose: int,
    quality: int = Query(
        default=95,
        ge=1,
        le=100,
        description="JPEG quality that controls compression level. Higher means better quality and larger file size.",
    ),
):
    """
    Return the last stored image for a given camera and pose.

    The camera adapter may cache captured frames per preset pose in its
    last_images mapping. This endpoint exposes that cache. If there is
    no image for the requested pose, the endpoint returns HTTP 204 with
    an empty body.
    """
    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam is None:
        raise HTTPException(status_code=404, detail="Unknown camera")

    if pose not in cam.last_images or cam.last_images[pose] is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    buffer = BytesIO()
    cam.last_images[pose].save(buffer, format="JPEG", quality=quality)
    return Response(buffer.getvalue(), media_type="image/jpeg")
