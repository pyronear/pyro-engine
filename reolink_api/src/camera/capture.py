# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
import time
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from PIL import Image

from camera.registry import CAMERA_REGISTRY
from camera.time_utils import update_command_time

router = APIRouter()


def _scale_and_clip_boxes(
    boxes_px: List[Tuple[int, int, int, int]],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> List[Tuple[int, int, int, int]]:
    if src_w == dst_w and src_h == dst_h:
        return boxes_px
    sx = dst_w / max(1, src_w)
    sy = dst_h / max(1, src_h)
    out: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes_px:
        nx1 = max(0, min(dst_w, int(round(x1 * sx))))
        ny1 = max(0, min(dst_h, int(round(y1 * sy))))
        nx2 = max(0, min(dst_w, int(round(x2 * sx))))
        ny2 = max(0, min(dst_h, int(round(y2 * sy))))
        if nx2 > nx1 and ny2 > ny1:
            nx1 = min(nx1, dst_w - 1)
            ny1 = min(ny1, dst_h - 1)
            nx2 = min(nx2, dst_w)
            ny2 = min(ny2, dst_h)
            out.append((nx1, ny1, nx2, ny2))
    return out


def _paint_boxes_black(img: Image.Image, boxes_px: List[Tuple[int, int, int, int]]) -> Image.Image:
    if not boxes_px:
        return img
    arr = np.array(img)
    for x1, y1, x2, y2 in boxes_px:
        arr[y1:y2, x1:x2, :] = 0
    return Image.fromarray(arr)


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
):
    update_command_time()

    if camera_ip not in CAMERA_REGISTRY:
        raise HTTPException(status_code=404, detail="Unknown camera")

    cam = CAMERA_REGISTRY[camera_ip]
    img: Optional[Image.Image] = cam.capture(pos_id=pos_id)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to capture image")

    if anonymize:
        boxes_px_to_apply: List[Tuple[int, int, int, int]] = []

        try:
            boxes_store = getattr(request.app.state, "boxes", None)
            frames_store = getattr(request.app.state, "frames", None)
            if boxes_store is None:
                logging.warning("boxes store not available on app.state")
            else:
                latest_boxes, ts_src = boxes_store.get()
                fresh = True
                if max_age_ms is not None:
                    fresh = (time.time() - ts_src) * 1000.0 <= max_age_ms
                if latest_boxes and fresh:
                    if frames_store is not None and frames_store.get() is not None:
                        pkt = frames_store.get()
                        src_h, src_w = pkt.array_bgr.shape[:2]  # type: ignore[attr-defined]
                        boxes_px_to_apply = _scale_and_clip_boxes(latest_boxes, src_w, src_h, img.width, img.height)
                    else:
                        boxes_px_to_apply = _scale_and_clip_boxes(
                            latest_boxes, img.width, img.height, img.width, img.height
                        )
        except Exception as e:
            logging.warning("Using latest boxes failed, %s", e)

        if boxes_px_to_apply:
            img = _paint_boxes_black(img, boxes_px_to_apply)
        elif strict:
            raise HTTPException(status_code=503, detail="No recent boxes available for anonymization")

    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return Response(buffer.getvalue(), media_type="image/jpeg")


@router.get("/latest_image")
def get_latest_image(camera_ip: str, pose: int):
    if camera_ip not in CAMERA_REGISTRY:
        raise HTTPException(status_code=404, detail="Unknown camera")

    cam = CAMERA_REGISTRY[camera_ip]
    if pose not in cam.last_images or cam.last_images[pose] is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    buffer = BytesIO()
    cam.last_images[pose].save(buffer, format="JPEG")
    return Response(buffer.getvalue(), media_type="image/jpeg")
