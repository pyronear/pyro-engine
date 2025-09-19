# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import logging
import threading
from io import BytesIO
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from anonymizer.vision import Anonymizer
from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import Response
from PIL import Image

from camera.registry import CAMERA_REGISTRY
from camera.time_utils import update_command_time

router = APIRouter()

_model: Optional[Anonymizer] = None
_model_lock = threading.Lock()


def _get_model() -> Anonymizer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logging.info("Loading Anonymizer model for capture endpoint")
                _model = Anonymizer()
                logging.info("Anonymizer model ready for capture endpoint")
    return _model


def _boxes_px_from_norm(
    boxes_norm: Iterable[Sequence[float]],
    W: int,
    H: int,
    conf_th: float,
) -> List[Tuple[int, int, int, int]]:
    out_px: List[Tuple[int, int, int, int]] = []
    for it in boxes_norm:
        if it is None or len(it) < 4:
            continue
        x1, y1, x2, y2 = map(float, it[:4])  # already normalized in [0, 1]
        conf = float(it[4]) if len(it) >= 5 else 1.0
        if conf < conf_th:
            continue
        x1p = max(0, min(W - 1, int(x1 * W)))
        y1p = max(0, min(H - 1, int(y1 * H)))
        x2p = max(0, min(W - 1, int(x2 * W)))
        y2p = max(0, min(H - 1, int(y2 * H)))
        if x2p > x1p and y2p > y1p:
            out_px.append((x1p, y1p, x2p, y2p))
    return out_px


def _paint_boxes_black(img: Image.Image, boxes_px: List[Tuple[int, int, int, int]]) -> Image.Image:
    if not boxes_px:
        return img
    arr = np.array(img)
    for x1, y1, x2, y2 in boxes_px:
        arr[y1:y2, x1:x2, :] = 0
    return Image.fromarray(arr)


@router.get("/capture")
def capture(
    camera_ip: str,
    pos_id: Optional[int] = Query(default=None),
    anonymize: bool = Query(default=True, description="Apply anonymization"),
    conf_thres: float = Query(default=0.30, ge=0.0, le=1.0, description="Confidence threshold"),
    strict: bool = Query(default=False, description="Fail if anonymization is unavailable"),
):
    update_command_time()

    if camera_ip not in CAMERA_REGISTRY:
        raise HTTPException(status_code=404, detail="Unknown camera")

    cam = CAMERA_REGISTRY[camera_ip]
    img: Optional[Image.Image] = cam.capture(pos_id=pos_id)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to capture image")

    if anonymize:
        try:
            model = _get_model()
            W, H = img.width, img.height

            preds = model(img)  # expected format: [[x1, y1, x2, y2, conf], ...] with coords in [0, 1]
            boxes_px = _boxes_px_from_norm(preds, W, H, conf_thres)
            img = _paint_boxes_black(img, boxes_px)

        except Exception as e:
            logging.exception("Capture anonymization failed: %s", e)
            if strict:
                raise HTTPException(status_code=503, detail="Anonymization unavailable")

    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return Response(buffer.getvalue(), media_type="image/jpeg")


@router.get("/latest_image")
def get_latest_image(camera_ip: str, pose: int):
    cam = CAMERA_REGISTRY[camera_ip]

    if pose not in cam.last_images or cam.last_images[pose] is None:
        # Explicitly signal "nothing yet"
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    buffer = BytesIO()
    cam.last_images[pose].save(buffer, format="JPEG")
    return Response(buffer.getvalue(), media_type="image/jpeg")
