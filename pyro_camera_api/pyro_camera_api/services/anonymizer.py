# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


def scale_and_clip_boxes(
    boxes_px: List[Box],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> List[Box]:
    """
    Rescale boxes from source size to destination size and clip to bounds.

    Boxes are given as (x1, y1, x2, y2) in pixel coordinates.
    """
    if src_w == dst_w and src_h == dst_h:
        return boxes_px

    sx = dst_w / max(1, src_w)
    sy = dst_h / max(1, src_h)
    out: List[Box] = []

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


def paint_boxes_black(img: Image.Image, boxes_px: List[Box]) -> Image.Image:
    """
    Fill given boxes with black pixels on a copy of the image.
    """
    if not boxes_px:
        return img

    arr = np.array(img)

    for x1, y1, x2, y2 in boxes_px:
        arr[y1:y2, x1:x2, :] = 0

    return Image.fromarray(arr)
