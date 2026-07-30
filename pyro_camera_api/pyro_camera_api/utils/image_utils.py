# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

__all__ = ["measure_sharpness"]


def measure_sharpness(image: Image.Image) -> float:
    """Variance of the Laplacian over the lower half of the grayscale image.

    The upper half is excluded: watchtower scenes are mostly sky there, which
    has no texture and only dilutes the sharpness signal with sensor noise.
    """
    arr = np.array(image.convert("L"))
    arr = arr[arr.shape[0] // 2 :, :]
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())
