# Copyright (C) 2022-2024, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import numpy as np
from tqdm import tqdm  # type: ignore[import-untyped]

__all__ = ["nms", "xywh2xyxy", "DownloadProgressBar", "letterbox"]


def xywh2xyxy(x: np.ndarray):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def box_iou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (np.ndarray): A numpy array of shape (N, 4) representing N bounding boxes.
        box2 (np.ndarray): A numpy array of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.ndarray): An NxM numpy array containing the pairwise IoU values for every element in box1 and box2.
    """

    (a1, a2), (b1, b2) = np.split(box1, 2, 1), np.split(box2, 2, 1)
    inter = (np.minimum(a2, b2[:, None, :]) - np.maximum(a1, b1[:, None, :])).clip(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(1) + (b2 - b1).prod(1)[:, None] - inter + eps)


def nms(boxes: np.ndarray, overlapThresh: int = 0):
    """Non maximum suppression

    Args:
        boxes (np.ndarray): A numpy array of shape (N, 4) representing N bounding boxes in (x1, y1, x2, y2, conf) format
        overlapThresh (int, optional): iou threshold. Defaults to 0.

    Returns:
        boxes: Boxes after NMS
    """
    # Return an empty list, if no boxes given
    boxes = boxes[boxes[:, -1].argsort()]
    if len(boxes) == 0:
        return []

    indices = np.arange(len(boxes))
    rr = box_iou(boxes[:, :4], boxes[:, :4])
    for i, box in enumerate(boxes):
        temp_indices = indices[indices != i]
        if np.any(rr[i, temp_indices] > overlapThresh):
            indices = indices[indices != i]

    return boxes[indices]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
