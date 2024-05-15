# Copyright (C) 2022-2024, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import cv2  # type: ignore[import-untyped]
import numpy as np

__all__ = ["letterbox", "nms", "xywh2xyxy"]


def xywh2xyxy(x: np.ndarray):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def letterbox(
    im: np.ndarray, new_shape: tuple = (640, 640), color: tuple = (0, 0, 0), auto: bool = False, stride: int = 32
):
    """Letterbox image transform for yolo models
    Args:
        im (np.ndarray): Input image
        new_shape (tuple, optional): Image size. Defaults to (640, 640).
        color (tuple, optional): Pixel fill value for the area outside the transformed image.
        Defaults to (0, 0, 0).
        auto (bool, optional): auto padding. Defaults to True.
        stride (int, optional): padding stride. Defaults to 32.
    Returns:
        np.ndarray: Output image
    """
    # Resize and pad image while meeting stride-multiple constraints
    im = np.array(im)
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # add border
    h, w = im.shape[:2]
    im_b = np.zeros((h + top + bottom, w + left + right, 3)) + color
    im_b[top : top + h, left : left + w, :] = im

    return im_b.astype("uint8"), (left, top)


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
