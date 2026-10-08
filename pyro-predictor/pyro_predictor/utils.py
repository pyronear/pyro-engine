# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import cv2
import numpy as np
from tqdm import tqdm

__all__ = ["DownloadProgressBar", "letterbox", "nms", "xywh2xyxy"]


def xywh2xyxy(x: np.ndarray):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def letterbox(
    im: np.ndarray,
    new_shape: tuple = (1024, 1024),
    color: tuple = (114, 114, 114),
    auto: bool = False,
    stride: int = 32,
):
    """Letterbox image transform for yolo models
    Args:
        im (np.ndarray): Input image
        new_shape (tuple, optional): Image size. Defaults to (1024, 1024).
        color (tuple, optional): Pixel fill value for the area outside the transformed image.
        Defaults to (114, 114, 114).
        auto (bool, optional): auto padding. Defaults to False.
        stride (int, optional): padding stride. Defaults to 32.
    Returns:
        np.ndarray: Output image
    """
    # Resize and pad image while meeting stride-multiple constraints
    im = np.asarray(im)
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
    # Padding is ultimately uint8. Allocate that buffer directly instead of two
    # full float64 buffers followed by another uint8 copy.
    im_b = np.empty((h + top + bottom, w + left + right, 3), dtype=np.uint8)
    im_b[:] = np.asarray(color).astype(np.uint8)
    im_b[top : top + h, left : left + w, :] = im
    return im_b, (left, top)


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
        (np.ndarray): An MxN numpy array containing the pairwise IoU values for every element in box1 and box2.
    """
    (a1, a2), (b1, b2) = np.split(box1, 2, 1), np.split(box2, 2, 1)
    area1 = (a2 - a1).prod(1)
    area2 = (b2 - b1).prod(1)
    # Work on two coordinate planes instead of a (M, N, 2) temporary. Keep
    # np.prod's integer promotion as well as the original floating-point order.
    inter_w = np.minimum(box1[:, 2], box2[:, None, 2]) - np.maximum(box1[:, 0], box2[:, None, 0])
    inter_h = np.minimum(box1[:, 3], box2[:, None, 3]) - np.maximum(box1[:, 1], box2[:, None, 1])
    inter_w.clip(0, out=inter_w)
    inter_h.clip(0, out=inter_h)
    inter = np.multiply(inter_w, inter_h, dtype=np.result_type(area1.dtype, area2.dtype))

    # IoU = inter / (area1 + area2 - inter)
    return inter / (area1 + area2[:, None] - inter + eps)


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

    # Preserve the original ascending-confidence suppression: a box is removed
    # if it overlaps ANY later box, even one that is itself removed later. This
    # differs from greedy NMS on overlap chains. Earlier survivors cannot overlap
    # a later box, so only the upper triangle is needed to make the same choices.
    # Compute short row blocks to bound temporary memory instead of allocating
    # the entire quadratic IoU matrix.
    keep = np.empty(len(boxes), dtype=bool)
    columns = np.arange(len(boxes))
    block_size = 128
    for start in range(0, len(boxes), block_size):
        end = min(start + block_size, len(boxes))
        overlaps = box_iou(boxes[start:, :4], boxes[start:end, :4]) > overlapThresh
        overlaps &= columns[None, start:] > np.arange(start, end)[:, None]
        keep[start:end] = ~overlaps.any(axis=1)

    return boxes[keep]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
