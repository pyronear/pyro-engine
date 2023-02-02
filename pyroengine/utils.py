import cv2
import numpy as np

__all__ = ["letterbox"]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, stride=32):
    """Letterbox image transform for yolo models
    Args:
        im (np.array): Input image
        new_shape (tuple, optional): Image size. Defaults to (640, 640).
        color (tuple, optional): Pixel fill value for the area outside the transformed image.
        Defaults to (114, 114, 114).
        auto (bool, optional): auto padding. Defaults to True.
        stride (int, optional): padding stride. Defaults to 32.
    Returns:
        np.array: Output image
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

    return im_b.astype("uint8")
