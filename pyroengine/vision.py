# Copyright (C) 2022-2023, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
import urllib
from typing import Optional

import numpy as np
import onnxruntime
from PIL import Image

from .utils import letterbox, nms, xywh2xyxy

__all__ = ["Classifier"]

MODEL_URL = "https://github.com/pyronear/pyro-vision/releases/download/v0.2.0/yolov8s_v001.onnx"


class Classifier:
    """Implements an image classification model using ONNX backend.

    Examples:
        >>> from pyroengine.vision import Classifier
        >>> model = Classifier()

    Args:
        model_path: model path
    """

    def __init__(self, model_path: Optional[str] = "data/model.onnx", img_size: tuple = (384, 640)) -> None:
        # Download model if not available
        if not os.path.isfile(model_path):
            os.makedirs(os.path.split(model_path)[0], exist_ok=True)
            print(f"Downloading model from {MODEL_URL} ...")
            urllib.request.urlretrieve(MODEL_URL, model_path)

        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.img_size = img_size

    def preprocess_image(self, pil_img: Image.Image, mask: np.array = None) -> np.ndarray:
        """Preprocess an image for inference

        Args:
            pil_img: a valid pillow image
            mask: occlusion mask to drop prediction in an area

        Returns:
            the resized and normalized image of shape (1, C, H, W)
        """

        np_img = letterbox(np.array(pil_img), self.img_size)  # letterbox
        np_img = np.expand_dims(np_img.astype("float"), axis=0)
        np_img = np.ascontiguousarray(np_img.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        np_img = np_img.astype("float32") / 255

        return np_img

    def __call__(self, pil_img: Image.Image) -> np.ndarray:
        np_img = self.preprocess_image(pil_img)

        # ONNX inference
        y = self.ort_session.run(["output0"], {"images": np_img})[0][0]
        # Drop low conf for speed-up
        y = y[:, y[-1, :] > 0.05]
        # Post processing
        y = np.transpose(y)
        y = xywh2xyxy(y)
        # Sort by confidence
        y = y[y[:, 4].argsort()]
        y = nms(y)
        # Normalize preds
        if len(y) > 0:
            y[:, :4:2] /= self.img_size[1]
            y[:, 1:4:2] /= self.img_size[0]

        # Remove prediction in occlusion mask
        if mask:
            hm, wm = mask.shape
            keep = []
            for p in y.copy():
                p[:4:2] *= wm
                p[1:4:2] *= hm
                x0, y0, x1, y1 = p.astype("int")[:4]
                if np.sum(mask[y0:y1, x0:x1]) > 0:
                    keep.append(True)
                else:
                    keep.append(False)

            y = y[keep]

        return y
