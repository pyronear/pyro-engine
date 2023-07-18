# Copyright (C) 2022-2023, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
import urllib
from typing import Optional

import numpy as np
import onnxruntime
from PIL import Image

from .utils import NMS, letterbox, xywh2xyxy

__all__ = ["Classifier"]

MODEL_URL = "https://github.com/pyronear/pyro-vision/releases/download/v0.2.0/yolov5s_v002.onnx"


class Classifier:
    """Implements an image classification model using ONNX backend.

    Examples:
        >>> from pyroengine.vision import Classifier
        >>> model = Classifier()

    Args:
        model_path: model path
    """

    def __init__(self, model_path: Optional[str] = "data/model.onnx", img_size=(384, 640)) -> None:
        # Download model if not available
        if not os.path.isfile(model_path):
            os.makedirs(os.path.split(model_path)[0], exist_ok=True)
            print(f"Downloading model from {MODEL_URL} ...")
            urllib.request.urlretrieve(MODEL_URL, model_path)

        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.img_size = img_size

    def preprocess_image(self, pil_img: Image.Image) -> np.ndarray:
        """Preprocess an image for inference

        Args:
            pil_img: a valid pillow image
            img_size: image size

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
        y = y[:, y[-1, :] > 0.1]
        y = np.transpose(y)
        y = xywh2xyxy(y)
        y = y[y[:, 4].argsort()]
        y = NMS(y)
        if len(y) > 0:
            y[:, :4:2] /= self.img_size[1]
            y[:, 1:4:2] /= self.img_size[0]

        return y
