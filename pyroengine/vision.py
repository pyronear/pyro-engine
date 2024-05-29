# Copyright (C) 2022-2024, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
from typing import Optional, Tuple
from urllib.request import urlretrieve

import cv2  # type: ignore[import-untyped]
import numpy as np
import onnxruntime
from PIL import Image

from .utils import DownloadProgressBar, nms, xywh2xyxy

__all__ = ["Classifier"]

MODEL_URL = "https://huggingface.co/pyronear/yolov8s/resolve/main/model.onnx"


class Classifier:
    """Implements an image classification model using ONNX backend.

    Examples:
        >>> from pyroengine.vision import Classifier
        >>> model = Classifier()

    Args:
        model_path: model path
    """

    def __init__(self, model_path: Optional[str] = "data/model.onnx", base_img_size: int = 1024) -> None:
        if model_path is None:
            model_path = "data/model.onnx"

        if not os.path.isfile(model_path):
            os.makedirs(os.path.split(model_path)[0], exist_ok=True)
            print(f"Downloading model from {MODEL_URL} ...")
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=model_path) as t:
                urlretrieve(MODEL_URL, model_path, reporthook=t.update_to)
            print("Model downloaded!")

        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.base_img_size = base_img_size

    def preprocess_image(self, pil_img: Image.Image, new_img_size: list) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess an image for inference

        Args:
            pil_img: A valid PIL image.

        Returns:
            A tuple containing:
            - The resized and normalized image of shape (1, C, H, W).
            - Padding information as a tuple of integers (pad_height, pad_width).
        """

        np_img = cv2.resize(np.array(pil_img), new_img_size, interpolation=cv2.INTER_LINEAR)
        np_img = np.expand_dims(np_img.astype("float"), axis=0)  # Add batch dimension
        np_img = np.ascontiguousarray(np_img.transpose((0, 3, 1, 2)))  # Convert from BHWC to BCHW format
        np_img = np_img.astype("float32") / 255  # Normalize to [0, 1]

        return np_img

    def __call__(self, pil_img: Image.Image, occlusion_mask: Optional[np.ndarray] = None) -> np.ndarray:

        w, h = pil_img.size
        ratio = self.base_img_size / max(w, h)
        new_img_size = [int(ratio * w), int(ratio * h)]
        new_img_size = [x - x % 32 for x in new_img_size]  # size need to be a multiple of 32 to fit the model
        np_img = self.preprocess_image(pil_img, new_img_size)

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
            # Normalize Output
            y[:, :4:2] /= new_img_size[0]
            y[:, 1:4:2] /= new_img_size[1]
        else:
            y = np.zeros((0, 5))  # normalize output

        # Remove prediction in occlusion mask
        if occlusion_mask is not None:
            hm, wm = occlusion_mask.shape
            keep = []
            for p in y.copy():
                p[:4:2] *= wm
                p[1:4:2] *= hm
                p[:4:2] = np.clip(p[:4:2], 0, wm)
                p[:4:2] = np.clip(p[:4:2], 0, hm)
                x0, y0, x1, y1 = p.astype("int")[:4]
                if np.sum(occlusion_mask[y0:y1, x0:x1]) > 0:
                    keep.append(True)
                else:
                    keep.append(False)

            y = y[keep]

        return np.clip(y, 0, 1)
