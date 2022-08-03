# Copyright (C) 2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import json

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from PIL import Image

__all__ = ["Classifier"]


class Classifier:
    def __init__(self, hub_repo: str) -> None:
        # Download model config & checkpoint
        with open(hf_hub_download(hub_repo, filename="config.json"), "rb") as f:
            self.cfg = json.load(f)

        self.ort_session = onnxruntime.InferenceSession(hf_hub_download(hub_repo, filename="model.onnx"))

    def preprocess_image(self, pil_img: Image.Image) -> np.ndarray:
        """Preprocess an image for inference

        Args:
            pil_img: a valid pillow image

        Returns:
            the resized and normalized image of shape (1, C, H, W)
        """

        # Resizing
        img = pil_img.resize(self.cfg["input_shape"][-2:], Image.BILINEAR)
        # (H, W, C) --> (C, H, W)
        img = np.asarray(img).transpose((2, 0, 1)).astype(np.float32) / 255
        # Normalization
        img -= np.array(self.cfg["mean"])[:, None, None]
        img /= np.array(self.cfg["std"])[:, None, None]

        return img[None, ...]

    def __call__(self, pil_img: Image.Image) -> np.ndarray:
        np_img = self.preprocess_image(pil_img)

        # ONNX inference
        ort_input = {self.ort_session.get_inputs()[0].name: np_img}
        ort_out = self.ort_session.run(None, ort_input)
        # Sigmoid
        return 1 / (1 + np.exp(-ort_out[0][0]))
