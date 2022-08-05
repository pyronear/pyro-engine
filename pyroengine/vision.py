# Copyright (C) 2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import json
from typing import Optional

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from PIL import Image

__all__ = ["Classifier"]


class Classifier:
    """Implements an image classification model using ONNX backend.

    Examples:
        >>> from pyroengine.vision import Classifier
        >>> model = Classifier("pyronear/rexnet1_3x")

    Args:
        hub_repo: repository from HuggingFace Hub to load the model from
        model_path: overrides the model path
        cfg_path: overrides the configuration file from the model
    """

    def __init__(self, hub_repo: str, model_path: Optional[str] = None, cfg_path: Optional[str] = None) -> None:
        # Download model config & checkpoint
        _path = cfg_path or hf_hub_download(hub_repo, filename="config.json")
        with open(_path, "rb") as f:
            self.cfg = json.load(f)

        _path = model_path or hf_hub_download(hub_repo, filename="model.onnx")
        self.ort_session = onnxruntime.InferenceSession(_path)

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
