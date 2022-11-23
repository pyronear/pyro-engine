# Copyright (C) 2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import json
from typing import Any, Optional

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from PIL import Image
from pathlib import Path
import requests

__all__ = ["Classifier"]


def dl_file(url, dst):
    print(f"Downloading {url} ...")
    response = requests.get(url)
    open(dst, "wb").write(response.content)




class Classifier:
    """Implements an image classification model using ONNX backend.

    Examples:
        >>> from pyroengine.vision import Classifier
        >>> model = Classifier("pyronear/rexnet1_3x")

    Args:
        model_list: list of model to use
    """

    def __init__(
        self,
        model_list: str,
        cache_folder: str 
    ) -> None:

        self.cfg = []
        self.ort_session = []

        # Check if model is available in cache
        cache = Path(cache_folder)

        model_files = []
        for model in model_list:
            folder = cache.joinpath(f"models/{model}")
            folder.mkdir(parents=True, exist_ok=True)
            model_file = folder.joinpath("model.onnx")
            cfg_file = folder.joinpath("config.json")

            model_files.append((str(model_file), str(cfg_file)))
            
            if not model_file.is_file():
                url_model = f"https://huggingface.co/pyronear/{model}/resolve/main/model.onnx"
                dl_file(url_model, model_file)
                
            if not cfg_file.is_file():
                url_cfg = f"https://huggingface.co/pyronear/{model}/resolve/main/config.json"
                dl_file(url_cfg, cfg_file)
                
        for model_file, cfg_file in model_files:

            with open(cfg_file, "rb") as f:
                self.cfg.append(json.load(f))

            self.ort_session.append(onnxruntime.InferenceSession(model_file))


    def preprocess_image(self, idx: int, pil_img: Image.Image) -> np.ndarray:
        """Preprocess an image for inference

        Args:
            model_idx: model index
            pil_img: a valid pillow image

        Returns:
            the resized and normalized image of shape (1, C, H, W)
        """

        # Resizing
        img = pil_img.resize(self.cfg[idx]["input_shape"][-2:][::-1], Image.BILINEAR)
        # (H, W, C) --> (C, H, W)
        img = np.asarray(img).transpose((2, 0, 1)).astype(np.float32) / 255
        # Normalization
        img -= np.array(self.cfg[idx]["mean"])[:, None, None]
        img /= np.array(self.cfg[idx]["std"])[:, None, None]

        return img[None, ...]

    def __call__(self, pil_img: Image.Image) -> np.ndarray:

        scores = []
        for idx in range(len(self.ort_session)):
            np_img = self.preprocess_image(idx, pil_img)

            # ONNX inference
            ort_input = {self.ort_session[idx].get_inputs()[0].name: np_img}
            ort_out = self.ort_session[idx].run(None, ort_input)
            # Sigmoid
            scores.append(1 / (1 + np.exp(-ort_out[0][0])))

        return np.mean(scores)
