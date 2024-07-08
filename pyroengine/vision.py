# Copyright (C) 2023-2024, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import os
from typing import Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import onnxruntime
from huggingface_hub import HfApi  # type: ignore[import-untyped]
from PIL import Image
import platform
import logging

from .utils import DownloadProgressBar, letterbox, nms, xywh2xyxy

__all__ = ["Classifier"]

MODEL_URL_FOLDER = "https://huggingface.co/pyronear/yolov8s/resolve/main/"
MODEL_ID = "pyronear/yolov8s"
MODEL_NAME = "yolov8s.pt"
METADATA_PATH = "data/model_metadata.json"


logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


def is_arm_architecture():
    # Check for ARM architecture
    return platform.machine().startswith('arm') or platform.machine().startswith('aarch')

# Utility function to save metadata
def save_metadata(metadata_path, metadata):
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

 


class Classifier:
    """Implements an image classification model using ONNX backend.

    Examples:
        >>> from pyroengine.vision import Classifier
        >>> model = Classifier()

    Args:
        model_path: model path
    """

    def __init__(self, model_folder="data", imgsz=1024, conf=0.15, iou=0, format="ncnn", model_path=None) -> None:

        if model_path is None:
            if format=="ncnn":
                if is_arm_architecture():
                    model = "yolov8s_ncnn_model.zip"
                else:
                    logging.info("NCNN format is optimized for arm architecture only, switching to onnx")
                    model = "yolov8s.onnx"
            elif format in ["onnx","pt"]:
                model = f"yolov8s.{format}"
            
            model_path = os.path.join(model_folder, model)
            model_url = MODEL_URL_FOLDER + model
        
            # Get the expected SHA256 from Hugging Face
            api = HfApi()
            model_info = api.model_info(MODEL_ID, files_metadata=True)
            expected_sha256 = self.get_sha(model_info.siblings)

            if not expected_sha256:
                raise ValueError("SHA256 hash for the model file not found in the Hugging Face model metadata.")


            # Check if the model file exists
            if os.path.isfile(model_path):
                # Load existing metadata
                metadata = self.load_metadata(METADATA_PATH)
                if metadata and metadata.get("sha256") == expected_sha256:
                    print("Model already exists and the SHA256 hash matches. No download needed.")
                else:
                    print("Model exists but the SHA256 hash does not match or the file doesn't exist.")
                    os.remove(model_path)
                    self.download_model(model_url, model_path, expected_sha256)
            else:
                self.download_model(model_url, model_path, expected_sha256)

        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.img_size = img_size

    def get_sha(self, siblings):
        # Extract the SHA256 hash from the model files metadata
        for file in siblings:
            if file.rfilename == os.path.basename(MODEL_NAME):
                expected_sha256 = file.lfs.sha256
                break
        return expected_sha256

    def download_model(self, model_path, expected_sha256):
        # Ensure the directory exists
        os.makedirs(os.path.split(model_path)[0], exist_ok=True)

        # Download the model
        print(f"Downloading model from {MODEL_URL} ...")
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=model_path) as t:
            urlretrieve(MODEL_URL, model_path, reporthook=t.update_to)
        print("Model downloaded!")

        # Save the metadata
        metadata = {"sha256": expected_sha256}
        save_metadata(METADATA_PATH, metadata)
        print("Metadata saved!")

    # Utility function to load metadata
    def load_metadata(self, metadata_path):
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

    def preprocess_image(self, pil_img: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess an image for inference

        Args:
            pil_img: A valid PIL image.

        Returns:
            A tuple containing:
            - The resized and normalized image of shape (1, C, H, W).
            - Padding information as a tuple of integers (pad_height, pad_width).
        """

        np_img, pad = letterbox(np.array(pil_img), self.img_size)  # Applies letterbox resize with padding
        np_img = np.expand_dims(np_img.astype("float"), axis=0)  # Add batch dimension
        np_img = np.ascontiguousarray(np_img.transpose((0, 3, 1, 2)))  # Convert from BHWC to BCHW format
        np_img = np_img.astype("float32") / 255  # Normalize to [0, 1]

        return np_img, pad

    def __call__(self, pil_img: Image.Image, occlusion_mask: Optional[np.ndarray] = None) -> np.ndarray:
        np_img, pad = self.preprocess_image(pil_img)

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
        y = y[::-1]

        # Normalize preds
        if len(y) > 0:
            # Remove padding
            left_pad, top_pad = pad
            y[:, :4:2] -= left_pad
            y[:, 1:4:2] -= top_pad
            y[:, :4:2] /= self.img_size[1] - 2 * left_pad
            y[:, 1:4:2] /= self.img_size[0] - 2 * top_pad
            y = np.clip(y, 0, 1)
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

        return y
