# Copyright (C) 2023-2024, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import logging
import os
import platform
import shutil
from typing import Optional
from urllib.request import urlretrieve

import numpy as np
from huggingface_hub import HfApi  # type: ignore[import-untyped]
from PIL import Image
from ultralytics import YOLO  # type: ignore[import-untyped]

from .utils import DownloadProgressBar

__all__ = ["Classifier"]

MODEL_URL_FOLDER = "https://huggingface.co/pyronear/yolov8s/resolve/main/"
MODEL_ID = "pyronear/yolov8s"
MODEL_NAME = "yolov8s.pt"
METADATA_NAME = "model_metadata.json"


logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


# Utility function to save metadata
def save_metadata(metadata_path, metadata):
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


class Classifier:
    """Implements an image classification model using YOLO backend.

    Examples:
        >>> from pyroengine.vision import Classifier
        >>> model = Classifier()

    Args:
        model_path: model path
    """

    def __init__(self, model_folder="data", imgsz=1024, conf=0.15, iou=0.05, format="ncnn", model_path=None) -> None:
        if model_path is None:
            if format == "ncnn":
                if self.is_arm_architecture():
                    model = "yolov8s_ncnn_model.zip"
                else:
                    logging.info("NCNN format is optimized for arm architecture only, switching to onnx")
                    model = "yolov8s.onnx"
            elif format in ["onnx", "pt"]:
                model = f"yolov8s.{format}"

            model_path = os.path.join(model_folder, model)
            metadata_path = os.path.join(model_folder, METADATA_NAME)
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
                metadata = self.load_metadata(metadata_path)
                if metadata and metadata.get("sha256") == expected_sha256:
                    logging.info("Model already exists and the SHA256 hash matches. No download needed.")
                else:
                    logging.info("Model exists but the SHA256 hash does not match or the file doesn't exist.")
                    os.remove(model_path)
                    self.download_model(model_url, model_path, expected_sha256, metadata_path)
            else:
                self.download_model(model_url, model_path, expected_sha256, metadata_path)

            file_name, ext = os.path.splitext(model_path)
            if ext == ".zip":
                if not os.path.isdir(file_name):
                    shutil.unpack_archive(model_path, model_folder)
                model_path = file_name

        self.model = YOLO(model_path, task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def is_arm_architecture(self):
        # Check for ARM architecture
        return platform.machine().startswith("arm") or platform.machine().startswith("aarch")

    def get_sha(self, siblings):
        # Extract the SHA256 hash from the model files metadata
        for file in siblings:
            if file.rfilename == os.path.basename(MODEL_NAME):
                return file.lfs["sha256"]
        return None

    def download_model(self, model_url, model_path, expected_sha256, metadata_path):
        # Ensure the directory exists
        os.makedirs(os.path.split(model_path)[0], exist_ok=True)

        # Download the model
        logging.info(f"Downloading model from {model_url} ...")
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=model_path) as t:
            urlretrieve(model_url, model_path, reporthook=t.update_to)
        logging.info("Model downloaded!")

        # Save the metadata
        metadata = {"sha256": expected_sha256}
        save_metadata(metadata_path, metadata)
        logging.info("Metadata saved!")

    # Utility function to load metadata
    def load_metadata(self, metadata_path):
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

    def __call__(self, pil_img: Image.Image, occlusion_mask: Optional[np.ndarray] = None) -> np.ndarray:

        results = self.model(pil_img, imgsz=self.imgsz, conf=self.conf, iou=self.iou)
        y = np.concatenate(
            (results[0].boxes.xyxyn.cpu().numpy(), results[0].boxes.conf.cpu().numpy().reshape((-1, 1))), axis=1
        )

        y = np.reshape(y, (-1, 5))

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
