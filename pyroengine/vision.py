# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import logging
import os
import platform
import shutil
from typing import Tuple
from urllib.request import urlretrieve

import ncnn
import numpy as np
import onnxruntime
from huggingface_hub import HfApi  # type: ignore[import-untyped]
from PIL import Image

from .utils import DownloadProgressBar, box_iou, letterbox, nms, xywh2xyxy

__all__ = ["Classifier"]

MODEL_URL_FOLDER = "https://huggingface.co/pyronear/yolov11s/resolve/main/"
MODEL_ID = "pyronear/yolov11s"
MODEL_NAME = "yolov11s.pt"
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

    def __init__(
        self,
        model_folder="data",
        imgsz=1024,
        conf=0.15,
        iou=0,
        format="ncnn",
        model_path=None,
        max_bbox_size=0.4,
    ) -> None:
        if model_path:
            # Checks that the file exists
            if not os.path.isfile(model_path):
                raise ValueError(f"Model file not found: {model_path}")
            # Checks that file format is .onnx
            if os.path.splitext(model_path)[-1].lower() != ".onnx":
                raise ValueError(f"Input model_path should point to an ONNX export but currently is {model_path}")
            self.format = "onnx"
        else:
            if format == "ncnn":
                if not self.is_arm_architecture():
                    logging.info("NCNN format is optimized for arm architecture only, switching to onnx is recommended")

                model = "yolov11s_ncnn_model.zip"
                self.format = "ncnn"

            elif format == "onnx":
                model = "yolov11s.onnx"
                self.format = "onnx"

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
                    extracted_path = os.path.splitext(model_path)[0]
                    if os.path.isdir(extracted_path):
                        shutil.rmtree(extracted_path)
                    self.download_model(model_url, model_path, expected_sha256, metadata_path)
            else:
                self.download_model(model_url, model_path, expected_sha256, metadata_path)

            file_name, ext = os.path.splitext(model_path)
            if ext == ".zip":
                if not os.path.isdir(file_name):
                    shutil.unpack_archive(model_path, model_folder)
                model_path = file_name

        if self.format == "ncnn":
            self.model = ncnn.Net()
            self.model.load_param(os.path.join(model_path, "model.ncnn.param"))
            self.model.load_model(os.path.join(model_path, "model.ncnn.bin"))

        else:
            try:
                self.ort_session = onnxruntime.InferenceSession(model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load the ONNX model from {model_path}: {e!s}") from e

            logging.info(f"ONNX model loaded successfully from {model_path}")

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_bbox_size = max_bbox_size

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

    def prep_process(self, pil_img: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess an image for inference

        Args:
            pil_img: A valid PIL image.

        Returns:
            A tuple containing:
            - The resized and normalized image of shape (1, C, H, W).
            - Padding information as a tuple of integers (pad_height, pad_width).
        """
        np_img, pad = letterbox(np.array(pil_img), self.imgsz)  # Applies letterbox resize with padding

        if self.format == "ncnn":
            np_img = ncnn.Mat.from_pixels(np_img, ncnn.Mat.PixelType.PIXEL_BGR, np_img.shape[1], np_img.shape[0])
            mean = [0, 0, 0]
            std = [1 / 255, 1 / 255, 1 / 255]
            np_img.substract_mean_normalize(mean=mean, norm=std)
        else:
            np_img = np.expand_dims(np_img.astype("float32"), axis=0)  # Add batch dimension
            np_img = np.ascontiguousarray(np_img.transpose((0, 3, 1, 2)))  # Convert from BHWC to BCHW format
            np_img /= 255.0  # Normalize to [0, 1]

        return np_img, pad

    def post_process(self, pred: np.ndarray, pad: Tuple[int, int]) -> np.ndarray:
        """Post-process model predictions.

        Args:
            pred: Raw predictions from the model.
            pad: Padding information as (left_pad, top_pad).

        Returns:
            Processed predictions as a numpy array.
        """
        pred = pred[:, pred[-1, :] > self.conf]  # Drop low-confidence predictions
        pred = np.transpose(pred)
        pred = xywh2xyxy(pred)
        pred = pred[pred[:, 4].argsort()]  # Sort by confidence
        pred = nms(pred)
        pred = pred[::-1]  # Reverse for highest confidence first

        if len(pred) > 0:
            left_pad, top_pad = pad  # Unpack the tuple
            pred[:, :4:2] -= left_pad
            pred[:, 1:4:2] -= top_pad
            pred[:, :4:2] /= self.imgsz - 2 * left_pad
            pred[:, 1:4:2] /= self.imgsz - 2 * top_pad
            pred = np.clip(pred, 0, 1)
        else:
            pred = np.zeros((0, 5))  # Return empty prediction array

        return pred

    def __call__(self, pil_img: Image.Image, occlusion_bboxes: dict = {}) -> np.ndarray:
        """Run the classifier on an input image.

        Args:
            pil_img: The input PIL image.
            occlusion_mask: Optional occlusion mask to exclude certain areas.

        Returns:
            Processed predictions.
        """
        np_img, pad = self.prep_process(pil_img)

        if self.format == "ncnn":
            extractor = self.model.create_extractor()
            extractor.set_light_mode(True)
            extractor.input("in0", np_img)
            pred = ncnn.Mat()
            extractor.extract("out0", pred)
            pred = np.asarray(pred)
        else:
            pred = self.ort_session.run(["output0"], {"images": np_img})[0][0]

        # Convert pad to a tuple if required
        if isinstance(pad, list):
            pad = tuple(pad)

        pred = self.post_process(pred, pad)  # Ensure pad is passed as a tuple

        # drop big detections
        pred = np.clip(pred, 0, 1)
        pred = pred[(pred[:, 2] - pred[:, 0]) < self.max_bbox_size, :]
        pred = np.reshape(pred, (-1, 5))

        print(pred, occlusion_bboxes)

        # Remove prediction in bbox occlusion mask
        if len(occlusion_bboxes):
            all_boxes = np.array([b[:4] for b in occlusion_bboxes.values()], dtype=pred.dtype)

            pred_boxes = pred[:, :4].astype(pred.dtype)
            ious = box_iou(pred_boxes, all_boxes)
            max_ious = ious.max(axis=0)
            keep = max_ious <= 0.3
            pred = pred[keep]

        return pred
