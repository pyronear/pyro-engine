# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import os
import platform
import tarfile
from typing import Tuple
from urllib.request import urlretrieve

import ncnn
import numpy as np
import onnxruntime
from PIL import Image

from .utils import DownloadProgressBar, box_iou, letterbox, nms, xywh2xyxy

__all__ = ["Classifier"]

MODEL_URL_FOLDER = "https://huggingface.co/pyronear/yolo11s_mighty-mongoose_v5.1.0/resolve/main/"
MODEL_NAME = "ncnn_cpu_yolo11s_mighty-mongoose_v5.1.0.tar.gz"

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


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
            if not os.path.isfile(model_path):
                raise ValueError(f"Model file not found: {model_path}")
            if os.path.splitext(model_path)[-1].lower() != ".onnx":
                raise ValueError(f"Input model_path should point to an ONNX export but currently is {model_path}")
            self.format = "onnx"
        else:
            if format == "ncnn":
                if not self.is_arm_architecture():
                    logging.info("NCNN format is optimized for arm architecture only, switching to onnx is recommended")
                model = MODEL_NAME
                self.format = "ncnn"
            elif format == "onnx":
                model = MODEL_NAME.replace("ncnn", "onnx")
                self.format = "onnx"
            else:
                raise ValueError("Unsupported format: should be 'ncnn' or 'onnx'")

            model_path = os.path.join(model_folder, model)
            model_url = MODEL_URL_FOLDER + model

            if not os.path.isfile(model_path):
                logging.info(f"Downloading model from {model_url} ...")
                os.makedirs(model_folder, exist_ok=True)
                with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=model_path) as t:
                    urlretrieve(model_url, model_path, reporthook=t.update_to)
                logging.info("Model downloaded!")

            # Extract .tar.gz archive
            if model_path.endswith(".tar.gz"):
                base_name = os.path.basename(model_path).replace(".tar.gz", "")
                extract_path = os.path.join(model_folder, base_name)
                if not os.path.isdir(extract_path):
                    with tarfile.open(model_path, "r:gz") as tar:
                        tar.extractall(model_folder)
                    logging.info(f"Extracted model to: {extract_path}")
                model_path = extract_path

        if self.format == "ncnn":
            self.model = ncnn.Net()
            self.model.load_param(os.path.join(model_path, "best_ncnn_model", "model.ncnn.param"))
            self.model.load_model(os.path.join(model_path, "best_ncnn_model", "model.ncnn.bin"))

        else:
            try:
                onnx_file = model_path if model_path.endswith(".onnx") else os.path.join(model_path, "model.onnx")
                self.ort_session = onnxruntime.InferenceSession(onnx_file)

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

        logging.info(f"Model original pred : {pred}")

        # Remove prediction in bbox occlusion mask
        if len(occlusion_bboxes):
            all_boxes = np.array([b[:4] for b in occlusion_bboxes.values()], dtype=pred.dtype)

            pred_boxes = pred[:, :4].astype(pred.dtype)
            ious = box_iou(pred_boxes, all_boxes)
            max_ious = ious.max(axis=0)
            keep = max_ious <= 0.1
            pred = pred[keep]

        return pred
