# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import pathlib
import platform
import tarfile
from typing import Optional, Tuple
from urllib.request import urlretrieve

import ncnn
import numpy as np
import onnxruntime
from PIL import Image

from .utils import DownloadProgressBar, letterbox, nms, xywh2xyxy

__all__ = ["Anonymizer"]

MODEL_URL_FOLDER = "https://huggingface.co/pyronear/yolov11n/resolve/main/"
MODEL_NAME = "ncnn_cpu_yolov11n.tar.gz"

logger = logging.getLogger(__name__)


class Anonymizer:
    """
    Object detection-based anonymizer for blurring faces and license plates in images.

    This class loads a YOLOv11n anonymization model in either NCNN or ONNX format,
    performs preprocessing, inference, and postprocessing to detect sensitive regions,
    and returns bounding boxes for anonymization.

    The model can be provided locally via `model_path` (ONNX only) or automatically
    downloaded and extracted from a remote source in either NCNN or ONNX format.

    Args:
        model_folder (str, optional): Directory where the model is stored or will be downloaded.
            Defaults to `"data"`.
        imgsz (int, optional): Inference image size (input resolution to the model).
            Defaults to `640`.
        conf (float, optional): Confidence threshold for detection filtering.
            Defaults to `0.4`.
        iou (float, optional): IoU threshold for non-maximum suppression.
            Defaults to `0`.
        format (str, optional): Model format, either `"ncnn"` or `"onnx"`.
            Defaults to `"ncnn"`.
        model_path (str, optional): Path to a local ONNX model file.
            If provided, `format` is ignored and ONNX is used.

    Raises:
        ValueError: If `model_path` does not point to an existing `.onnx` file.
        ValueError: If `format` is not `"ncnn"` or `"onnx"`.
        RuntimeError: If loading the ONNX model fails.

    Examples:
        >>> from pyroengine.vision import Anonymizer
        >>> anonymizer = Anonymizer(model_format="onnx", conf=0.2)
        >>> from PIL import Image
        >>> img = Image.open("car.jpg")
        >>> detections = anonymizer(img)
        >>> print(detections)  # array of bounding boxes with confidences

    Notes:
        - NCNN format is optimized for ARM architectures (e.g., Raspberry Pi).
        - ONNX format is recommended for x86 and GPU acceleration.
        - The returned detections are normalized to [0, 1] relative to the original image size.
    """

    def __init__(
        self,
        model_folder="data",
        imgsz=640,
        conf=0.25,
        iou=0,
        model_format="ncnn",
        model_path=None,
    ) -> None:
        if model_path:
            if not pathlib.Path(model_path).is_file():
                raise ValueError(f"Model file not found: {model_path}")
            if pathlib.Path(model_path).suffix.lower() != ".onnx":
                raise ValueError(f"Input model_path should point to an ONNX export but currently is {model_path}")
            self.format = "onnx"
        else:
            if model_format == "ncnn":
                if not self.is_arm_architecture():
                    logger.info("NCNN format is optimized for arm architecture only, switching to onnx is recommended")
                model = MODEL_NAME
                self.format = "ncnn"
            elif model_format == "onnx":
                model = MODEL_NAME.replace("ncnn", "onnx")
                self.format = "onnx"
            else:
                raise ValueError("Unsupported format: should be 'ncnn' or 'onnx'")

            model_path = str(pathlib.Path(model_folder) / model)
            model_url = MODEL_URL_FOLDER + model

            if not pathlib.Path(model_path).is_file():
                logger.info(f"Downloading model from {model_url} ...")
                pathlib.Path(model_folder).mkdir(exist_ok=True, parents=True)
                with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=model_path) as t:
                    urlretrieve(model_url, model_path, reporthook=t.update_to)  # noqa: S310
                logger.info("Model downloaded!")

            # Extract .tar.gz archive
            if model_path.endswith(".tar.gz"):
                base_name = pathlib.Path(model_path).name.replace(".tar.gz", "")
                extract_path = str(pathlib.Path(model_folder) / base_name)
                if not pathlib.Path(extract_path).is_dir():
                    pathlib.Path(extract_path).mkdir(exist_ok=True, parents=True)
                    with tarfile.open(model_path, "r:gz") as tar:
                        tar.extractall(extract_path)  # noqa: S202  # trusted source
                    logger.info(f"Extracted model to: {extract_path}")
                model_path = extract_path

        if self.format == "ncnn":
            self.model = ncnn.Net()
            self.model.load_param(str(pathlib.Path(model_path) / "best_ncnn_model" / "model.ncnn.param"))
            self.model.load_model(str(pathlib.Path(model_path) / "best_ncnn_model" / "model.ncnn.bin"))

        else:
            try:
                onnx_file = model_path if model_path.endswith(".onnx") else str(pathlib.Path(model_path) / "model.onnx")
                self.ort_session = onnxruntime.InferenceSession(onnx_file)

            except Exception as e:
                raise RuntimeError(f"Failed to load the ONNX model from {model_path}: {e!s}") from e

            logger.info(f"ONNX model loaded successfully from {model_path}")

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

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
            np_img = ncnn.Mat.from_pixels(np_img, ncnn.Mat.PixelType.PIXEL_RGB, np_img.shape[1], np_img.shape[0])
            mean = [0, 0, 0]
            std = [1 / 255, 1 / 255, 1 / 255]
            np_img.substract_mean_normalize(mean=mean, norm=std)
        else:
            np_img = np.expand_dims(np_img.astype("float32"), axis=0)  # Add batch dimension
            np_img = np.ascontiguousarray(np_img.transpose((0, 3, 1, 2)))  # Convert from BHWC to BCHW format
            np_img /= 255.0  # Normalize to [0, 1]

        return np_img, pad

    def post_process(self, pred: np.ndarray, pad: Tuple[int, int]) -> np.ndarray:
        # pred can be shape (C, N) or (N, C)
        if pred.ndim != 2 or pred.size == 0:
            return np.zeros((0, 5), dtype=np.float32)

        n_ch, n_det = pred.shape
        if not (n_ch < n_det and n_ch >= 5):
            pred = pred.T
            n_ch, n_det = pred.shape
            if n_ch < 5:
                return np.zeros((0, 5), dtype=np.float32)

        # split
        xywh = pred[:4, :]  # shape 4 x N
        cls_scores = pred[4:, :]  # shape K x N, can be empty

        # confidences and class indices on the unfiltered set
        if cls_scores.size == 0:
            conf = pred[-1, :]  # one class export
            cls_idx = np.zeros(n_det, dtype=int)
        else:
            cls_idx = np.argmax(cls_scores, axis=0)
            conf = np.max(cls_scores, axis=0)

        # filter by class index first, keep only classes < 10
        cls_mask = cls_idx < 10
        if not np.any(cls_mask):
            return np.zeros((0, 5), dtype=np.float32)

        xywh = xywh[:, cls_mask]
        conf = conf[cls_mask]
        cls_idx = cls_idx[cls_mask]
        keep_conf = conf > self.conf

        # now apply confidence threshold
        keep = keep_conf
        if not np.any(keep):
            return np.zeros((0, 5), dtype=np.float32)

        xywh = xywh[:, keep]
        conf = conf[keep]

        det = np.concatenate([xywh.T, conf[:, None]], axis=1)  # M x 5
        det[:, :4] = xywh2xyxy(det[:, :4])

        # NMS on [x1 y1 x2 y2 conf]
        det = det[det[:, 4].argsort()]
        det = nms(det)[::-1]

        if det.shape[0] == 0:
            return np.zeros((0, 5), dtype=np.float32)

        # undo letterbox padding and normalize to [0, 1]
        left_pad, top_pad = pad if isinstance(pad, tuple) else tuple(pad)
        det[:, 0:4:2] -= left_pad
        det[:, 1:4:2] -= top_pad
        det[:, 0:4:2] /= float(self.imgsz - 2 * left_pad)
        det[:, 1:4:2] /= float(self.imgsz - 2 * top_pad)
        det[:, :4] = np.clip(det[:, :4], 0.0, 1.0)

        return det.astype(np.float32)

    def __call__(self, pil_img: Image.Image, occlusion_bboxes: Optional[dict] = None) -> np.ndarray:
        """Run the classifier on an input image.

        Args:
            pil_img: The input PIL image.
            occlusion_mask: Optional occlusion mask to exclude certain areas.

        Returns:
            Processed predictions.
        """
        if occlusion_bboxes is None:
            occlusion_bboxes = {}
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

        pred = np.reshape(pred, (-1, 5))

        return pred.tolist()
