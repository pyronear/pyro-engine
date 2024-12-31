# Copyright (C) 2023-2024, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import os
import platform
import shutil
from typing import Optional, Tuple, Union
from urllib.request import urlretrieve

import ncnn  # type: ignore
import numpy as np
import onnxruntime
from huggingface_hub import HfApi  # type: ignore[import-untyped]
from PIL import Image

from .logger_config import logger
from .utils import DownloadProgressBar, letterbox, nms, xywh2xyxy

__all__ = ["Classifier"]

MODEL_URL_FOLDER = "https://huggingface.co/pyronear/yolov8s/resolve/main/"
MODEL_ID = "pyronear/yolov8s"
MODEL_NAME = "yolov8s.pt"
METADATA_NAME = "model_metadata.json"


# Utility function to save metadata
def save_metadata(metadata_path: str, metadata: dict) -> None:
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


class Classifier:
    def __init__(
        self,
        model_folder: str = "data",
        imgsz: int = 1024,
        conf: float = 0.15,
        iou: float = 0,
        format: str = "ncnn",
        model_path: Optional[str] = None,
    ) -> None:
        if model_path is None:
            if format == "ncnn":
                if not self.is_arm_architecture():
                    logger.info("NCNN format is optimized for arm architecture only, switching to onnx is recommended")
                model = "yolov8s_ncnn_model.zip"
                self.format = "ncnn"
            elif format == "onnx":
                model = "yolov8s.onnx"
                self.format = "onnx"
            else:
                raise ValueError(f"Unsupported format: {format}")

            model_path = os.path.join(model_folder, model)
            metadata_path = os.path.join(model_folder, METADATA_NAME)
            model_url = MODEL_URL_FOLDER + model

            api = HfApi()
            model_info = api.model_info(MODEL_ID, files_metadata=True)
            expected_sha256 = self.get_sha(model_info.siblings)

            if not expected_sha256:
                raise ValueError("SHA256 hash for the model file not found in the Hugging Face model metadata.")

            if os.path.isfile(model_path):
                metadata = self.load_metadata(metadata_path)
                if metadata and metadata.get("sha256") == expected_sha256:
                    logger.info("Model already exists and the SHA256 hash matches. No download needed.")
                else:
                    logger.info("Model exists but the SHA256 hash does not match or the file doesn't exist.")
                    os.remove(model_path)
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
            self.ort_session = onnxruntime.InferenceSession(model_path)

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def is_arm_architecture(self) -> bool:
        return platform.machine().startswith("arm") or platform.machine().startswith("aarch")

    def get_sha(self, siblings: list) -> Optional[str]:
        for file in siblings:
            if file.rfilename == os.path.basename(MODEL_NAME):
                return file.lfs["sha256"]
        return None

    def download_model(self, model_url: str, model_path: str, expected_sha256: str, metadata_path: str) -> None:
        os.makedirs(os.path.split(model_path)[0], exist_ok=True)
        logger.info(f"Downloading model from {model_url} ...")
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=model_path) as t:
            urlretrieve(model_url, model_path, reporthook=t.update_to)
        logger.info("Model downloaded!")

        metadata = {"sha256": expected_sha256}
        save_metadata(metadata_path, metadata)
        logger.info("Metadata saved!")

    def load_metadata(self, metadata_path: str) -> Optional[dict]:
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

    def prep_process(self, pil_img: Image.Image) -> Tuple[Union[np.ndarray, ncnn.Mat], Tuple[int, int]]:
        np_img, pad = letterbox(np.array(pil_img), (self.imgsz, self.imgsz))
        if self.format == "ncnn":
            np_img = ncnn.Mat.from_pixels(np_img, ncnn.Mat.PixelType.PIXEL_BGR, np_img.shape[1], np_img.shape[0])
            mean = [0, 0, 0]
            std = [1 / 255, 1 / 255, 1 / 255]
            np_img.substract_mean_normalize(mean=mean, norm=std)
        else:
            np_img = np.expand_dims(np_img.astype("float32"), axis=0)
            np_img = np.ascontiguousarray(np_img.transpose((0, 3, 1, 2)))
            np_img /= 255.0

        return np_img, pad

    def post_process(self, pred: np.ndarray, pad: Tuple[int, int]) -> np.ndarray:
        pred = pred[:, pred[-1, :] > self.conf]
        pred = np.transpose(pred)
        pred = xywh2xyxy(pred)
        pred = pred[pred[:, 4].argsort()]
        pred = nms(pred)
        pred = pred[::-1]

        if len(pred) > 0:
            left_pad, top_pad = pad
            pred[:, :4:2] -= left_pad
            pred[:, 1:4:2] -= top_pad
            pred[:, :4:2] /= max(1, self.imgsz - 2 * left_pad)
            pred[:, 1:4:2] /= max(1, self.imgsz - 2 * top_pad)
            pred = np.clip(pred, 0, 1)
        else:
            pred = np.zeros((0, 5))

        return pred

    def __call__(self, pil_img: Image.Image, occlusion_mask: Optional[np.ndarray] = None) -> np.ndarray:
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

        pred = self.post_process(pred, pad)

        if occlusion_mask is not None:
            hm, wm = occlusion_mask.shape
            keep = []
            for p in pred.copy():
                p[:4:2] *= wm
                p[1:4:2] *= hm
                p[:4:2] = np.clip(p[:4:2], 0, wm)
                p[1:4:2] = np.clip(p[1:4:2], 0, hm)
                x0, y0, x1, y1 = p.astype(int)[:4]
                keep.append(np.sum(occlusion_mask[y0:y1, x0:x1]) > 0)

            pred = pred[keep]

        return pred
