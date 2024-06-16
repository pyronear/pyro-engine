# Copyright (C) 2022-2024, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import glob
import io
import json
import logging
import os
import shutil
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2  # type: ignore[import-untyped]
import numpy as np
from PIL import Image
from pyroclient import client
from requests.exceptions import ConnectionError
from requests.models import Response

from pyroengine.utils import box_iou, nms

from .vision import Classifier

__all__ = ["Engine"]

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


def is_day_time(cache, frame, strategy, delta=0):
    """This function allows to know if it is daytime or not. We have two strategies.
    The first one is to take the current time and compare it to the sunset time.
    The second is to see if we have a color image. The ir cameras switch to ir mode at night and
    therefore produce black and white images. This function can use one or more strategies depending on the use case.

    Args:
        cache (Path): cache folder where sunset_sunrise.txt is located
        frame (PIL image): frame to analyze with ir strategy
        strategy (str): Strategy to define day time [None, time, ir or both]
        delta (int): delta before and after sunset / sunrise in sec

    Returns:
        bool: is day time
    """
    is_day = True
    if strategy in ["both", "time"]:
        with open(cache.joinpath("sunset_sunrise.txt")) as f:
            lines = f.readlines()
        sunrise = datetime.strptime(lines[0][:-1], "%H:%M")
        sunset = datetime.strptime(lines[1][:-1], "%H:%M")
        now = datetime.strptime(datetime.now().isoformat().split("T")[1][:5], "%H:%M")
        if (now - sunrise).total_seconds() < -delta or (sunset - now).total_seconds() < -delta:
            is_day = False

    if strategy in ["both", "ir"]:
        frame = np.array(frame)
        if np.max(frame[:, :, 0] - frame[:, :, 1]) == 0:
            is_day = False

    return is_day


class Engine:
    """This implements an object to manage predictions and API interactions for wildfire alerts.

    Args:
        hub_repo: repository on HF Hub to load the ONNX model from
        conf_thresh: confidence threshold to send an alert
        api_url: url of the pyronear API
        cam_creds: api credectials for each camera, the dictionary should be as the one in the example
        latitude: device latitude
        longitude: device longitude
        alert_relaxation: number of consecutive positive detections required to send the first alert, and also
            the number of consecutive negative detections before stopping the alert
        frame_size: Resize frame to frame_size before sending it to the api in order to save bandwidth (H, W)
        cache_backup_period: number of minutes between each cache backup to disk
        frame_saving_period: Send one frame over N to the api for our dataset
        cache_size: maximum number of alerts to save in cache
        day_time_strategy: strategy to define if it's daytime
        kwargs: keyword args of Classifier

    Examples:
        >>> from pyroengine import Engine
        >>> cam_creds ={
        >>> "cam_id_1": {'login':'log1', 'password':'pwd1'},
        >>> "cam_id_2": {'login':'log2', 'password':'pwd2'},
        >>> }
        >>> pyroEngine = Engine("data/model.onnx", 0.25, 'https://api.pyronear.org', cam_creds, 48.88, 2.38)
    """

    def __init__(
        self,
        model_path: Optional[str] = "data/model.onnx",
        conf_thresh: float = 0.25,
        api_url: Optional[str] = None,
        cam_creds: Optional[Dict[str, Dict[str, str]]] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        nb_consecutive_frames: int = 4,
        frame_size: Optional[Tuple[int, int]] = None,
        cache_backup_period: int = 60,
        frame_saving_period: Optional[int] = None,
        cache_size: int = 100,
        cache_folder: str = "data/",
        backup_size: int = 30,
        jpeg_quality: int = 80,
        day_time_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Init engine"""
        # Engine Setup

        self.model = Classifier(model_path)
        self.conf_thresh = conf_thresh

        # API Setup
        if isinstance(api_url, str):
            assert isinstance(latitude, float) and isinstance(longitude, float) and isinstance(cam_creds, dict)
        self.latitude = latitude
        self.longitude = longitude
        self.api_client = {}
        if isinstance(api_url, str) and isinstance(cam_creds, dict):
            # Instantiate clients for each camera
            for _id, vals in cam_creds.items():
                self.api_client[_id] = client.Client(api_url, vals["login"], vals["password"])

        # Cache & relaxation
        self.frame_saving_period = frame_saving_period
        self.nb_consecutive_frames = nb_consecutive_frames
        self.frame_size = frame_size
        self.jpeg_quality = jpeg_quality
        self.cache_backup_period = cache_backup_period
        self.day_time_strategy = day_time_strategy

        # Local backup
        self._backup_size = backup_size

        # Var initialization
        self._states: Dict[str, Dict[str, Any]] = {
            "-1": {"last_predictions": deque([], self.nb_consecutive_frames), "ongoing": False},
        }
        if isinstance(cam_creds, dict):
            for cam_id in cam_creds:
                self._states[cam_id] = {
                    "last_predictions": deque([], self.nb_consecutive_frames),
                    "ongoing": False,
                }

        self.occlusion_masks = {"-1": None}
        if isinstance(cam_creds, dict):
            for cam_id in cam_creds:
                mask_file = cache_folder + "/occlusion_masks/" + cam_id + ".jpg"
                if os.path.isfile(mask_file):
                    self.occlusion_masks[cam_id] = cv2.imread(mask_file, 0)
                else:
                    self.occlusion_masks[cam_id] = None

        # Restore pending alerts cache
        self._alerts: deque = deque([], cache_size)
        self._cache = Path(cache_folder)  # with Docker, the path has to be a bind volume
        assert self._cache.is_dir()
        self._load_cache()
        self.last_cache_dump = datetime.now(timezone.utc)

    def clear_cache(self) -> None:
        """Clear local cache"""
        for file in self._cache.rglob("pending*"):
            file.unlink()

    def _dump_cache(self) -> None:
        # Remove previous dump
        json_path = self._cache.joinpath("pending_alerts.json")
        if json_path.is_file():
            with open(json_path, "rb") as f:
                data = json.load(f)

            for entry in data:
                os.remove(entry["frame_path"])
            os.remove(json_path)

        data = []
        for idx, info in enumerate(self._alerts):
            # Save frame to disk
            info["frame"].save(self._cache.joinpath(f"pending_frame{idx}.jpg"))

            # Save path in JSON
            data.append(
                {
                    "frame_path": str(self._cache.joinpath(f"pending_frame{idx}.jpg")),
                    "cam_id": info["cam_id"],
                    "ts": info["ts"],
                    "localization": info["localization"],
                }
            )

        # JSON dump
        if len(data) > 0:
            with open(json_path, "w") as f:
                json.dump(data, f)

    def _load_cache(self) -> None:
        # Read json
        json_path = self._cache.joinpath("pending_alerts.json")
        if json_path.is_file():
            with open(json_path, "rb") as f:
                data = json.load(f)

            for entry in data:
                # Open image
                frame = Image.open(entry["frame_path"], mode="r")
                self._alerts.append({"frame": frame, "cam_id": entry["cam_id"], "ts": entry["ts"]})

    def heartbeat(self, cam_id: str) -> Response:
        """Updates last ping of device"""
        return self.api_client[cam_id].heartbeat()

    def _update_states(self, frame: Image.Image, preds: np.ndarray, cam_key: str) -> int:
        """Updates the detection states"""

        conf_th = self.conf_thresh * self.nb_consecutive_frames
        # Reduce threshold once we are in alert mode to collect more data
        if self._states[cam_key]["ongoing"]:
            conf_th *= 0.8

        # Get last predictions
        boxes = np.zeros((0, 5))
        boxes = np.concatenate([boxes, preds])
        for _, box, _, _, _ in self._states[cam_key]["last_predictions"]:
            if box.shape[0] > 0:
                boxes = np.concatenate([boxes, box])

        conf = 0
        output_predictions = np.zeros((0, 5))
        # Get the best ones
        if boxes.shape[0]:
            best_boxes = nms(boxes)
            ious = box_iou(best_boxes[:, :4], boxes[:, :4])
            best_boxes_scores = np.array([sum(boxes[iou > 0, 4]) for iou in ious.T])
            combine_predictions = best_boxes[best_boxes_scores > conf_th, :]
            conf = np.max(best_boxes_scores) / (self.nb_consecutive_frames + 1)  # memory + preds

            if len(combine_predictions):

                # send only preds boxes that match combine_predictions
                ious = box_iou(combine_predictions[:, :4], preds[:, :4])
                iou_match = [np.max(iou) > 0 for iou in ious]
                output_predictions = preds[iou_match, :]

                # Limit bbox size for api
                output_predictions = np.round(output_predictions, 3)  # max 3 digit
                output_predictions = output_predictions[:5, :]  # max 5 bbox

        self._states[cam_key]["last_predictions"].append(
            (frame, preds, output_predictions.tolist(), datetime.now(timezone.utc).isoformat(), False)
        )

        # update state
        if conf > self.conf_thresh:
            self._states[cam_key]["ongoing"] = True
        else:
            self._states[cam_key]["ongoing"] = False

        return conf

    def predict(self, frame: Image.Image, cam_id: Optional[str] = None) -> float:
        """Computes the confidence that the image contains wildfire cues

        Args:
            frame: a PIL image
            cam_id: the name of the camera that sent this image
        Returns:
            the predicted confidence
        """

        # Heartbeat
        if len(self.api_client) > 0 and isinstance(cam_id, str):
            try:
                self.heartbeat(cam_id)
            except ConnectionError:
                logging.warning(f"Unable to reach the pyro-api with {cam_id}")

        cam_key = cam_id or "-1"
        # Reduce image size to save bandwidth
        if isinstance(self.frame_size, tuple):
            frame_resize = frame.resize(self.frame_size[::-1], getattr(Image, "BILINEAR"))
        else:
            frame_resize = frame

        if is_day_time(self._cache, frame, self.day_time_strategy):
            # Inference with ONNX
            preds = self.model(frame.convert("RGB"), self.occlusion_masks[cam_key])
            conf = self._update_states(frame_resize, preds, cam_key)

            # Log analysis result
            device_str = f"Camera '{cam_id}' - " if isinstance(cam_id, str) else ""
            pred_str = "Wildfire detected" if conf > self.conf_thresh else "No wildfire"
            logging.info(f"{device_str}{pred_str} (confidence: {conf:.2%})")

            # Alert
            if conf > self.conf_thresh and len(self.api_client) > 0 and isinstance(cam_id, str):
                # Save the alert in cache to avoid connection issues
                for idx, (frame, preds, localization, ts, is_staged) in enumerate(
                    self._states[cam_key]["last_predictions"]
                ):
                    if not is_staged:
                        self._stage_alert(frame, cam_id, ts, localization)
                        self._states[cam_key]["last_predictions"][idx] = frame, preds, localization, ts, True

        else:
            conf = 0  # return default value

        # Check if it's time to backup pending alerts
        ts = datetime.now(timezone.utc)
        if ts > self.last_cache_dump + timedelta(minutes=self.cache_backup_period):
            self._dump_cache()
            self.last_cache_dump = ts

        return float(conf)

    def _upload_frame(self, cam_id: str, media_data: bytes) -> Response:
        """Save frame"""
        logging.info(f"Camera '{cam_id}' - Uploading media...")
        # Create a media
        response = self.api_client[cam_id].create_media_from_device()
        if response.status_code // 100 == 2:
            media = response.json()
            # Upload media
            self.api_client[cam_id].upload_media(media_id=media["id"], media_data=media_data)

        return response

    def _stage_alert(self, frame: Image.Image, cam_id: str, ts: int, localization: list) -> None:
        # Store information in the queue
        self._alerts.append(
            {
                "frame": frame,
                "cam_id": cam_id,
                "ts": ts,
                "media_id": None,
                "alert_id": None,
                "localization": localization,
            }
        )

    def _process_alerts(self) -> None:
        for _ in range(len(self._alerts)):
            # try to upload the oldest element
            frame_info = self._alerts[0]
            cam_id = frame_info["cam_id"]
            logging.info(f"Camera '{cam_id}' - Sending alert from {frame_info['ts']}...")

            # Save alert on device
            self._local_backup(frame_info["frame"], cam_id)

            try:
                # Media creation
                if not isinstance(self._alerts[0]["media_id"], int):
                    self._alerts[0]["media_id"] = self.api_client[cam_id].create_media_from_device().json()["id"]
                # Alert creation
                if not isinstance(self._alerts[0]["alert_id"], int):
                    self._alerts[0]["alert_id"] = (
                        self.api_client[cam_id]
                        .send_alert_from_device(
                            lat=self.latitude,
                            lon=self.longitude,
                            media_id=self._alerts[0]["media_id"],
                            localization=self._alerts[0]["localization"],
                        )
                        .json()["id"]
                    )
                # Media upload
                stream = io.BytesIO()
                frame_info["frame"].save(stream, format="JPEG", quality=self.jpeg_quality)
                response = self.api_client[cam_id].upload_media(
                    self._alerts[0]["media_id"],
                    media_data=stream.getvalue(),
                )
                # Force a KeyError if the request failed
                response.json()["id"]
                # Clear
                self._alerts.popleft()
                logging.info(f"Camera '{cam_id}' - alert sent")
                stream.seek(0)  # "Rewind" the stream to the beginning so we can read its content
            except (KeyError, ConnectionError) as e:
                logging.warning(f"Camera '{cam_id}' - unable to upload cache")
                logging.warning(e)
                break

    def _local_backup(self, img: Image.Image, cam_id: str) -> None:
        """Save image on device

        Args:
            img (Image.Image): Image to save
            cam_id (str): camera id (ip address)
        """
        backup_cache = self._cache.joinpath("backup/alerts/")
        self._clean_local_backup(backup_cache)  # Dump old cache
        backup_cache = backup_cache.joinpath(f"{time.strftime('%Y%m%d')}/{cam_id}")
        backup_cache.mkdir(parents=True, exist_ok=True)
        file = backup_cache.joinpath(f"{time.strftime('%Y%m%d-%H%M%S')}.jpg")
        img.save(file)

    def _clean_local_backup(self, backup_cache) -> None:
        """Clean local backup when it's bigger than _backup_size MB

        Args:
            backup_cache (Path): backup to clean
        """
        backup_by_days = list(backup_cache.glob("*"))
        backup_by_days.sort()
        for folder in backup_by_days:
            s = (
                sum(
                    os.path.getsize(f)
                    for f in glob.glob(str(backup_cache) + "/**/*", recursive=True)
                    if os.path.isfile(f)
                )
                // 1024**2
            )
            if s > self._backup_size:
                shutil.rmtree(folder)
            else:
                break
