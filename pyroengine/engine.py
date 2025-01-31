# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import glob
import io
import json
import logging
import os
import shutil
import signal
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from pyroclient import client
from requests.exceptions import ConnectionError
from requests.models import Response

from pyroengine.utils import box_iou, nms

from .vision import Classifier

__all__ = ["Engine"]

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


def handler(signum, frame):
    raise TimeoutError("Heartbeat check timed out")


def heartbeat_with_timeout(api_instance, cam_id, timeout=1):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        api_instance.heartbeat(cam_id)
    except TimeoutError:
        logging.warning(f"Heartbeat check timed out for {cam_id}")
    except ConnectionError:
        logging.warning(f"Unable to reach the pyro-api with {cam_id}")
    finally:
        signal.alarm(0)


class Engine:
    """This implements an object to manage predictions and API interactions for wildfire alerts.

    Args:
        hub_repo: repository on HF Hub to load the ONNX model from
        conf_thresh: confidence threshold to send an alert
        api_url: url of the pyronear API
        cam_creds: api credectials for each camera, the dictionary should be as the one in the example
        alert_relaxation: number of consecutive positive detections required to send the first alert, and also
            the number of consecutive negative detections before stopping the alert
        frame_size: Resize frame to frame_size before sending it to the api in order to save bandwidth (H, W)
        cache_backup_period: number of minutes between each cache backup to disk
        frame_saving_period: Send one frame over N to the api for our dataset
        cache_size: maximum number of alerts to save in cache
        day_time_strategy: strategy to define if it's daytime
        save_captured_frames: save all captured frames for debugging
        kwargs: keyword args of Classifier

    Examples:
        >>> from pyroengine import Engine
        >>> cam_creds ={
        >>> "cam_id_1": {'login':'log1', 'password':'pwd1'},
        >>> "cam_id_2": {'login':'log2', 'password':'pwd2'},
        >>> }
        >>> pyroEngine = Engine(None, 0.25, 'https://api.pyronear.org', cam_creds, 48.88, 2.38)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.15,
        max_bbox_size: float = 0.4,
        api_url: Optional[str] = None,
        cam_creds: Optional[Dict[str, Dict[str, str]]] = None,
        nb_consecutive_frames: int = 4,
        frame_size: Optional[Tuple[int, int]] = None,
        cache_backup_period: int = 60,
        frame_saving_period: Optional[int] = None,
        cache_size: int = 100,
        cache_folder: str = "data/",
        backup_size: int = 30,
        jpeg_quality: int = 80,
        day_time_strategy: Optional[str] = None,
        save_captured_frames: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        """Init engine"""
        # Engine Setup

        self.model = Classifier(model_path=model_path, conf=0.05, max_bbox_size=max_bbox_size)
        self.conf_thresh = conf_thresh

        # API Setup
        self.api_client: dict[str, Any] = {}
        if isinstance(api_url, str) and isinstance(cam_creds, dict):
            # Instantiate clients for each camera
            for _id, (camera_token, _) in cam_creds.items():
                ip = _id.split("_")[0]
                if ip not in self.api_client.keys():
                    self.api_client[ip] = client.Client(camera_token, api_url)

        # Cache & relaxation
        self.frame_saving_period = frame_saving_period
        self.nb_consecutive_frames = nb_consecutive_frames
        self.frame_size = frame_size
        self.jpeg_quality = jpeg_quality
        self.cache_backup_period = cache_backup_period
        self.day_time_strategy = day_time_strategy
        self.save_captured_frames = save_captured_frames
        self.cam_creds = cam_creds

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

        self.occlusion_masks: Dict[str, Optional[np.ndarray]] = {"-1": None}
        if isinstance(cam_creds, dict):
            for cam_id in cam_creds:
                mask_file = cache_folder + "/occlusion_masks/" + cam_id + ".jpg"
                if os.path.isfile(mask_file):
                    self.occlusion_masks[cam_id] = np.array(Image.open(mask_file).convert(("L")))
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
                    "bboxes": info["bboxes"],
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
        ip = cam_id.split("_")[0]
        return self.api_client[ip].heartbeat()

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
            # We keep only detections with at least two boxes above conf_th
            detections = boxes[boxes[:, -1] > self.conf_thresh, :]
            ious_detections = box_iou(best_boxes[:, :4], detections[:, :4])
            strong_detection = np.sum(ious_detections > 0, 0) > 1
            best_boxes = best_boxes[strong_detection, :]
            if best_boxes.shape[0]:
                ious = box_iou(best_boxes[:, :4], boxes[:, :4])

                best_boxes_scores = np.array([sum(boxes[iou > 0, 4]) for iou in ious.T])
                combine_predictions = best_boxes[best_boxes_scores > conf_th, :]
                conf = np.max(best_boxes_scores) / (self.nb_consecutive_frames + 1)  # memory + preds
                if len(combine_predictions):

                    # send only preds boxes that match combine_predictions
                    ious = box_iou(combine_predictions[:, :4], preds[:, :4])
                    iou_match = [np.max(iou) > 0 for iou in ious]
                    output_predictions = preds[iou_match, :]

                    # Add missing bboxes
                    ious = box_iou(combine_predictions[:, :4], output_predictions[:, :4])
                    missing_bbox = combine_predictions[ious[0] == 0, :]
                    if len(missing_bbox):
                        missing_bbox[:, -1] = 0
                        output_predictions = np.concatenate([output_predictions, missing_bbox])

                    # Limit bbox size for api
                    output_predictions = np.round(output_predictions, 3)  # max 3 digit
                    output_predictions = output_predictions[:5, :]  # max 5 bbox

        # Add default bbox
        if len(output_predictions) == 0:
            output_predictions = np.zeros((1, 5))
            output_predictions[:, 2:4] += 0.0001

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
            heartbeat_with_timeout(self, cam_id, timeout=1)

        cam_key = cam_id or "-1"
        # Reduce image size to save bandwidth
        if isinstance(self.frame_size, tuple):
            frame = frame.resize(self.frame_size[::-1], getattr(Image, "BILINEAR"))

        # Inference with ONNX
        preds = self.model(frame.convert("RGB"), self.occlusion_masks[cam_key])
        print(preds)
        conf = self._update_states(frame, preds, cam_key)

        if self.save_captured_frames:
            self._local_backup(frame, cam_id, is_alert=False)

        # Log analysis result
        device_str = f"Camera '{cam_id}' - " if isinstance(cam_id, str) else ""
        pred_str = "Wildfire detected" if conf > self.conf_thresh else "No wildfire"
        logging.info(f"{device_str}{pred_str} (confidence: {conf:.2%})")

        # Alert
        if conf > self.conf_thresh and len(self.api_client) > 0 and isinstance(cam_id, str):
            # Save the alert in cache to avoid connection issues
            for idx, (frame, preds, bboxes, ts, is_staged) in enumerate(self._states[cam_key]["last_predictions"]):
                if not is_staged:
                    self._stage_alert(frame, cam_id, ts, bboxes)
                    self._states[cam_key]["last_predictions"][idx] = frame, preds, bboxes, ts, True

        # Check if it's time to backup pending alerts
        ts = datetime.now(timezone.utc)
        if ts > self.last_cache_dump + timedelta(minutes=self.cache_backup_period):
            self._dump_cache()
            self.last_cache_dump = ts

        return float(conf)

    def _stage_alert(self, frame: Image.Image, cam_id: str, ts: int, bboxes: list) -> None:
        # Store information in the queue
        self._alerts.append(
            {
                "frame": frame,
                "cam_id": cam_id,
                "ts": ts,
                "media_id": None,
                "alert_id": None,
                "bboxes": bboxes,
            }
        )

    def _process_alerts(self) -> None:
        if self.cam_creds is not None:
            for _ in range(len(self._alerts)):
                # try to upload the oldest element
                frame_info = self._alerts[0]
                cam_id = frame_info["cam_id"]
                logging.info(f"Camera '{cam_id}' - Sending alert from {frame_info['ts']}...")

                # Save alert on device
                self._local_backup(frame_info["frame"], cam_id)

                try:
                    # Detection creation
                    stream = io.BytesIO()
                    frame_info["frame"].save(stream, format="JPEG", quality=self.jpeg_quality)
                    bboxes = self._alerts[0]["bboxes"]
                    bboxes = [tuple(bboxe) for bboxe in bboxes]
                    _, cam_azimuth = self.cam_creds[cam_id]
                    ip = cam_id.split("_")[0]
                    response = self.api_client[ip].create_detection(stream.getvalue(), cam_azimuth, bboxes)
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

    def _local_backup(self, img: Image.Image, cam_id: Optional[str], is_alert: bool = True) -> None:
        """Save image on device

        Args:
            img (Image.Image): Image to save
            cam_id (str): camera id (ip address)
            is_alert (bool): is the frame an alert ?
        """
        folder = "alerts" if is_alert else "save"
        backup_cache = self._cache.joinpath(f"backup/{folder}/")
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
