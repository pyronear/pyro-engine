# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import io
import json
import logging
import os
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from pyroclient import client
from requests.exceptions import ConnectionError
from requests.models import Response

from .vision import Classifier

__all__ = ["Engine"]

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


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
        frame_size: Resize frame to frame_size before sending it to the api in order to save bandwidth
        cache_backup_period: number of minutes between each cache backup to disk
        frame_saving_period: Send one frame over N to the api for our dataset
        cache_size: maximum number of alerts to save in cache
        kwargs: keyword args of Classifier

    Examples:
        >>> from pyroengine import Engine
        >>> cam_creds ={
        >>> "cam_id_1": {'login':'log1', 'password':'pwd1'},
        >>> "cam_id_2": {'login':'log2', 'password':'pwd2'},
        >>> }
        >>> pyroEngine = Engine("pyronear/rexnet1_3x", 0.5, 'https://api.pyronear.org', cam_creds, 48.88, 2.38)
    """

    def __init__(
        self,
        hub_repo: str,
        conf_thresh: float = 0.5,
        api_url: Optional[str] = None,
        cam_creds: Optional[Dict[str, Dict[str, str]]] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        alert_relaxation: int = 3,
        frame_size: Optional[Tuple[int, int]] = None,
        cache_backup_period: int = 60,
        frame_saving_period: Optional[int] = None,
        cache_size: int = 100,
        cache_folder: str = "data/",
        **kwargs: Any,
    ) -> None:
        """Init engine"""
        # Engine Setup

        self.model = Classifier(hub_repo, **kwargs)
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
        self.alert_relaxation = alert_relaxation
        self.frame_size = frame_size
        self.cache_backup_period = cache_backup_period

        # Var initialization
        self._states: Dict[str, Dict[str, Any]] = {
            "-1": {"consec": 0, "frame_count": 0, "ongoing": False},
        }
        if isinstance(cam_creds, dict):
            for cam_id in cam_creds:
                self._states[cam_id] = {"consec": 0, "frame_count": 0, "ongoing": False}

        # Restore pending alerts cache
        self._alerts: deque = deque([], cache_size)
        self._cache = Path(cache_folder)  # with Docker, the path has to be a bind volume
        assert self._cache.is_dir()
        self._load_cache()
        self.last_cache_dump = datetime.utcnow()

    def clear_cache(self) -> None:
        """Clear local cache"""
        for file in self._cache.rglob("*"):
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

    def _update_states(self, conf: float, cam_key: str) -> bool:
        """Updates the detection states"""
        # Detection
        if conf >= self.conf_thresh:
            # Don't increment beyond relaxation
            if not self._states[cam_key]["ongoing"] and self._states[cam_key]["consec"] < self.alert_relaxation:
                self._states[cam_key]["consec"] += 1

            if self._states[cam_key]["consec"] == self.alert_relaxation:
                self._states[cam_key]["ongoing"] = True

            return self._states[cam_key]["ongoing"]
        # No wildfire
        else:
            if self._states[cam_key]["consec"] > 0:
                self._states[cam_key]["consec"] -= 1
            # Consider event as finished
            if self._states[cam_key]["consec"] == 0:
                self._states[cam_key]["ongoing"] = False

        return False

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
            self.heartbeat(cam_id)

        # Inference with ONNX
        pred = float(self.model(frame.convert("RGB")))
        # Log analysis result
        device_str = f"Camera '{cam_id}' - " if isinstance(cam_id, str) else ""
        pred_str = "Wildfire detected" if pred >= self.conf_thresh else "No wildfire"
        logging.info(f"{device_str}{pred_str} (confidence: {pred:.2%})")

        # Reduce image size to save bandwidth
        if isinstance(self.frame_size, tuple):
            frame = frame.resize(self.frame_size, Image.BILINEAR)

        # Alert
        cam_key = cam_id or "-1"
        to_be_staged = self._update_states(pred, cam_key)
        if to_be_staged and len(self.api_client) > 0 and isinstance(cam_id, str):
            # Save the alert in cache to avoid connection issues
            self._stage_alert(frame, cam_id)

        # Uploading pending alerts
        if len(self._alerts) > 0:
            self._process_alerts()

        # Check if it's time to backup pending alerts
        ts = datetime.utcnow()
        if ts > self.last_cache_dump + timedelta(minutes=self.cache_backup_period):
            self._dump_cache()
            self.last_cache_dump = ts

        # save frame
        if len(self.api_client) > 0 and isinstance(self.frame_saving_period, int) and isinstance(cam_id, str):
            self._states[cam_key]["frame_count"] += 1
            if self._states[cam_key]["frame_count"] == self.frame_saving_period:
                # Send frame to the api
                stream = io.BytesIO()
                frame.save(stream, format="JPEG")
                try:
                    self._upload_frame(cam_id, stream.getvalue())
                    # Reset frame counter
                    self._states[cam_key]["frame_count"] = 0
                except ConnectionError:
                    stream.seek(0)  # "Rewind" the stream to the beginning so we can read its content

        return pred

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

    def _stage_alert(self, frame: Image.Image, cam_id: str) -> None:
        # Store information in the queue
        self._alerts.append(
            {
                "frame": frame,
                "cam_id": cam_id,
                "ts": datetime.utcnow().isoformat(),
                "media_id": None,
                "alert_id": None,
            }
        )

    def _process_alerts(self) -> None:

        for _ in range(len(self._alerts)):
            # try to upload the oldest element
            frame_info = self._alerts[0]
            cam_id = frame_info["cam_id"]
            logging.info(f"Camera {cam_id} - Sending alert from {frame_info['ts']}...")

            try:
                # Media creation
                if not isinstance(self._alerts[0]["media_id"], int):
                    self._alerts[0]["media_id"] = self.api_client[cam_id].create_media_from_device().json()["id"]
                # Alert creation
                if not isinstance(self._alerts[0]["alert_id"], int):
                    self._alerts[0]["alert_id"] = (
                        self.api_client[cam_id]
                        .send_alert_from_device(
                            self.latitude,
                            self.longitude,
                            self._alerts[0]["media_id"],
                        )
                        .json()["id"]
                    )

                # Media upload
                stream = io.BytesIO()
                frame_info["frame"].save(stream, format="JPEG")
                self.api_client[cam_id].upload_media(
                    self._alerts[0]["media_id"],
                    media_data=stream.getvalue(),
                ).json()["id"]
                # Clear
                self._alerts.popleft()
                logging.info(f"Camera {cam_id} - alert sent")
                stream.seek(0)  # "Rewind" the stream to the beginning so we can read its content
            except (KeyError, ConnectionError):
                logging.warning(f"Camera {cam_id} - unable to upload cache")
                break
