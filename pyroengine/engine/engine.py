# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import io
import os
import json
import logging
from PIL import Image
from pathlib import Path
from requests.exceptions import ConnectionError
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Dict
import numpy as np

from pyroclient import client
from .predictor import PyronearPredictor

logging.basicConfig(
    format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True
)


class PyronearEngine:
    """
    This class is the Pyronear Engine. This engine manage the whole Fire Detection
    process by capturing and saving the image and by predicting if there is a fire or
    not based on this image.

    Args:
        detection_thresh: wildfire detection threshold in [0, 1]
        api_url: url of the pyronear API
        client_creds: api credectials for each pizero, the dictionary should be as the one in the example
        frame_saving_period: Send one frame over N to the api for our dataset
        latitude: device latitude
        longitude: device longitude
        cache_size: maximum number of alerts to save in cache
        alert_relaxation: number of consecutive positive detections required to send the first alert, and also
            the number of consecutive negative detections before stopping the alert
        cache_backup_period: number of minutes between each cache backup to disk
        frame_size: Resize frame to frame_size before sending it to the api in order to save bandwidth
        model_weights: Path / url model yolov5 model weights

    Examples:
        >>> client_creds ={}
        >>> client_creds['cam_id_1']={'login':'log1', 'password':'pwd1'}
        >>> client_creds['cam_id_2']={'login':'log2', 'password':'pwd2'}
        >>> pyroEngine = PyronearEngine(0.6, 'https://api.pyronear.org', client_creds, 50)
    """

    def __init__(
        self,
        detection_thresh: float = 0.25,
        api_url: Optional[str] = None,
        client_creds: Optional[Dict[str, str]] = None,
        frame_saving_period: Optional[int] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        cache_size: int = 100,
        alert_relaxation: int = 3,
        cache_backup_period: int = 60,
        frame_size: tuple = None,
        model_weights: str = None,
    ) -> None:
        """Init engine"""
        # Engine Setup

        self.pyronearPredictor = PyronearPredictor(model_weights, detection_thresh)
        self.detection_thresh = detection_thresh
        self.frame_saving_period = frame_saving_period
        self.alert_relaxation = alert_relaxation
        self.frame_size = frame_size

        # API Setup
        self.api_url = api_url
        self.latitude = latitude
        self.longitude = longitude

        # Var initialization
        self.stream = io.BytesIO()
        self.consec_dets = {}
        self.ongoing_alert = {}
        self.frames_counter = {}
        if isinstance(client_creds, dict):
            for cam_id in client_creds.keys():
                self.consec_dets[cam_id] = 0
                self.frames_counter[cam_id] = 0
                self.ongoing_alert[cam_id] = False
        else:
            self.consec_dets["-1"] = 0
            self.ongoing_alert["-1"] = 0

        if self.api_url is not None:
            # Instantiate clients for each camera
            self.api_client = {}
            for _id, vals in client_creds.items():
                self.api_client[_id] = client.Client(
                    self.api_url, vals["login"], vals["password"]
                )

        # Restore pending alerts cache
        self.pending_alerts = deque([], cache_size)
        self._backup_folder = Path(
            "data/"
        )  # with Docker, the path has to be a bind volume
        self.load_cache_from_disk()
        self.cache_backup_period = cache_backup_period
        self.last_cache_dump = datetime.utcnow()

    def predict(self, frame: Image.Image, cam_id: Optional[int] = None) -> float:
        """run prediction on comming frame"""
        pred = self.pyronearPredictor.predict(frame.convert("RGB"))  # run prediction

        if len(pred) > 0:
            prob = np.max(pred[:, 4])
            if cam_id is None:
                logging.info(f"Wildfire detected with score ({prob:.2%})")
            else:
                self.heartbeat(cam_id)
                logging.info(
                    f"Wildfire detected with score ({prob:.2%}), on device {cam_id}"
                )

        else:
            if cam_id is None:
                logging.info("No wildfire detected")
            else:
                self.heartbeat(cam_id)
                logging.info(f"No wildfire detected on device {cam_id}")

        # Reduce image size to save bandwidth
        if isinstance(self.frame_size, tuple):
            frame = frame.resize(self.frame_size)

        # Alert
        if prob > self.detection_thresh:
            if cam_id is None:
                cam_id = "-1"  # add default key value

            # Don't increment beyond relaxation
            if (
                not self.ongoing_alert[cam_id]
                and self.consec_dets[cam_id] < self.alert_relaxation
            ):
                self.consec_dets[cam_id] += 1

            if self.consec_dets[cam_id] == self.alert_relaxation:
                self.ongoing_alert[cam_id] = True

            if isinstance(self.api_url, str) and self.ongoing_alert[cam_id]:
                # Save the alert in cache to avoid connection issues
                self.save_to_cache(frame, cam_id)

        # No wildfire
        else:
            if cam_id is None:
                cam_id = "-1"  # add default key value

            if self.consec_dets[cam_id] > 0:
                self.consec_dets[cam_id] -= 1
            # Consider event as finished
            if self.consec_dets[cam_id] == 0:
                self.ongoing_alert[cam_id] = False

        # Uploading pending alerts
        if len(self.pending_alerts) > 0:
            self.upload_pending_alerts()

        # Check if it's time to backup pending alerts
        ts = datetime.utcnow()
        if ts > self.last_cache_dump + timedelta(minutes=self.cache_backup_period):
            self.save_cache_to_disk()
            self.last_cache_dump = ts

        # save frame
        if (
            isinstance(self.api_url, str)
            and isinstance(self.frame_saving_period, int)
            and isinstance(cam_id, int)
        ):
            self.frames_counter[cam_id] += 1
            if self.frames_counter[cam_id] == self.frame_saving_period:
                # Reset frame counter
                self.frames_counter[cam_id] = 0
                # Send frame to the api
                frame.save(self.stream, format="JPEG")
                self.save_frame(cam_id)
                self.stream.seek(
                    0
                )  # "Rewind" the stream to the beginning so we can read its content

        return prob

    def send_alert(self, cam_id: int) -> None:
        """Send alert"""
        logging.info("Sending alert...")
        # Create a media
        media_id = self.api_client[cam_id].create_media_from_device().json()["id"]
        # Create an alert linked to the media and the event
        self.api_client[cam_id].send_alert_from_device(
            lat=self.latitude, lon=self.longitude, media_id=media_id
        )
        self.api_client[cam_id].upload_media(
            media_id=media_id, media_data=self.stream.getvalue()
        )

    def upload_frame(self, cam_id: int) -> None:
        """Save frame"""
        logging.info("Uploading media...")
        # Create a media
        media_id = self.api_client[cam_id].create_media_from_device().json()["id"]
        # Send media
        self.api_client[cam_id].upload_media(
            media_id=media_id, media_data=self.stream.getvalue()
        )

    def heartbeat(self, cam_id: int) -> None:
        """Updates last ping of device"""
        self.api_client[cam_id].heartbeat()

    def save_to_cache(self, frame: Image.Image, cam_id: int) -> None:
        # Store information in the queue
        self.pending_alerts.append(
            {"frame": frame, "cam_id": cam_id, "ts": datetime.utcnow()}
        )

    def upload_pending_alerts(self) -> None:

        for _ in range(len(self.pending_alerts)):
            # try to upload the oldest element
            frame_info = self.pending_alerts[0]

            try:
                frame_info["frame"].save(self.stream, format="JPEG")
                # Send alert to the api
                self.send_alert(frame_info["cam_id"])
                # No need to upload it anymore
                self.pending_alerts.popleft()
                logging.info(f"Alert sent by device {frame_info['cam_id']}")
            except ConnectionError:
                logging.warning(
                    f"Unable to upload cache for device {frame_info['cam_id']}"
                )
                self.stream.seek(
                    0
                )  # "Rewind" the stream to the beginning so we can read its content
                break

    def save_cache_to_disk(self) -> None:

        # Remove previous dump
        json_path = self._backup_folder.joinpath("pending_alerts.json")
        if json_path.is_file():
            with open(json_path, "rb") as f:
                data = json.load(f)

            for entry in data:
                os.remove(entry["frame_path"])
            os.remove(json_path)

        data = []
        for idx, info in enumerate(self.pending_alerts):
            # Save frame to disk
            info["frame"].save(self._backup_folder.joinpath(f"pending_frame{idx}.jpg"))

            # Save path in JSON
            data.append(
                {
                    "frame_path": str(
                        self._backup_folder.joinpath(f"pending_frame{idx}.jpg")
                    ),
                    "cam_id": info["cam_id"],
                    "ts": info["ts"],
                }
            )

        # JSON dump
        if len(data) > 0:
            with open(json_path, "w") as f:
                json.dump(data, f)

    def load_cache_from_disk(self) -> None:
        # Read json
        json_path = self._backup_folder.joinpath("pending_alerts.json")
        if json_path.is_file():
            with open(json_path, "rb") as f:
                data = json.load(f)

            for entry in data:
                # Open image
                frame = Image.open(entry["frame_path"], mode="r")
                self.pending_alerts.append(
                    {"frame": frame, "cam_id": entry["cam_id"], "ts": entry["ts"]}
                )
