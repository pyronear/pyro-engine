# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

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

from pyroclient import client
from .predictor import PyronearPredictor


class PyronearEngine:
    """
    This class is the Pyronear Engine. This engine manage the whole Fire Detection
    process by capturing and saving the image and by predicting if there is a fire or
    not based on this image.

    Args:
        detection_thresh: wildfire detection threshold in [0, 1]
        api_url: url of the pyronear API
        client_creds: api credectials for each pizero, the dictionary should as the one in the example
        frame_saving_period: Send one frame over N to the api for our dataset
        latitude: device latitude
        longitude: device longitude
        cache_size: maximum number of alerts to save in cache
        alert_relaxation: number of consecutive positive detections required to send the first alert, and also
            the number of consecutive negative detections before stopping the alert
        cache_backup_period: number of minutes between each cache backup to disk

    Examples:
        >>> client_creds ={}
        >>> client_creds['pi_zero_id_1']={'login':'log1', 'password':'pwd1'}
        >>> client_creds['pi_zero_id_2']={'login':'log2', 'password':'pwd2'}
        >>> pyroEngine = PyronearEngine(0.6, 'https://api.pyronear.org', client_creds, 50)
    """
    def __init__(
        self,
        detection_thresh: float = 0.5,
        api_url: Optional[str] = None,
        client_creds: Optional[Dict[str, str]] = None,
        frame_saving_period: Optional[int] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        cache_size: int = 100,
        alert_relaxation: int = 3,
        cache_backup_period: int = 60,
    ) -> None:
        """Init engine"""
        # Engine Setup
        self.pyronearPredictor = PyronearPredictor()
        self.detection_thresh = detection_thresh
        self.frame_saving_period = frame_saving_period
        self.alert_relaxation = alert_relaxation

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
            for pi_zero_id in client_creds.keys():
                self.consec_dets[pi_zero_id] = 0
                self.frames_counter[pi_zero_id] = 0
                self.ongoing_alert[pi_zero_id] = False
        else:
            self.consec_dets['-1'] = 0

        if self.api_url is not None:
            # Instantiate clients for each camera
            self.api_client = {}
            for _id, vals in client_creds.items():
                self.api_client[_id] = client.Client(self.api_url, vals['login'], vals['password'])

        # Restore pending alerts cache
        self.pending_alerts = deque([], cache_size)
        self.load_cache_from_disk()
        self.cache_backup_period = cache_backup_period
        self.last_cache_dump = datetime.utcnow()
        self._backup_folder = Path("data/")  # with Docker, the path has to be a bind volume

    def predict(self, frame: Image.Image, pi_zero_id: Optional[int] = None) -> float:
        """ run prediction on comming frame"""
        prob = self.pyronearPredictor.predict(frame.convert('RGB'))  # run prediction
        if pi_zero_id is None:
            logging.info(f"Wildfire detection score ({prob:.2%})")
        else:
            self.heartbeat(pi_zero_id)
            logging.info(f"Wildfire detection score ({prob:.2%}), on device {pi_zero_id}")

        # Alert
        if prob > self.detection_thresh:
            if pi_zero_id is None:
                pi_zero_id = '-1'  # add default key value

            # Don't increment beyond relaxation
            if not self.ongoing_alert[pi_zero_id] and self.consec_dets[pi_zero_id] < self.alert_relaxation:
                self.consec_dets[pi_zero_id] += 1

            if self.consec_dets[pi_zero_id] == self.alert_relaxation:
                self.ongoing_alert[pi_zero_id] = True

            if isinstance(self.api_url, str) and self.ongoing_alert[pi_zero_id]:
                # Save the alert in cache to avoid connection issues
                self.save_to_cache(frame, pi_zero_id)

        # No wildfire
        else:
            if self.consec_dets[pi_zero_id] > 0:
                self.consec_dets[pi_zero_id] -= 1
            # Consider event as finished
            if self.consec_dets[pi_zero_id] == 0:
                self.ongoing_alert[pi_zero_id] = False

        # Uploading pending alerts
        if len(self.pending_alerts) > 0:
            self.upload_pending_alerts()

        # Check if it's time to backup pending alerts
        ts = datetime.utcnow()
        if ts > self.last_cache_dump + timedelta(minutes=self.cache_backup_period):
            self.save_cache_to_disk()
            self.last_cache_dump = ts

        # save frame
        if isinstance(self.api_url, str) and isinstance(self.frame_saving_period, int) and isinstance(pi_zero_id, int):
            self.frames_counter[pi_zero_id] += 1
            if self.frames_counter[pi_zero_id] == self.frame_saving_period:
                # Reset frame counter
                self.frames_counter[pi_zero_id] = 0
                # Send frame to the api
                frame.save(self.stream, format='JPEG')
                self.save_frame(pi_zero_id)
                self.stream.seek(0)  # "Rewind" the stream to the beginning so we can read its content

        return prob

    def send_alert(self, pi_zero_id: int) -> None:
        """Send alert"""
        logging.info("Sending alert...")
        # Create a media
        media_id = self.api_client[pi_zero_id].create_media_from_device().json()["id"]
        # Create an alert linked to the media and the event
        self.api_client[pi_zero_id].send_alert_from_device(lat=self.latitude, lon=self.longitude, media_id=media_id)
        self.api_client[pi_zero_id].upload_media(media_id=media_id, media_data=self.stream.getvalue())

    def upload_frame(self, pi_zero_id: int) -> None:
        """Save frame"""
        logging.info("Uploading media...")
        # Create a media
        media_id = self.api_client[pi_zero_id].create_media_from_device().json()["id"]
        # Send media
        self.api_client[pi_zero_id].upload_media(media_id=media_id, media_data=self.stream.getvalue())

    def heartbeat(self, pi_zero_id: int) -> None:
        """Updates last ping of device"""
        self.api_client[pi_zero_id].heartbeat()

    def save_to_cache(self, frame: Image.Image, pi_zero_id: int) -> None:
        # Store information in the queue
        self.pending_alerts.append(
            {"frame": frame, "pi_zero_id": pi_zero_id, "ts": datetime.utcnow()}
        )

    def upload_pending_alerts(self) -> None:

        for _ in range(len(self.pending_alerts)):
            # try to upload the oldest element
            frame_info = self.pending_alerts[0]

            try:
                frame_info['frame'].save(self.stream, format='JPEG')
                # Send alert to the api
                self.send_alert(frame_info['pi_zero_id'])
                # No need to upload it anymore
                self.pending_alerts.popleft()
                logging.info(f"Alert sent by device {frame_info['pi_zero_id']}")
            except ConnectionError:
                logging.warning(f"Unable to upload cache for device {frame_info['pi_zero_id']}")
                self.stream.seek(0)  # "Rewind" the stream to the beginning so we can read its content
                break

    def save_cache_to_disk(self) -> None:

        # Remove previous dump
        json_path = self._backup_folder.joinpath('pending_alerts.json')
        if json_path.is_file():
            with open(json_path, 'rb') as f:
                data = json.load(f)

            for entry in data:
                os.remove(entry['frame_path'])
            os.remove(json_path)

        data = []
        for idx, info in enumerate(self.pending_alerts):
            # Save frame to disk
            info['frame'].save(self._backup_folder.joinpath(f"pending_frame{idx}.jpg"))

            # Save path in JSON
            data.append({
                "frame_path": str(self._backup_folder.joinpath(f"pending_frame{idx}.jpg")),
                "pi_zero_id": info["pi_zero_id"],
                "ts": info['ts']
            })

        # JSON dump
        if len(data) > 0:
            with open(json_path, 'w') as f:
                json.dump(data, f)

    def load_cache_from_disk(self) -> None:
        # Read json
        json_path = self._backup_folder.joinpath('pending_alerts.json')
        if json_path.is_file():
            with open(json_path, 'rb') as f:
                data = json.load(f)

            for entry in data:
                # Open image
                frame = Image.open(entry['frame_path'], mode='r')
                self.pending_alerts.append(
                    {"frame": frame, "pi_zero_id": entry['pi_zero_id'], "ts": entry['ts']}
                )
