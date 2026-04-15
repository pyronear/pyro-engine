# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import io
import logging
import shutil
import signal
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Never, Optional, Tuple

import numpy as np
from PIL import Image
from pyro_predictor import Predictor
from pyroclient import client
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException
from requests.models import Response

__all__ = ["Engine"]

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def handler(_signum: int, _frame: object) -> Never:
    raise TimeoutError("Heartbeat check timed out")


def heartbeat_with_timeout(api_instance: Any, cam_id: str, timeout: int = 1) -> None:  # noqa: ANN401
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        api_instance.heartbeat(cam_id)
    except TimeoutError:
        logger.warning(f"Heartbeat check timed out for {cam_id}")
    except RequestsConnectionError:
        logger.warning(f"Unable to reach the pyro-api with {cam_id}")
    finally:
        signal.alarm(0)


class Engine(Predictor):
    """Manages predictions and API interactions for wildfire alerts.

    Extends Predictor with pyroclient API integration: heartbeats, image uploads, alert staging and caching.

    Args:
        hub_repo: repository on HF Hub to load the ONNX model from
        conf_thresh: confidence threshold to send an alert
        api_url: url of the pyronear API
        cam_creds: api credentials for each camera, the dictionary should be as the one in the example
        alert_relaxation: number of consecutive positive detections required to send the first alert, and also
            the number of consecutive negative detections before stopping the alert
        frame_size: Resize frame to frame_size before sending it to the api in order to save bandwidth (H, W)
        cache_backup_period: number of minutes between each cache backup to disk
        frame_saving_period: Send one frame over N to the api for our dataset
        cache_size: maximum number of alerts to save in cache
        day_time_strategy: strategy to define if it's daytime
        save_captured_frames: save all captured frames for debugging
        save_detections_frames: Save all locally detection frames locally
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
        conf_thresh: float = 0.35,
        model_conf_thresh: float = 0.05,
        max_bbox_size: float = 0.4,
        api_url: Optional[str] = None,
        cam_creds: Optional[Dict[str, Dict[str, str]]] = None,
        nb_consecutive_frames: int = 7,
        frame_size: Optional[Tuple[int, int]] = None,
        cache_backup_period: int = 60,
        frame_saving_period: Optional[int] = None,
        cache_size: int = 100,
        cache_folder: str = "data/",
        backup_size: int = 30,
        jpeg_quality: int = 80,
        day_time_strategy: Optional[str] = None,
        save_captured_frames: Optional[bool] = False,
        save_detections_frames: Optional[bool] = False,
        send_last_image_period: int = 3600,  # 1H
        last_bbox_mask_fetch_period: int = 3600,  # 1H
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        cam_ids = list(cam_creds.keys()) if isinstance(cam_creds, dict) else None
        super().__init__(
            model_path=model_path,
            conf_thresh=conf_thresh,
            model_conf_thresh=model_conf_thresh,
            max_bbox_size=max_bbox_size,
            nb_consecutive_frames=nb_consecutive_frames,
            frame_size=frame_size,
            cam_ids=cam_ids,
            **kwargs,
        )

        # API Setup
        self.api_client: dict[str, Any] = {}
        if isinstance(api_url, str) and isinstance(cam_creds, dict):
            # Instantiate clients for each camera
            for id_, (camera_token, _) in cam_creds.items():
                ip = id_.split("_")[0]
                if ip not in self.api_client:
                    self.api_client[ip] = client.Client(camera_token, api_url)

        # Cache & relaxation
        self.frame_saving_period = frame_saving_period
        self.jpeg_quality = jpeg_quality
        self.cache_backup_period = cache_backup_period
        self.day_time_strategy = day_time_strategy
        self.save_captured_frames = save_captured_frames
        self.save_detections_frames = save_detections_frames
        self.cam_creds = cam_creds
        self.send_last_image_period = send_last_image_period
        self.last_bbox_mask_fetch_period = last_bbox_mask_fetch_period

        # Local backup
        self._backup_size = backup_size

        # Augment states with API-specific fields
        for state in self._states.values():
            state["last_image_sent"] = None
            state["last_bbox_mask_fetch"] = None

        # Occlusion masks: cam_id -> dict of bboxes (keyed by mask id)
        self.occlusion_masks: Dict[str, Dict[Any, Any]] = {}

        # Restore pending alerts cache
        self._alerts: deque = deque(maxlen=cache_size)
        self._cache = Path(cache_folder)  # with Docker, the path has to be a bind volume
        if not self._cache.is_dir():
            raise ValueError(f"Cache folder does not exist: {self._cache}")

    def _new_state(self) -> Dict[str, Any]:
        state = super()._new_state()
        state["last_image_sent"] = None
        state["last_bbox_mask_fetch"] = None
        return state

    def heartbeat(self, cam_id: str) -> Response:
        """Updates last ping of device"""
        ip = cam_id.split("_")[0]
        return self.api_client[ip].heartbeat()

    def predict(
        self,
        frame: Image.Image,
        cam_id: Optional[str] = None,
        occlusion_bboxes: Optional[Dict[Any, Any]] = None,  # noqa: ARG002
        fake_pred: Optional[np.ndarray] = None,
    ) -> float:
        """Computes the confidence that the image contains wildfire cues

        Args:
            frame: a PIL image
            cam_id: the name of the camera that sent this image
            occlusion_bboxes: ignored — Engine manages occlusion masks internally via URL fetch
            fake_pred: replace model prediction by another one for evaluation purposes, need to be given in onnx format:
                fake_pred = [[x1, x2]
                            [y1, y2]
                            [w1, w2]
                            [h1, h2]
                            [conf1, conf2]]
        Returns:
            the predicted confidence
        """
        cam_key = cam_id or "-1"
        if cam_key not in self._states:
            self._states[cam_key] = self._new_state()

        # Reduce image size to save bandwidth
        if isinstance(self.frame_size, tuple):
            frame = frame.resize(self.frame_size[::-1], Image.BILINEAR)  # type: ignore[attr-defined]

        # Heartbeat
        if len(self.api_client) > 0 and isinstance(cam_id, str):
            heartbeat_with_timeout(self, cam_id, timeout=1)
            if (
                self._states[cam_key]["last_image_sent"] is None
                or time.time() - self._states[cam_key]["last_image_sent"] > self.send_last_image_period
            ):
                # send image periodically
                logger.info(f"Uploading periodical image for cam {cam_id}")
                self._states[cam_key]["last_image_sent"] = time.time()
                ip = cam_id.split("_")[0]
                if ip in self.api_client:
                    stream = io.BytesIO()
                    frame.save(stream, format="JPEG", quality=self.jpeg_quality)
                    response = self.api_client[ip].update_last_image(stream.getvalue())
                    logger.info(response.text)

        # Update occlusion masks from API
        if (
            self._states[cam_key]["last_bbox_mask_fetch"] is None
            or time.time() - self._states[cam_key]["last_bbox_mask_fetch"] > self.last_bbox_mask_fetch_period
        ):
            logger.info(f"Update occlusion masks for cam {cam_key}")
            self._states[cam_key]["last_bbox_mask_fetch"] = time.time()
            if isinstance(cam_id, str) and isinstance(self.cam_creds, dict) and cam_id in self.cam_creds:
                _, pose_id = self.cam_creds[cam_id]
                ip = cam_id.split("_")[0]
                if ip in self.api_client:
                    try:
                        response = self.api_client[ip].list_pose_masks(pose_id)
                        response.raise_for_status()
                        masks_data = response.json()
                        bbox_mask_dict: Dict[Any, Any] = {}
                        for mask_entry in masks_data:
                            mask_str = mask_entry["mask"].strip("()")
                            coords = tuple(float(c) for c in mask_str.split(","))
                            bbox_mask_dict[str(mask_entry["id"])] = coords
                        self.occlusion_masks[cam_key] = bbox_mask_dict
                        logger.info(f"Downloaded occlusion masks for cam {cam_key}: {bbox_mask_dict}")
                    except RequestException as e:
                        logger.warning(f"Failed to fetch occlusion masks for cam {cam_key} (pose {pose_id}): {e}")

        # Inference with ONNX
        if fake_pred is None:
            bbox_mask_dict = self.occlusion_masks.get(cam_key, {})
            preds = self.model(frame.convert("RGB"), bbox_mask_dict)
        else:
            if fake_pred.size == 0:
                preds = np.empty((0, 5))
            else:
                # Apply classifier post_process method for confidence filter and nms
                preds = self.model.post_process(fake_pred, pad=(0, 0))
                # Filter predictions larger than max_bbox_size
                preds = preds[(preds[:, 2] - preds[:, 0]) < self.max_bbox_size, :]
                preds = np.reshape(preds, (-1, 5))

        logger.info(f"pred for {cam_key} : {preds}")
        conf = self._update_states(frame, preds, cam_key)

        if self.save_captured_frames:
            self._local_backup(frame, cam_id, is_alert=False)

        # Log analysis result
        device_str = f"Camera '{cam_id}' - " if isinstance(cam_id, str) else ""
        pred_str = "Wildfire detected" if conf > self.conf_thresh else "No wildfire"
        logger.info(f"{device_str}{pred_str} (confidence: {conf:.2%})")

        # Alert
        if conf > self.conf_thresh and len(self.api_client) > 0 and isinstance(cam_id, str):
            # Save the alert in cache to avoid connection issues
            for idx, (frame, preds, bboxes, ts, is_staged) in enumerate(self._states[cam_key]["last_predictions"]):
                if not is_staged:
                    self._stage_alert(frame, cam_id, ts, bboxes)
                    self._states[cam_key]["last_predictions"][idx] = (
                        frame,
                        preds,
                        bboxes,
                        ts,
                        True,
                    )

        return float(conf)

    def _stage_alert(self, frame: Image.Image, cam_id: str, ts: int, bboxes: list) -> None:
        # Store information in the queue
        self._alerts.append({
            "frame": frame,
            "cam_id": cam_id,
            "ts": ts,
            "media_id": None,
            "alert_id": None,
            "bboxes": bboxes,
        })

    def fill_empty_bboxes(self) -> None:
        cam_id_to_indices: Dict[str, list[int]] = {}
        for i, alert in enumerate(self._alerts):
            cam_id_to_indices.setdefault(alert["cam_id"], []).append(i)

        for indices in cam_id_to_indices.values():
            non_empty_indices = [i for i in indices if self._alerts[i]["bboxes"]]
            if not non_empty_indices:
                continue
            for i in indices:
                if not self._alerts[i]["bboxes"]:
                    closest_index = min(non_empty_indices, key=lambda x: abs(x - i))
                    src = np.array(self._alerts[closest_index]["bboxes"], dtype=float)
                    if src.size == 0:
                        continue
                    filled = src.copy()
                    filled[:, -1] = 0.0  # force confidence to 0 for duplicated boxes
                    self._alerts[i]["bboxes"] = [tuple(row) for row in filled.tolist()]

    def _process_alerts(self) -> None:
        if self.cam_creds is not None:
            self.fill_empty_bboxes()
            for _ in range(len(self._alerts)):
                # try to upload the oldest element
                frame_info = self._alerts[0]
                cam_id = frame_info["cam_id"]
                logger.info(f"Camera '{cam_id}' - Sending alert from {frame_info['ts']}...")

                # Save alert on device
                if self.save_detections_frames:
                    self._local_backup(frame_info["frame"], cam_id)

                try:
                    # Detection creation
                    bboxes = self._alerts[0]["bboxes"]
                    if not bboxes:
                        logger.warning(f"Camera '{cam_id}' - skipping alert with empty bboxes")
                        self._alerts.popleft()
                        continue
                    stream = io.BytesIO()
                    frame_info["frame"].save(stream, format="JPEG", quality=self.jpeg_quality)
                    bboxes = [tuple(bboxe) for bboxe in bboxes]
                    _, pose_id = self.cam_creds[cam_id]
                    ip = cam_id.split("_")[0]
                    response = self.api_client[ip].create_detection(stream.getvalue(), bboxes, pose_id)

                    try:
                        response.json()["id"]
                    except ValueError:
                        logger.error(f"Camera '{cam_id}' - non-JSON response body: {response.text}")
                        raise

                    # Clear
                    self._alerts.popleft()
                    logger.info(f"Camera '{cam_id}' - alert sent")
                    stream.seek(0)  # "Rewind" the stream to the beginning so we can read its content

                except (KeyError, RequestsConnectionError, ValueError) as e:
                    logger.error(f"Camera '{cam_id}' - unable to upload cache")
                    logger.error(e)
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

    def _clean_local_backup(self, backup_cache: Path) -> None:
        """Clean local backup when it's bigger than _backup_size MB

        Args:
            backup_cache (Path): backup to clean
        """
        backup_by_days = list(backup_cache.glob("*"))
        backup_by_days.sort()
        for folder in backup_by_days:
            s = sum(f.stat().st_size for f in backup_cache.rglob("*") if f.is_file()) // 1024**2
            if s > self._backup_size:
                shutil.rmtree(folder)
            else:
                break
