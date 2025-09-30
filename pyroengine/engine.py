# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import glob
import io
import logging
import shutil
import signal
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import requests
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
        model_conf_thresh: float = 0.05,
        max_bbox_size: float = 0.4,
        api_url: Optional[str] = None,
        cam_creds: Optional[Dict[str, Dict[str, str]]] = None,
        nb_consecutive_frames: int = 8,
        frame_size: Optional[Tuple[int, int]] = None,
        cache_backup_period: int = 60,
        frame_saving_period: Optional[int] = None,
        cache_size: int = 100,
        cache_folder: str = "data/",
        backup_size: int = 30,
        jpeg_quality: int = 80,
        day_time_strategy: Optional[str] = None,
        save_captured_frames: Optional[bool] = False,
        send_last_image_period: int = 3600,  # 1H
        last_bbox_mask_fetch_period: int = 3600,  # 1H
        **kwargs: Any,
    ) -> None:
        """Init engine"""
        # Engine Setup

        self.model = Classifier(model_path=model_path, conf=model_conf_thresh, max_bbox_size=max_bbox_size)
        self.conf_thresh = conf_thresh
        self.model_conf_thresh = model_conf_thresh
        self.max_bbox_size = max_bbox_size

        # API Setup
        self.api_client: dict[str, Any] = {}
        if isinstance(api_url, str) and isinstance(cam_creds, dict):
            # Instantiate clients for each camera
            for _id, (camera_token, _, _) in cam_creds.items():
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
        self.send_last_image_period = send_last_image_period
        self.last_bbox_mask_fetch_period = last_bbox_mask_fetch_period

        # Local backup
        self._backup_size = backup_size

        # Var initialization
        self._states: Dict[str, Dict[str, Any]] = {
            "-1": {
                "last_predictions": deque(maxlen=self.nb_consecutive_frames),
                "ongoing": False,
                "last_image_sent": None,
                "last_bbox_mask_fetch": None,
                "anchor_bbox": None,
                "anchor_ts": None,
                "miss_count": 0,
            },
        }
        if isinstance(cam_creds, dict):
            for cam_id in cam_creds:
                self._states[cam_id] = {
                    "last_predictions": deque(maxlen=self.nb_consecutive_frames),
                    "ongoing": False,
                    "last_image_sent": None,
                    "last_bbox_mask_fetch": None,
                    "anchor_bbox": None,
                    "anchor_ts": None,
                    "miss_count": 0,
                }

        self.occlusion_masks: Dict[str, Tuple[Optional[str], Dict[Any, Any], int]] = {"-1": (None, {}, 0)}
        if isinstance(cam_creds, dict):
            for cam_id, (_, azimuth, bbox_mask_url) in cam_creds.items():
                self.occlusion_masks[cam_id] = (bbox_mask_url, {}, int(azimuth))

        # Restore pending alerts cache
        self._alerts: deque = deque(maxlen=cache_size)
        self._cache = Path(cache_folder)  # with Docker, the path has to be a bind volume
        assert self._cache.is_dir()

    def heartbeat(self, cam_id: str) -> Response:
        """Updates last ping of device"""
        ip = cam_id.split("_")[0]
        return self.api_client[ip].heartbeat()

    def _update_states(self, frame: Image.Image, preds: np.ndarray, cam_key: str) -> float:
        prev_ongoing = self._states[cam_key]["ongoing"]

        conf_th = self.conf_thresh * self.nb_consecutive_frames
        if prev_ongoing:
            conf_th *= 0.8

        boxes = np.zeros((0, 5), dtype=np.float64)
        boxes = np.concatenate([boxes, preds])
        for _, box, _, _, _ in self._states[cam_key]["last_predictions"]:
            if box.shape[0] > 0:
                boxes = np.concatenate([boxes, box])

        conf = 0.0
        output_predictions: npt.NDArray[np.float64] = np.zeros((0, 5), dtype=np.float64)

        if boxes.shape[0]:
            best_boxes = nms(boxes)
            detections = boxes[boxes[:, -1] > self.conf_thresh, :]
            ious_detections = box_iou(best_boxes[:, :4], detections[:, :4])
            strong_detection = np.sum(ious_detections > 0, axis=0) >= int(self.nb_consecutive_frames / 2)
            best_boxes = best_boxes[strong_detection, :]

            if best_boxes.shape[0]:
                ious = box_iou(best_boxes[:, :4], boxes[:, :4])
                best_boxes_scores = np.array([sum(boxes[iou > 0, 4]) for iou in ious.T])
                combine_predictions = best_boxes[best_boxes_scores > conf_th, :]
                if len(best_boxes_scores) > 0:
                    conf = np.max(best_boxes_scores) / (self.nb_consecutive_frames + 1)

                if combine_predictions.shape[0] > 0:
                    ious = box_iou(combine_predictions[:, :4], preds[:, :4])
                    iou_match = np.array([np.max(iou) > 0 for iou in ious], dtype=bool)
                    matched_preds = preds[iou_match, :]
                    if matched_preds.ndim == 1:
                        matched_preds = matched_preds[np.newaxis, :]
                    output_predictions = matched_preds.astype(np.float64)

        # no zero confidence fabrication before ongoing
        # if empty and we were already ongoing, reuse anchor but set conf to 0
        if output_predictions.shape[0] == 0:
            anchor = self._states[cam_key]["anchor_bbox"]
            if prev_ongoing and anchor is not None:
                output_predictions = anchor.copy()
                output_predictions[:, -1] = 0.0  # filled during ongoing, confidence forced to 0
            else:
                output_predictions = np.empty((0, 5), dtype=np.float64)  # stays empty for backfill later
        else:
            # refresh anchor during ongoing with light smoothing
            if prev_ongoing:
                best_idx = int(np.argmax(output_predictions[:, 4]))
                best = output_predictions[best_idx : best_idx + 1]
                anchor = self._states[cam_key]["anchor_bbox"]
                if anchor is None:
                    self._states[cam_key]["anchor_bbox"] = best.copy()
                else:
                    alpha = 0.3
                    self._states[cam_key]["anchor_bbox"] = alpha * best + (1.0 - alpha) * anchor
                self._states[cam_key]["miss_count"] = 0

        output_predictions = np.round(output_predictions, 3)
        output_predictions = output_predictions[:5, :]
        if output_predictions.size > 0:
            output_predictions = np.atleast_2d(output_predictions)

        self._states[cam_key]["last_predictions"].append((
            frame,
            preds,
            output_predictions.tolist(),  # [] if empty
            datetime.now(timezone.utc).isoformat(),
            False,
        ))

        new_ongoing = conf > self.conf_thresh
        if prev_ongoing and not new_ongoing:
            self._states[cam_key]["anchor_bbox"] = None
            self._states[cam_key]["anchor_ts"] = None
            self._states[cam_key]["miss_count"] = 0
        elif not prev_ongoing and new_ongoing:
            if output_predictions.size > 0:
                self._states[cam_key]["anchor_bbox"] = output_predictions.copy()
                self._states[cam_key]["miss_count"] = 0

        self._states[cam_key]["ongoing"] = new_ongoing
        return conf

    def predict(
        self, frame: Image.Image, cam_id: Optional[str] = None, fake_pred: Optional[np.ndarray] = None
    ) -> float:
        """Computes the confidence that the image contains wildfire cues

        Args:
            frame: a PIL image
            cam_id: the name of the camera that sent this image
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
                logging.info(f"Uploading periodical image for cam {cam_id}")
                self._states[cam_key]["last_image_sent"] = time.time()
                ip = cam_id.split("_")[0]
                if ip in self.api_client.keys():
                    stream = io.BytesIO()
                    frame.save(stream, format="JPEG", quality=self.jpeg_quality)
                    response = self.api_client[ip].update_last_image(stream.getvalue())
                    logging.info(response.text)

        # Update occlusion masks
        if (
            self._states[cam_key]["last_bbox_mask_fetch"] is None
            or time.time() - self._states[cam_key]["last_bbox_mask_fetch"] > self.last_bbox_mask_fetch_period
        ):
            logging.info(f"Update occlusion masks for cam {cam_key}")
            self._states[cam_key]["last_bbox_mask_fetch"] = time.time()
            bbox_mask_url, bbox_mask_dict, azimuth = self.occlusion_masks[cam_key]
            if bbox_mask_url is not None:
                full_url = f"{bbox_mask_url}_{azimuth}.json"
                try:
                    response = requests.get(full_url)
                    bbox_mask_dict = response.json()
                    self.occlusion_masks[cam_key] = (bbox_mask_url, bbox_mask_dict, azimuth)
                    logging.info(f"Downloaded occlusion masks for cam {cam_key} at {bbox_mask_url} :{bbox_mask_dict}")
                except requests.exceptions.RequestException:
                    logging.info(f"No occluson available for: {cam_key}")

        # Inference with ONNX
        if fake_pred is None:
            _, bbox_mask_dict, _ = self.occlusion_masks[cam_key]
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

        logging.info(f"pred for {cam_key} : {preds}")
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

    def fill_empty_bboxes(self):
        cam_id_to_indices: Dict[str, list[int]] = {}
        for i, alert in enumerate(self._alerts):
            cam_id_to_indices.setdefault(alert["cam_id"], []).append(i)

        for cam_id, indices in cam_id_to_indices.items():
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
                    self._alerts[i]["bboxes"] = [tuple(row) for row in filled]

    def _process_alerts(self) -> None:
        if self.cam_creds is not None:
            self.fill_empty_bboxes()
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
                    _, cam_azimuth, _ = self.cam_creds[cam_id]
                    ip = cam_id.split("_")[0]
                    response = self.api_client[ip].create_detection(stream.getvalue(), cam_azimuth, bboxes)

                    try:
                        # Force a KeyError if the request failed
                        response.json()["id"]
                    except ValueError:
                        logging.error(f"Camera '{cam_id}' - non-JSON response body: {response.text}")
                        raise

                    # Clear
                    self._alerts.popleft()
                    logging.info(f"Camera '{cam_id}' - alert sent")
                    stream.seek(0)  # "Rewind" the stream to the beginning so we can read its content

                except (KeyError, ConnectionError, ValueError) as e:
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
                    Path(f).stat().st_size
                    for f in glob.glob(str(backup_cache) + "/**/*", recursive=True)
                    if Path(f).is_file()
                )
                // 1024**2
            )
            if s > self._backup_size:
                shutil.rmtree(folder)
            else:
                break
