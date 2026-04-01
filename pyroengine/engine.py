# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import io
import logging
import os
import shutil
import signal
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Never, Optional, Tuple

import numpy as np
import requests
from PIL import Image
from pyro_predictor import Predictor
from pyroclient import client
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.models import Response

try:
    import boto3

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

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
        avif_quality: int = 50,
        day_time_strategy: Optional[str] = None,
        save_captured_frames: Optional[bool] = False,
        save_detections_frames: Optional[bool] = False,
        cam_names: Optional[Dict[str, str]] = None,
        save_detections_to_s3: bool = False,
        s3_bucket: str = "test-engine-capture",
        s3_prefix: str = "detections",
        force_detections: bool = False,
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
            for id_, (camera_token, _, _) in cam_creds.items():
                ip = id_.split("_")[0]
                if ip not in self.api_client:
                    self.api_client[ip] = client.Client(camera_token, api_url)

        # Cache & relaxation
        self.frame_saving_period = frame_saving_period
        self.jpeg_quality = jpeg_quality
        self.avif_quality = avif_quality
        self.cache_backup_period = cache_backup_period
        self.day_time_strategy = day_time_strategy
        self.save_captured_frames = save_captured_frames
        self.save_detections_frames = save_detections_frames
        # S3 uploads (R&D): auto-enable if AWS credentials are in env
        env_key = os.environ.get("AWS_ACCESS_KEY_ID")
        if env_key and not save_detections_to_s3:
            save_detections_to_s3 = True
            logger.info(f"S3: auto-enabled, bucket={s3_bucket}, prefix={s3_prefix}")
        else:
            logger.info(f"S3: disabled (AWS_ACCESS_KEY_ID present={bool(env_key)}, boto3={BOTO3_AVAILABLE})")
        self.save_detections_to_s3 = save_detections_to_s3
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self._s3_client: Any = None
        if save_detections_to_s3:
            if not BOTO3_AVAILABLE:
                raise ImportError("boto3 is required for S3 uploads. Install it with: pip install boto3")
            self._s3_client = boto3.client(
                "s3",  # type: ignore[union-attr]
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                region_name="eu-west-3",
            )
            logger.info("S3: client initialized")
        self._original_frames: Dict[str, deque] = {}
        self.force_detections = force_detections
        if force_detections:
            logger.warning("force_detections=True: injecting fake bbox on every frame (TEST MODE)")
        self.cam_names: Dict[str, str] = cam_names or {}
        self.cam_creds = cam_creds
        self.send_last_image_period = send_last_image_period
        self.last_bbox_mask_fetch_period = last_bbox_mask_fetch_period

        # Local backup
        self._backup_size = backup_size

        # Augment states with API-specific fields
        for state in self._states.values():
            state["last_image_sent"] = None
            state["last_bbox_mask_fetch"] = None

        # Occlusion masks
        self.occlusion_masks: Dict[str, Tuple[Optional[str], Dict[Any, Any], int]] = {"-1": (None, {}, 0)}
        if isinstance(cam_creds, dict):
            for cam_id, (_, azimuth, bbox_mask_url) in cam_creds.items():
                self.occlusion_masks[cam_id] = (bbox_mask_url, {}, int(azimuth))

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

        # Keep original frame for S3 4K crop before resize
        original_frame = frame if not isinstance(self.frame_size, tuple) else frame.copy()

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

        # Update occlusion masks
        if (
            self._states[cam_key]["last_bbox_mask_fetch"] is None
            or time.time() - self._states[cam_key]["last_bbox_mask_fetch"] > self.last_bbox_mask_fetch_period
        ):
            logger.info(f"Update occlusion masks for cam {cam_key}")
            self._states[cam_key]["last_bbox_mask_fetch"] = time.time()
            bbox_mask_url, bbox_mask_dict, azimuth = self.occlusion_masks[cam_key]
            if bbox_mask_url is not None:
                full_url = f"{bbox_mask_url}_{azimuth}.json"
                try:
                    response = requests.get(full_url, timeout=5)
                    bbox_mask_dict = response.json()
                    self.occlusion_masks[cam_key] = (bbox_mask_url, bbox_mask_dict, azimuth)
                    logger.info(f"Downloaded occlusion masks for cam {cam_key} at {bbox_mask_url} :{bbox_mask_dict}")
                except requests.exceptions.RequestException:
                    logger.info(f"No occluson available for: {cam_key}")

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

        # TEST MODE: override preds with a fake high-confidence detection
        if self.force_detections:
            preds = np.array([[0.45, 0.45, 0.55, 0.55, 0.95]])

        logger.info(f"pred for {cam_key} : {preds}")
        conf = self._update_states(frame, preds, cam_key)

        # Track original frames in sync with last_predictions for S3 4K crops
        if cam_key not in self._original_frames:
            self._original_frames[cam_key] = deque(maxlen=self.nb_consecutive_frames)
        self._original_frames[cam_key].append(original_frame)

        if self.save_captured_frames:
            self._local_backup(frame, cam_id, is_alert=False)

        # Log analysis result
        device_str = f"Camera '{cam_id}' - " if isinstance(cam_id, str) else ""
        pred_str = "Wildfire detected" if conf > self.conf_thresh else "No wildfire"
        logger.info(f"{device_str}{pred_str} (confidence: {conf:.2%})")

        # Alert
        if conf > self.conf_thresh and len(self.api_client) > 0 and isinstance(cam_id, str):
            # Save the alert in cache to avoid connection issues
            originals = self._original_frames.get(cam_key, deque())
            for idx, (frame, preds, bboxes, ts, is_staged) in enumerate(self._states[cam_key]["last_predictions"]):
                if not is_staged:
                    orig = originals[idx] if idx < len(originals) else None
                    self._stage_alert(frame, cam_id, ts, bboxes, original_frame=orig)
                    self._states[cam_key]["last_predictions"][idx] = (
                        frame,
                        preds,
                        bboxes,
                        ts,
                        True,
                    )

        return float(conf)

    def _stage_alert(
        self,
        frame: Image.Image,
        cam_id: str,
        ts: int,
        bboxes: list,
        original_frame: Optional[Image.Image] = None,
    ) -> None:
        # Store information in the queue
        self._alerts.append({
            "frame": frame,
            "cam_id": cam_id,
            "ts": ts,
            "media_id": None,
            "alert_id": None,
            "bboxes": bboxes,
            "original_frame": original_frame,
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
                    self._alerts[i]["bboxes"] = [tuple(row) for row in filled]

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
                        logger.error(f"Camera '{cam_id}' - non-JSON response body: {response.text}")
                        raise

                    # Upload to S3 (R&D): inference frame + 4K crops
                    if self.save_detections_to_s3:
                        self._upload_to_s3(frame_info)

                    # Clear
                    self._alerts.popleft()
                    logger.info(f"Camera '{cam_id}' - alert sent")
                    stream.seek(0)  # "Rewind" the stream to the beginning so we can read its content

                except (KeyError, RequestsConnectionError, ValueError) as e:
                    logger.warning(f"Camera '{cam_id}' - unable to upload cache")
                    logger.warning(e)
                    break

    @staticmethod
    def _crop_detection(
        image: Image.Image, bbox: tuple, padding: float = 0.2, min_size: int = 112
    ) -> Image.Image:
        """Crop around a detection bbox with padding, enforcing a minimum crop size.

        Args:
            image: source image (typically the original 4K frame)
            bbox: normalized (x1, y1, x2, y2, conf) bounding box
            padding: fractional padding to add around the bbox (0.2 = 20%)
            min_size: minimum crop dimension in pixels
        """
        w, h = image.size
        x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h

        bw, bh = x2 - x1, y2 - y1
        pad_w, pad_h = bw * padding, bh * padding
        x1 -= pad_w
        y1 -= pad_h
        x2 += pad_w
        y2 += pad_h

        # Enforce minimum crop size
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < min_size:
            diff = min_size - crop_w
            x1 -= diff / 2
            x2 += diff / 2
        if crop_h < min_size:
            diff = min_size - crop_h
            y1 -= diff / 2
            y2 += diff / 2

        # Clamp to image bounds
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        return image.crop((x1, y1, x2, y2))

    @staticmethod
    def _multi_resolution_frame(
        high_resolution_frame: Image.Image, low_resolution_frame: Image.Image, bboxes: list
    ) -> Image.Image:
        """Creates an image at high-res size where only bbox regions are sharp, the rest is upscaled low-res.

        Args:
            high_resolution_frame: the original 4K frame
            low_resolution_frame: the resized inference frame
            bboxes: list of normalized [xmin, ymin, xmax, ymax, conf] bounding boxes
        """
        high_res_width, high_res_height = high_resolution_frame.size
        result_frame = low_resolution_frame.resize((high_res_width, high_res_height), Image.BILINEAR)

        for bbox in bboxes:
            high_res_bbox = (
                round(bbox[0] * high_res_width),
                round(bbox[1] * high_res_height),
                round(bbox[2] * high_res_width),
                round(bbox[3] * high_res_height),
            )
            result_frame.paste(high_resolution_frame.crop(high_res_bbox), (high_res_bbox[0], high_res_bbox[1]))
        return result_frame

    def _upload_to_s3(self, frame_info: dict) -> None:
        """Upload inference frame, 4K crops, and multi-resolution AVIF to S3 (R&D)."""
        cam_id = frame_info["cam_id"]
        cam_name = self.cam_names.get(cam_id, cam_id)
        ts = frame_info["ts"]
        date_str = time.strftime("%Y%m%d")
        ts_str = ts if isinstance(ts, str) else time.strftime("%Y%m%d-%H%M%S")
        prefix = f"{self.s3_prefix}/{cam_name}/{cam_id}/{date_str}/{ts_str}"

        try:
            # Upload inference frame
            buf = io.BytesIO()
            frame_info["frame"].save(buf, format="JPEG", quality=self.jpeg_quality)
            buf.seek(0)
            key = f"{prefix}_frame.jpg"
            self._s3_client.upload_fileobj(buf, self.s3_bucket, key)
            logger.info(f"S3: uploaded {key}")

            original = frame_info.get("original_frame")
            bboxes = frame_info.get("bboxes", [])
            if original is not None and bboxes:
                # Upload 4K crops
                for i, bbox in enumerate(bboxes):
                    crop = self._crop_detection(original, bbox)
                    crop_buf = io.BytesIO()
                    crop.save(crop_buf, format="JPEG", quality=self.jpeg_quality)
                    crop_buf.seek(0)
                    crop_key = f"{prefix}_crop_{i}.jpg"
                    self._s3_client.upload_fileobj(crop_buf, self.s3_bucket, crop_key)
                    logger.info(f"S3: uploaded {crop_key}")

                # Upload multi-resolution AVIF
                multires = self._multi_resolution_frame(original, frame_info["frame"], bboxes)
                multires_buf = io.BytesIO()
                multires.save(multires_buf, format="AVIF", quality=self.avif_quality)
                multires_buf.seek(0)
                multires_key = f"{prefix}_multires.avif"
                self._s3_client.upload_fileobj(multires_buf, self.s3_bucket, multires_key)
                logger.info(f"S3: uploaded {multires_key}")
        except Exception:
            logger.warning(f"S3: failed to upload for cam {cam_id}", exc_info=True)

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
