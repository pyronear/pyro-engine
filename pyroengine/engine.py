# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import io
import logging
import shutil
import signal
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Never, Optional, Tuple

import numpy as np
from PIL import Image
from pyro_predictor import Predictor
from pyro_predictor.utils import box_iou
from pyroclient import client
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException
from requests.models import Response

__all__ = ["ContextCrop", "Engine"]

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# Context crop kept in RAM instead of the full-resolution frame: the region is sized on what the
# final crop needs (the frozen box, CROP_PADDING around the bbox) plus a fixed jitter margin on
# each side, with a floor of CONTEXT_MIN_SIDE px. The margin is fixed (not a multiple of the bbox)
# so a very large detection cannot blow the stored region up toward the full frame.
CONTEXT_MARGIN = 384
CONTEXT_MIN_SIDE = 1024
CONTEXT_JPEG_QUALITY = 95
# Padding applied around a bbox cluster for the final 224x224 detection crops.
CROP_PADDING = 0.20
# Fraction of a bbox that must fall inside a frozen event crop box to reuse it; below this
# (e.g. a plume that outgrew its box) a new frozen box is added so the crop re-anchors once.
MIN_BBOX_COVERAGE = 0.8


@dataclass(frozen=True)
class ContextCrop:
    """JPEG-encoded region of a full-resolution frame, with its position and the full frame size."""

    jpeg: bytes
    left: int
    top: int
    full_w: int
    full_h: int


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
        nb_consecutive_frames: int = 5,
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

        # Augment states with API-specific fields. Anchor the daily pose timestamp at
        # construction so a startup after noon does not trigger an immediate send;
        # the next noon crossing is the first one that fires.
        init_now = datetime.now()
        for state in self._states.values():
            state["last_image_sent"] = None
            state["last_bbox_mask_fetch"] = None
            state["last_pose_image_sent"] = init_now
            state["event_crop_boxes"] = []

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
        state["last_pose_image_sent"] = datetime.now()
        state["event_crop_boxes"] = []
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

        # Keep the pre-resize frame so detection crops can be taken at original resolution.
        original_frame = frame

        # Reduce image size to save bandwidth
        if isinstance(self.frame_size, tuple):
            target = (self.frame_size[1], self.frame_size[0])  # PIL expects (W, H)
            if frame.size != target:
                frame = frame.resize(target, Image.BILINEAR)  # type: ignore[attr-defined]

        # Encode once for API uploads and on-disk backup; feed the uncompressed frame to the model.
        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=self.jpeg_quality)
        encoded_bytes = buf.getvalue()

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
                    response = self.api_client[ip].update_last_image(encoded_bytes)
                    logger.info(response.text)

            # Send one pose image per day at 12:00
            if isinstance(self.cam_creds, dict) and cam_id in self.cam_creds:
                now = datetime.now()
                today_noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
                last_pose_sent = self._states[cam_key]["last_pose_image_sent"]
                if now >= today_noon and last_pose_sent < today_noon:
                    _, pose_id = self.cam_creds[cam_id]
                    ip = cam_id.split("_")[0]
                    if ip in self.api_client:
                        logger.info(f"Uploading daily pose image for cam {cam_id} (pose {pose_id})")
                        self._states[cam_key]["last_pose_image_sent"] = now
                        response = self.api_client[ip].update_pose_image(pose_id, encoded_bytes)
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
        # Store only a compact JPEG region around the detections so _process_alerts can crop at
        # full resolution without keeping the whole original frame in RAM.
        context_crop = self._build_context_crop(original_frame, preds)
        conf = self._update_states(context_crop, preds, cam_key, encoded_bytes=encoded_bytes)
        if not self._states[cam_key]["ongoing"]:
            # Event over: drop the frozen crop boxes so the next event re-centers.
            self._states[cam_key]["event_crop_boxes"] = []

        if self.save_captured_frames:
            self._local_backup(frame, cam_id, is_alert=False, encoded_bytes=encoded_bytes)

        # Log analysis result
        device_str = f"Camera '{cam_id}' - " if isinstance(cam_id, str) else ""
        pred_str = "Wildfire detected" if conf > self.conf_thresh else "No wildfire"
        logger.info(f"{device_str}{pred_str} (confidence: {conf:.2%})")

        # Alert (use ongoing so hysteresis-relaxed threshold keeps staging frames during a dip)
        if self._states[cam_key]["ongoing"] and len(self.api_client) > 0 and isinstance(cam_id, str):
            state = self._states[cam_key]
            # Collect every bbox the predictor emitted across the window; treat these as
            # tracked locations and backfill missing per-frame bboxes from raw preds with conf=0.
            tracked = [b[:4] for _, _, bbs, _, _, _ in state["last_predictions"] for b in bbs]
            tracked_arr = np.array(tracked, dtype=np.float64) if tracked else np.empty((0, 4))

            # Freeze one square crop box per cluster of tracked bboxes so every frame of the
            # event is cropped at the same location, even when individual bboxes move.
            full_size = next(((cc.full_w, cc.full_h) for cc, *_ in state["last_predictions"] if cc is not None), None)
            if tracked and full_size is not None:
                self._update_event_crop_boxes(cam_key, tracked, *full_size)

            for idx, (crop_, preds_, bboxes, ts, is_staged, jpeg_bytes) in enumerate(state["last_predictions"]):
                if not is_staged:
                    bboxes = self._backfill_bboxes(bboxes, preds_, tracked_arr)
                    crop_boxes = (
                        self._assign_crop_boxes(bboxes, cam_key, *full_size)
                        if bboxes and full_size is not None
                        else None
                    )
                    self._stage_alert(crop_, cam_id, ts, bboxes, jpeg_bytes, crop_boxes)
                    state["last_predictions"][idx] = (
                        crop_,
                        preds_,
                        bboxes,
                        ts,
                        True,
                        jpeg_bytes,
                    )

        return float(conf)

    @staticmethod
    def _fit_box(
        box: Tuple[float, float, float, float], img_w: float, img_h: float
    ) -> Tuple[float, float, float, float]:
        """Shift a box back inside the image to preserve its size; clip only if larger than the image."""
        left, top, right, bottom = box
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > img_w:
            left -= right - img_w
            right = img_w
        if bottom > img_h:
            top -= bottom - img_h
            bottom = img_h
        return max(left, 0.0), max(top, 0.0), right, bottom

    @staticmethod
    def _compute_crop_box(
        bboxes: list,
        img_w: int,
        img_h: int,
        padding: float = CROP_PADDING,
    ) -> Tuple[int, int, int, int]:
        """Square crop covering all bboxes (normalized coords) with `padding` on the largest dim."""
        arr = np.asarray(bboxes, dtype=float)
        x1 = float(arr[:, 0].min()) * img_w
        y1 = float(arr[:, 1].min()) * img_h
        x2 = float(arr[:, 2].max()) * img_w
        y2 = float(arr[:, 3].max()) * img_h

        side = max(x2 - x1, y2 - y1) * (1.0 + padding)
        side = min(side, float(min(img_w, img_h)))

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half = side / 2.0
        left, top, right, bottom = Engine._fit_box((cx - half, cy - half, cx + half, cy + half), img_w, img_h)
        return round(left), round(top), round(right), round(bottom)

    def _build_context_crop(self, frame: Image.Image, preds: np.ndarray) -> Optional[ContextCrop]:
        """Encode a region around the raw predictions instead of keeping the full frame.

        Sized on what the final crop needs (the square frozen box, CROP_PADDING around the largest
        preds-union side) plus a fixed CONTEXT_MARGIN on each side to absorb bbox jitter between the
        frozen box and this frame, with a floor of CONTEXT_MIN_SIDE px. The fixed margin caps the
        region size for very large detections instead of scaling it with the bbox.
        """
        if preds.shape[0] == 0:
            return None
        img_w, img_h = frame.size
        arr = np.asarray(preds, dtype=float)
        x1 = float(arr[:, 0].min()) * img_w
        y1 = float(arr[:, 1].min()) * img_h
        x2 = float(arr[:, 2].max()) * img_w
        y2 = float(arr[:, 3].max()) * img_h

        side = max(x2 - x1, y2 - y1) * (1.0 + CROP_PADDING) + 2.0 * CONTEXT_MARGIN
        target_w = min(max(side, CONTEXT_MIN_SIDE), img_w)
        target_h = min(max(side, CONTEXT_MIN_SIDE), img_h)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        box = self._fit_box(
            (cx - target_w / 2.0, cy - target_h / 2.0, cx + target_w / 2.0, cy + target_h / 2.0), img_w, img_h
        )
        left, top, right, bottom = (round(v) for v in box)
        buf = io.BytesIO()
        frame.crop((left, top, right, bottom)).save(buf, format="JPEG", quality=CONTEXT_JPEG_QUALITY)
        return ContextCrop(jpeg=buf.getvalue(), left=left, top=top, full_w=img_w, full_h=img_h)

    @staticmethod
    def _cluster_bboxes(bboxes: list) -> List[list]:
        """Group bboxes (normalized coords) into clusters of transitively overlapping boxes."""
        clusters = [[list(b[:4]), [b]] for b in bboxes]
        merged = True
        while merged:
            merged = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    a, b = clusters[i][0], clusters[j][0]
                    if a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]:
                        clusters[i][0] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                        clusters[i][1].extend(clusters[j][1])
                        del clusters[j]
                        merged = True
                        break
                if merged:
                    break
        return [members for _, members in clusters]

    def _update_event_crop_boxes(self, cam_key: str, tracked_bboxes: list, full_w: int, full_h: int) -> None:
        """Add a frozen square crop box for any cluster of tracked bboxes not yet covered.

        Existing boxes are never moved or resized, so all crops of one event stay centered on the
        same spot regardless of bbox jitter. A bbox that grew mostly outside its box (coverage below
        MIN_BBOX_COVERAGE) gets a new frozen box, so the crop re-anchors once instead of drifting.
        """
        frozen = self._states[cam_key]["event_crop_boxes"]
        uncovered = [
            bbox
            for bbox in tracked_bboxes
            if not any(self._bbox_coverage(bbox, box, full_w, full_h) >= MIN_BBOX_COVERAGE for box in frozen)
        ]
        for cluster in self._cluster_bboxes(uncovered):
            frozen.append(self._compute_crop_box(cluster, full_w, full_h))

    @staticmethod
    def _bbox_coverage(bbox: list, box: Tuple[int, int, int, int], img_w: int, img_h: int) -> float:
        """Fraction of the bbox area (normalized coords) covered by the pixel box."""
        bx1, by1 = bbox[0] * img_w, bbox[1] * img_h
        bx2, by2 = bbox[2] * img_w, bbox[3] * img_h
        inter_w = max(0.0, min(bx2, box[2]) - max(bx1, box[0]))
        inter_h = max(0.0, min(by2, box[3]) - max(by1, box[1]))
        area = (bx2 - bx1) * (by2 - by1)
        if area <= 0:
            # Degenerate bbox: covered if its center falls inside the box
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            return 1.0 if box[0] <= cx <= box[2] and box[1] <= cy <= box[3] else 0.0
        return inter_w * inter_h / area

    def _assign_crop_boxes(self, bboxes: list, cam_key: str, full_w: int, full_h: int) -> list:
        """Pick, for each bbox, the frozen event box with the largest overlap; add one if none overlaps."""
        frozen = self._states[cam_key]["event_crop_boxes"]
        assigned = []
        for bbox in bboxes:
            bx1, by1 = bbox[0] * full_w, bbox[1] * full_h
            bx2, by2 = bbox[2] * full_w, bbox[3] * full_h
            best, best_area = None, 0.0
            for box in frozen:
                inter_w = max(0.0, min(bx2, box[2]) - max(bx1, box[0]))
                inter_h = max(0.0, min(by2, box[3]) - max(by1, box[1]))
                if inter_w * inter_h > best_area:
                    best, best_area = box, inter_w * inter_h
            if best is None:
                best = self._compute_crop_box([bbox], full_w, full_h)
                frozen.append(best)
            assigned.append(best)
        return assigned

    def _encode_detection_crops(
        self,
        context_crop: Optional[ContextCrop],
        bboxes: list,
        crop_boxes: Optional[list],
    ) -> Optional[list[bytes]]:
        """Cut one 224x224 JPEG per bbox out of the context crop, using the frozen event crop boxes."""
        if context_crop is None or not bboxes or not crop_boxes or len(crop_boxes) != len(bboxes):
            return None
        region = Image.open(io.BytesIO(context_crop.jpeg))
        region_w, region_h = region.size
        crops: list[bytes] = []
        for box in crop_boxes:
            local = self._fit_box(
                (
                    box[0] - context_crop.left,
                    box[1] - context_crop.top,
                    box[2] - context_crop.left,
                    box[3] - context_crop.top,
                ),
                region_w,
                region_h,
            )
            lx1, ly1, lx2, ly2 = (round(v) for v in local)
            # A frozen box larger than the stored region gets clipped above; re-square the
            # crop on its center so the 224x224 resize never distorts the aspect ratio.
            w, h = lx2 - lx1, ly2 - ly1
            if w != h:
                side = min(w, h)
                lx1 += (w - side) // 2
                ly1 += (h - side) // 2
                lx2, ly2 = lx1 + side, ly1 + side
            crop = region.crop((lx1, ly1, lx2, ly2))
            crop_w, crop_h = crop.size
            downscaling = crop_w > 224 or crop_h > 224
            if (crop_w, crop_h) != (224, 224):
                crop = crop.resize((224, 224), Image.LANCZOS)  # type: ignore[attr-defined]
            buf = io.BytesIO()
            if downscaling:
                crop.save(buf, format="JPEG", quality=95)
            else:
                # Crop was at or below 224 — preserve detail with no chroma subsampling.
                crop.save(buf, format="JPEG", quality=100, subsampling=0, optimize=True)
            crops.append(buf.getvalue())
        return crops

    @staticmethod
    def _backfill_bboxes(bboxes: list, preds: np.ndarray, tracked: np.ndarray) -> list:
        """For each raw pred overlapping a tracked location but not yet in bboxes, append it with conf=0."""
        if tracked.shape[0] == 0 or preds.shape[0] == 0:
            return list(bboxes)
        existing = np.array([b[:4] for b in bboxes], dtype=np.float64) if bboxes else np.empty((0, 4))
        ious_tracked = box_iou(preds[:, :4], tracked)  # shape (n_tracked, n_preds)
        hits_tracked = (ious_tracked > 0).any(axis=0)  # per pred
        out = list(bboxes)
        for p_idx in np.where(hits_tracked)[0]:
            box = preds[p_idx, :4]
            if existing.shape[0] and (box_iou(box[None, :], existing) > 0).any():
                continue
            out.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), 0.0])
            existing = np.vstack([existing, box[None, :]])
        return out

    def _stage_alert(
        self,
        context_crop: Optional[ContextCrop],
        cam_id: str,
        ts: int,
        bboxes: list,
        jpeg_bytes: Optional[bytes] = None,
        crop_boxes: Optional[list] = None,
    ) -> None:
        # Store information in the queue
        self._alerts.append({
            "context_crop": context_crop,
            "cam_id": cam_id,
            "ts": ts,
            "media_id": None,
            "alert_id": None,
            "bboxes": bboxes,
            "jpeg_bytes": jpeg_bytes,
            "crop_boxes": crop_boxes,
        })

    def fill_empty_bboxes(self) -> None:
        # Stamp a tiny placeholder bbox at conf=0 on any alert with none,
        # so the upload guard always sees a non-empty payload.
        for alert in self._alerts:
            if not alert["bboxes"]:
                alert["bboxes"] = [(0.0, 0.0, 0.0001, 0.0001, 0.0)]

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
                    self._local_backup(
                        None,
                        cam_id,
                        encoded_bytes=frame_info.get("jpeg_bytes"),
                    )

                try:
                    # Detection creation
                    bboxes = self._alerts[0]["bboxes"]
                    if not bboxes:
                        logger.warning(f"Camera '{cam_id}' - skipping alert with empty bboxes")
                        self._alerts.popleft()
                        continue
                    jpeg_bytes = frame_info.get("jpeg_bytes")
                    if jpeg_bytes is None:
                        # The full frame is no longer kept in RAM, so there is nothing to re-encode.
                        logger.warning(f"Camera '{cam_id}' - skipping alert without encoded frame")
                        self._alerts.popleft()
                        continue
                    bboxes = [tuple(bboxe) for bboxe in bboxes]
                    crops = self._encode_detection_crops(
                        frame_info.get("context_crop"), bboxes, frame_info.get("crop_boxes")
                    )
                    _, pose_id = self.cam_creds[cam_id]
                    ip = cam_id.split("_")[0]
                    response = self.api_client[ip].create_detection(jpeg_bytes, bboxes, pose_id, crops=crops)

                    try:
                        response.json()["id"]
                    except ValueError:
                        logger.error(f"Camera '{cam_id}' - non-JSON response body: {response.text}")
                        raise

                    # Clear
                    self._alerts.popleft()
                    logger.info(f"Camera '{cam_id}' - alert sent")

                except (KeyError, RequestsConnectionError, ValueError) as e:
                    logger.error(f"Camera '{cam_id}' - unable to upload cache")
                    logger.error(e)
                    break

    def _local_backup(
        self,
        img: Optional[Image.Image],
        cam_id: Optional[str],
        is_alert: bool = True,
        encoded_bytes: Optional[bytes] = None,
    ) -> None:
        """Save image on device

        Args:
            img: Image to save; may be None when `encoded_bytes` is provided
            cam_id (str): camera id (ip address)
            is_alert (bool): is the frame an alert ?
            encoded_bytes: pre-encoded JPEG bytes — written verbatim when provided so the
                on-disk file is byte-identical to what was scored / uploaded.
        """
        if img is None and encoded_bytes is None:
            return
        folder = "alerts" if is_alert else "save"
        backup_cache = self._cache.joinpath(f"backup/{folder}/")
        self._clean_local_backup(backup_cache)  # Dump old cache
        backup_cache = backup_cache.joinpath(f"{time.strftime('%Y%m%d')}/{cam_id}")
        backup_cache.mkdir(parents=True, exist_ok=True)
        file = backup_cache.joinpath(f"{time.strftime('%Y%m%d-%H%M%S')}.jpg")
        if encoded_bytes is not None:
            file.write_bytes(encoded_bytes)
        elif img is not None:
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
