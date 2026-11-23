# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Temporary R&D: score a native-resolution crop around every detection, in shadow mode.

The engine scores the 4K frame resized to HD. For every box it outputs at MIN_CONF or more, a native
1024x1024 window of the 4K frame centred on the box is scored again by a second model. Nothing here
changes alerting. Every check is written as one JSON line (both scores per box) and uploaded to S3 in
batches. When the two scores fall on different sides of THRESH, the original 4K frame and the crop
are uploaded too, so every score can be recomputed offline. Image uploads are capped per camera pose
and per hour, so a long-lasting false positive does not flood the bucket.
"""

from __future__ import annotations

import io
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

__all__ = ["CropCheck", "crop_score", "crop_window", "s3_uploader_from_env"]

logger = logging.getLogger(__name__)

CROP = (1024, 1024)  # the model input size, so the crop is scored without resizing
MIN_CONF = 0.1  # boxes below this are not checked
THRESH = 0.25
# Frames waiting for the worker, the one being checked included; a 4K RGB frame holds 25 MB.
MAX_PENDING_FRAMES = 4
# 4K frame + crop uploads per camera pose and per rolling hour; records are always kept.
MAX_IMAGE_UPLOADS_PER_HOUR = 2
FLUSH_PERIOD_S = 600.0
# Records kept in memory while S3 is unreachable; the oldest are dropped beyond this.
MAX_PENDING_RECORDS = 20_000
JPEG_QUALITY = 95

# (key, data, content_type) -> None
Uploader = Callable[[str, bytes, str], None]
# PIL image -> (N, 5) array of normalized [x1, y1, x2, y2, conf]
Model = Callable[[Image.Image], np.ndarray]
Window = Tuple[int, int, int, int]


def s3_uploader_from_env() -> Optional[Tuple[Uploader, str]]:
    """Build an S3 uploader from S3_BUCKET / S3_PREFIX, or None when S3_BUCKET is unset.

    AWS credentials and region come from the standard AWS_* variables.
    """
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return None
    import boto3  # imported lazily: R&D-only dependency
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 2}))
    prefix = os.environ.get("S3_PREFIX", "").strip("/")

    def upload(key: str, data: bytes, content_type: str) -> None:
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)

    return upload, prefix


def crop_window(frame_size: Tuple[int, int], box: Sequence[float] | np.ndarray) -> Window:
    """Pixel window of CROP size centred on a normalized box, shifted inside the frame (clipped if smaller)."""
    w, h = frame_size
    cx, cy = (box[0] + box[2]) / 2 * w, (box[1] + box[3]) / 2 * h
    left = max(0, min(round(cx - CROP[0] / 2), w - CROP[0]))
    top = max(0, min(round(cy - CROP[1] / 2), h - CROP[1]))
    return left, top, min(w, left + CROP[0]), min(h, top + CROP[1])


def crop_score(
    crop_preds: np.ndarray, box: Sequence[float] | np.ndarray, window: Window, frame_size: Tuple[int, int]
) -> float:
    """Highest confidence among crop detections that overlap the original box, 0 if none."""
    w, h = frame_size
    left, top, right, bottom = window
    cw, ch = right - left, bottom - top
    # Original box in normalized crop coordinates
    bx1, by1 = (box[0] * w - left) / cw, (box[1] * h - top) / ch
    bx2, by2 = (box[2] * w - left) / cw, (box[3] * h - top) / ch
    best = 0.0
    for x1, y1, x2, y2, conf in np.reshape(crop_preds, (-1, 5)):
        if min(x2, bx2) > max(x1, bx1) and min(y2, by2) > max(y1, by1):
            best = max(best, float(conf))
    return best


class CropCheck:
    """Scores native crops in one background thread so the inference loop never waits on it."""

    def __init__(
        self,
        model: Model,
        upload: Uploader,
        prefix: str = "",
        names: Optional[Dict[str, str]] = None,
        max_pending_frames: int = MAX_PENDING_FRAMES,
    ) -> None:
        self.model = model
        self.upload = upload
        self.prefix = f"{prefix}/" if prefix else ""
        self.names = names or {}
        self.dropped = 0  # checks skipped because the worker was behind
        self.max_pending_frames = max_pending_frames
        self._pending_frames = 0
        self._image_uploads: Dict[str, Deque[float]] = defaultdict(deque)  # cam_id -> upload times
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._records: List[str] = []
        self._last_flush = time.monotonic()
        threading.Thread(target=self._worker, name="crop-check", daemon=True).start()

    def submit(self, cam_id: str, frame: Image.Image, preds: np.ndarray) -> None:
        """Queue a check of the boxes in `preds` (normalized to `frame`) at MIN_CONF or more.

        The check is skipped when the backlog is full. The frame is kept until checked, to upload it on disagreement.
        """
        boxes = np.reshape(preds, (-1, 5))
        boxes = boxes[boxes[:, 4] >= MIN_CONF].copy()
        if not len(boxes):
            return
        with self._lock:
            if self._pending_frames >= self.max_pending_frames:
                self.dropped += 1
                logger.debug("[%s] Crop check skipped, worker busy (%d skipped)", cam_id, self.dropped)
                return
            self._pending_frames += 1
        self._queue.put_nowait({"cam_id": cam_id, "ts": datetime.now(timezone.utc), "frame": frame, "boxes": boxes})

    def _worker(self) -> None:
        while True:
            timeout = max(0.0, self._last_flush + FLUSH_PERIOD_S - time.monotonic())
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                item = None
            if item is not None:
                try:
                    self._check(item)
                except Exception:
                    logger.exception("[%s] Crop check failed", item["cam_id"])
                finally:
                    with self._lock:
                        self._pending_frames -= 1
                    self._queue.task_done()
            if time.monotonic() - self._last_flush >= FLUSH_PERIOD_S:
                self.flush()

    def _check(self, item: Dict[str, Any]) -> None:
        cam_id = item["cam_id"]
        ip, _, pose = cam_id.partition("_")
        name = self.names.get(ip, ip)
        ts: datetime = item["ts"]
        frame: Image.Image = item["frame"]
        windows = [crop_window(frame.size, b[:4]) for b in item["boxes"]]
        crops = [frame.crop(win).convert("RGB") for win in windows]
        results = []
        for box, win, crop in zip(item["boxes"], windows, crops, strict=True):
            full = float(box[4])
            crop_conf = crop_score(self.model(crop), box[:4], win, frame.size)
            results.append({
                "bbox": [round(float(v), 4) for v in box[:4]],
                "full": round(full, 3),
                "crop": round(crop_conf, 3),
                "window": list(win),
                "disagree": (full >= THRESH) != (crop_conf >= THRESH),
            })

        keys: Dict[str, str] = {}
        capped = False
        if any(r["disagree"] for r in results):
            logger.info("[%s] Crop check disagreement: %s", cam_id, [(r["full"], r["crop"]) for r in results])
            capped = not self._take_upload_slot(cam_id)
        if any(r["disagree"] for r in results) and not capped:
            base = f"{self.prefix}{name}/{ts:%Y-%m-%d}/{ts:%Y%m%dT%H%M%S}Z" + (f"_pose{pose}" if pose else "")
            images = {"full": _jpeg(frame.convert("RGB"))}
            for i, (r, crop) in enumerate(zip(results, crops, strict=True)):
                if r["disagree"]:
                    images[f"crop{i}"] = _jpeg(crop)
            for label, data in images.items():
                key = f"{base}_{label}.jpg"
                try:
                    self.upload(key, data, "image/jpeg")
                    keys[label] = key
                except Exception as e:  # ruff: ignore[blind-except]
                    logger.warning("[%s] Crop check upload of %s failed: %s", cam_id, label, e)

        self._records.append(
            json.dumps({
                "camera": name,
                "cam_id": cam_id,
                "ts": ts.isoformat(),
                "boxes": results,
                "keys": keys,
                "upload_capped": capped,
                "skipped_so_far": self.dropped,
            })
        )

    def _take_upload_slot(self, cam_id: str) -> bool:
        """Count one image upload for this camera pose, unless it already used its hourly budget."""
        now = time.monotonic()
        times = self._image_uploads[cam_id]
        while times and now - times[0] >= 3600:
            times.popleft()
        if len(times) >= MAX_IMAGE_UPLOADS_PER_HOUR:
            return False
        times.append(now)
        return True

    def flush(self) -> None:
        """Upload the pending records as one JSONL object; keep them for the next try on failure."""
        self._last_flush = time.monotonic()
        if not self._records:
            return
        now = datetime.now(timezone.utc)
        key = f"{self.prefix}records/{now:%Y-%m-%d}/{now:%Y%m%dT%H%M%S}Z_{uuid.uuid4().hex[:8]}.jsonl"
        try:
            self.upload(key, ("\n".join(self._records) + "\n").encode(), "application/x-ndjson")
            self._records = []
        except Exception as e:  # ruff: ignore[blind-except]
            logger.warning("Crop check records upload failed, keeping %d: %s", len(self._records), e)
            self._records = self._records[-MAX_PENDING_RECORDS:]


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()
