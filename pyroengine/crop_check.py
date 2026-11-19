# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Temporary R&D: score a native-resolution HD crop around every detection, in shadow mode.

The engine scores the 4K frame resized to HD. For every box it outputs, a native 1280x720 window
of the 4K frame centred on the box is scored again by a second model. Nothing here changes
alerting. Every check is written as one JSON line (both scores per box) and uploaded to S3 in
batches. When the two scores fall on different sides of THRESH, the HD frame and the crop are
uploaded too.
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
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

__all__ = ["CropCheck", "crop_score", "hd_window", "s3_uploader_from_env"]

logger = logging.getLogger(__name__)

HD = (1280, 720)
THRESH = 0.25
# Crops waiting for the worker; each HD RGB crop holds 2.7 MB, so this caps the backlog near 90 MB.
MAX_PENDING_CROPS = 32
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


def hd_window(frame_size: Tuple[int, int], box: Sequence[float] | np.ndarray) -> Window:
    """Pixel window of HD size centred on a normalized box, shifted inside the frame (clipped if smaller)."""
    w, h = frame_size
    cx, cy = (box[0] + box[2]) / 2 * w, (box[1] + box[3]) / 2 * h
    left = max(0, min(round(cx - HD[0] / 2), w - HD[0]))
    top = max(0, min(round(cy - HD[1] / 2), h - HD[1]))
    return left, top, min(w, left + HD[0]), min(h, top + HD[1])


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
    """Scores HD crops in one background thread so the inference loop never waits on it."""

    def __init__(
        self,
        model: Model,
        upload: Uploader,
        prefix: str = "",
        names: Optional[Dict[str, str]] = None,
        max_pending_crops: int = MAX_PENDING_CROPS,
    ) -> None:
        self.model = model
        self.upload = upload
        self.prefix = f"{prefix}/" if prefix else ""
        self.names = names or {}
        self.dropped = 0  # checks skipped because the worker was behind
        self.max_pending_crops = max_pending_crops
        self._pending_crops = 0
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._records: List[str] = []
        self._last_flush = time.monotonic()
        threading.Thread(target=self._worker, name="crop-check", daemon=True).start()

    def submit(self, cam_id: str, frame: Image.Image, hd_jpeg: bytes, preds: np.ndarray) -> None:
        """Queue a check of every box in `preds` (normalized to `frame`); skip it when the backlog is full."""
        boxes = np.reshape(preds, (-1, 5)).copy()
        with self._lock:
            if self._pending_crops + len(boxes) > self.max_pending_crops:
                self.dropped += 1
                logger.debug("[%s] Crop check skipped, worker busy (%d skipped)", cam_id, self.dropped)
                return
            self._pending_crops += len(boxes)
        windows = [hd_window(frame.size, b[:4]) for b in boxes]
        # Cut the crops now so the 4K frame is not kept alive in the queue.
        item = {
            "cam_id": cam_id,
            "ts": datetime.now(timezone.utc),
            "frame_size": frame.size,
            "hd_jpeg": hd_jpeg,
            "boxes": boxes,
            "windows": windows,
            "crops": [frame.crop(win).convert("RGB") for win in windows],
        }
        self._queue.put_nowait(item)

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
                        self._pending_crops -= len(item["boxes"])
                    self._queue.task_done()
            if time.monotonic() - self._last_flush >= FLUSH_PERIOD_S:
                self.flush()

    def _check(self, item: Dict[str, Any]) -> None:
        cam_id = item["cam_id"]
        ip, _, pose = cam_id.partition("_")
        name = self.names.get(ip, ip)
        ts: datetime = item["ts"]
        results = []
        for box, win, crop in zip(item["boxes"], item["windows"], item["crops"], strict=True):
            full = float(box[4])
            crop_conf = crop_score(self.model(crop), box[:4], win, item["frame_size"])
            results.append({
                "bbox": [round(float(v), 4) for v in box[:4]],
                "full": round(full, 3),
                "crop": round(crop_conf, 3),
                "window": list(win),
                "disagree": (full >= THRESH) != (crop_conf >= THRESH),
            })

        keys: Dict[str, str] = {}
        if any(r["disagree"] for r in results):
            base = f"{self.prefix}{name}/{ts:%Y-%m-%d}/{ts:%Y%m%dT%H%M%S}Z" + (f"_pose{pose}" if pose else "")
            images = {"full": item["hd_jpeg"]}
            for i, (r, crop) in enumerate(zip(results, item["crops"], strict=True)):
                if r["disagree"]:
                    images[f"crop{i}"] = _jpeg(crop)
            for label, data in images.items():
                key = f"{base}_{label}.jpg"
                try:
                    self.upload(key, data, "image/jpeg")
                    keys[label] = key
                except Exception as e:  # ruff: ignore[blind-except]
                    logger.warning("[%s] Crop check upload of %s failed: %s", cam_id, label, e)
            logger.info("[%s] Crop check disagreement: %s", cam_id, [(r["full"], r["crop"]) for r in results])

        self._records.append(
            json.dumps({
                "camera": name,
                "cam_id": cam_id,
                "ts": ts.isoformat(),
                "boxes": results,
                "keys": keys,
                "skipped_so_far": self.dropped,
            })
        )

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
