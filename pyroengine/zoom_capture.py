# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Temporary R&D: save a closer look at validated detections to S3.

For every validated detection on a Reolink PTZ pose, a 1024 px crop is cut from the 4K patrol
frame without moving the camera. The camera is then driven onto the best bbox (goto_preset,
click_to_move, zoom), a second 4K frame is captured and its centre crop is uploaded. One JSON
manifest per trigger records bboxes, click, S3 keys, step timings and the outcome.
"""

from __future__ import annotations

import io
import json
import logging
import operator
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from pyro_camera_api_client.client import PyroCameraAPIClient

from .engine import Engine

__all__ = ["ZoomCapture", "crop_square", "s3_uploader_from_env"]

logger = logging.getLogger(__name__)

ZOOM_LEVEL = 10
CROP_SIDE = 1024
# Bboxes per trigger: zoom on the first, crop the 4K frame for all of them.
MAX_BBOXES = 3
# Seconds kept before the loop deadline for zoom 0 + patrol restart.
RETURN_MARGIN_S = 6.0
# stop_patrol only flags the patrol thread; let its in-flight pose move finish.
PATROL_STOP_SETTLE_S = 1.5
JPEG_QUALITY = 95

# (key, data, content_type) -> None
Uploader = Callable[[str, bytes, str], None]


class _DeadlineError(Exception):
    """Raised before a PTZ step when the loop deadline is already past."""


def s3_uploader_from_env() -> Optional[Tuple[Uploader, str]]:
    """Build an S3 uploader from S3_BUCKET / S3_PREFIX, or None when S3_BUCKET is unset.

    AWS credentials and region come from the standard AWS_* variables.
    """
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return None
    import boto3  # imported lazily: R&D-only dependency

    s3 = boto3.client("s3")
    prefix = os.environ.get("S3_PREFIX", "").strip("/")

    def upload(key: str, data: bytes, content_type: str) -> None:
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)

    return upload, prefix


def crop_square(frame: Image.Image, bbox: Sequence[float], side: int = CROP_SIDE) -> bytes:
    """JPEG of a square of at least `side` px centred on a normalized bbox, kept inside the frame."""
    w, h = frame.size
    x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
    s = min(max(float(side), max(x2 - x1, y2 - y1) * 1.2), float(min(w, h)))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left, top, right, bottom = Engine._fit_box((cx - s / 2.0, cy - s / 2.0, cx + s / 2.0, cy + s / 2.0), w, h)
    region = frame.crop((round(left), round(top), round(right), round(bottom)))
    buf = io.BytesIO()
    region.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


class ZoomCapture:
    """Runs one capture job per trigger in a background thread, at most one PTZ sequence per camera."""

    def __init__(self, upload: Uploader, prefix: str = "") -> None:
        self.upload = upload
        self.prefix = f"{prefix}/" if prefix else ""
        self._ptz_busy: Dict[str, threading.Lock] = {}
        self._threads: List[threading.Thread] = []

    def trigger(
        self,
        client: PyroCameraAPIClient,
        ip: str,
        pose: int,
        name: str,
        frame: Image.Image,
        bboxes: Sequence[Sequence[float]],
        deadline: float,
    ) -> None:
        """Crop the 4K frame now (so it can be dropped) and run uploads + PTZ sequence in a thread."""
        best = sorted(bboxes, key=operator.itemgetter(4), reverse=True)[:MAX_BBOXES]
        if not best:
            return
        crops = [crop_square(frame, b) for b in best]
        self._ptz_busy.setdefault(ip, threading.Lock())
        thread = threading.Thread(
            target=self._run, args=(client, ip, pose, name, best, crops, deadline), daemon=True, name=f"zoom-{ip}"
        )
        self._threads = [t for t in self._threads if t.is_alive()]
        self._threads.append(thread)
        thread.start()

    def wait(self, timeout: float) -> None:
        """Block until every job is done or `timeout` seconds have elapsed."""
        end = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, end - time.monotonic()))

    def _put(self, key: str, data: bytes, content_type: str) -> str:
        self.upload(key, data, content_type)
        return key

    def _run(
        self,
        client: PyroCameraAPIClient,
        ip: str,
        pose: int,
        name: str,
        bboxes: List[Sequence[float]],
        crops: List[bytes],
        deadline: float,
    ) -> None:
        ts = datetime.now(timezone.utc)
        base = f"{self.prefix}{name}/{ts:%Y-%m-%d}/{ts:%Y%m%dT%H%M%SZ}_pose{pose}"
        manifest: Dict[str, Any] = {
            "camera_ip": ip,
            "camera_name": name,
            "pose": pose,
            "timestamp": ts.isoformat(),
            "bboxes": [[round(float(v), 4) for v in b] for b in bboxes],
            "crop_side": CROP_SIDE,
            "zoom_level": ZOOM_LEVEL,
            "crops": [],
            "timings": {},
            "errors": [],
            "status": "started",
        }
        try:
            manifest["crops"] = [self._put(f"{base}_crop{i}.jpg", data, "image/jpeg") for i, data in enumerate(crops)]
            if not self._ptz_busy[ip].acquire(blocking=False):
                manifest["status"] = "busy"
            else:
                try:
                    self._ptz_sequence(client, ip, pose, bboxes[0], base, deadline, manifest)
                finally:
                    self._ptz_busy[ip].release()
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{ip}] zoom capture failed: {e}")
            manifest["status"] = "error"
            manifest["errors"].append(str(e))
        try:
            self._put(f"{base}_manifest.json", json.dumps(manifest, indent=1).encode(), "application/json")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{ip}] zoom capture manifest upload failed: {e}")
        logger.info(f"[{ip}] zoom capture pose {pose}: {manifest['status']} {manifest['timings']}")

    def _ptz_sequence(
        self,
        client: PyroCameraAPIClient,
        ip: str,
        pose: int,
        bbox: Sequence[float],
        base: str,
        deadline: float,
        manifest: Dict[str, Any],
    ) -> None:
        timings = manifest["timings"]

        def step(label: str, fn: Callable[[], Any]) -> Any:  # noqa: ANN401
            if time.monotonic() > deadline:
                raise _DeadlineError(label)
            t0 = time.monotonic()
            result = fn()
            timings[label] = round(time.monotonic() - t0, 2)
            return result

        click_x, click_y = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        manifest["click"] = [round(click_x, 4), round(click_y, 4)]
        patrol_stopped = False
        try:
            step("stop_patrol", lambda: client.stop_patrol(ip))
            patrol_stopped = True
            # ponytail: fixed settle, the API has no "patrol thread idle" signal
            time.sleep(PATROL_STOP_SETTLE_S)
            # The patrol has moved on since the analysed frame: go back to its pose first.
            step("goto_preset", lambda: client.goto_preset(ip, pose))
            step("click_to_move", lambda: client.click_to_move(ip, click_x, click_y))
            step("zoom", lambda: client.zoom(ip, ZOOM_LEVEL))
            zoomed = step("capture", lambda: client.capture_image(ip))
            data = crop_square(zoomed, (0.5, 0.5, 0.5, 0.5))
            manifest["zoom_crop"] = step(
                "upload", lambda: self._put(f"{base}_zoom{ZOOM_LEVEL}.jpg", data, "image/jpeg")
            )
            manifest["status"] = "done"
        except _DeadlineError as e:
            logger.warning(f"[{ip}] zoom capture: deadline reached before {e}")
            manifest["status"] = f"deadline before {e}"
        finally:
            if patrol_stopped:
                for label, fn in (
                    ("zoom_reset", lambda: client.zoom(ip, 0)),
                    ("start_patrol", lambda: client.start_patrol(ip)),
                ):
                    t0 = time.monotonic()
                    try:
                        fn()
                        timings[label] = round(time.monotonic() - t0, 2)
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"[{ip}] zoom capture {label} failed: {e}")
                        manifest["errors"].append(f"{label}: {e}")
