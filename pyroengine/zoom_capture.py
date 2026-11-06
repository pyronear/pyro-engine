# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Temporary R&D: save a closer look at validated detections to S3.

Every image is 1280x720 (HD), so they can all be re-analysed the same way later:
- full: the 4K patrol frame resized to HD,
- crop{i}: a native-resolution HD window of that frame around each validated bbox,
- centred: after goto_preset + click_to_move onto the best bbox, the native HD window at the
  centre of a new 4K frame,
- zoom10: the same centre window after zooming in.
Images are uploaded once the camera is back on patrol. One JSON manifest per trigger records
bboxes, click, S3 keys, step timings and the outcome.
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

__all__ = ["ZoomCapture", "crop_hd", "resize_hd", "s3_uploader_from_env"]

logger = logging.getLogger(__name__)

ZOOM_LEVEL = 10
HD = (1280, 720)
# Bboxes per trigger: move onto the first, crop the 4K frame for all of them.
MAX_BBOXES = 3
# Seconds kept before the loop deadline for zoom 0 + patrol restart.
RETURN_MARGIN_S = 6.0
# Seconds a restarted patrol needs to refresh every pose image (4 poses, ~2.5 s each).
PATROL_PASS_S = 12.0
ZOOM_RESET_ATTEMPTS = 2
JPEG_QUALITY = 95

# (key, data, content_type) -> None
Uploader = Callable[[str, bytes, str], None]


class _AbortError(Exception):
    """Raised before a PTZ step when the deadline is past or a stream took the camera."""


def s3_uploader_from_env() -> Optional[Tuple[Uploader, str]]:
    """Build an S3 uploader from S3_BUCKET / S3_PREFIX, or None when S3_BUCKET is unset.

    AWS credentials and region come from the standard AWS_* variables.
    """
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return None
    import boto3  # imported lazily: R&D-only dependency
    from botocore.config import Config

    # Short timeouts: a job holding a camera must never wait minutes on a dead uplink.
    s3 = boto3.client("s3", config=Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 2}))
    prefix = os.environ.get("S3_PREFIX", "").strip("/")

    def upload(key: str, data: bytes, content_type: str) -> None:
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)

    return upload, prefix


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def resize_hd(frame: Image.Image) -> bytes:
    """JPEG of the whole frame resized to HD (the cameras are 16:9, so nothing is distorted)."""
    return _jpeg(frame.resize(HD, Image.Resampling.BILINEAR))


def crop_hd(frame: Image.Image, cx: float, cy: float) -> bytes:
    """JPEG of a native-resolution HD window centred on normalized (cx, cy), kept inside the frame."""
    w, h = frame.size
    x, y = cx * w, cy * h
    half_w, half_h = HD[0] / 2.0, HD[1] / 2.0
    left, top, right, bottom = Engine._fit_box((x - half_w, y - half_h, x + half_w, y + half_h), w, h)
    return _jpeg(frame.crop((round(left), round(top), round(right), round(bottom))))


def _centre(bbox: Sequence[float]) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


class ZoomCapture:
    """Runs one capture job per trigger in a background thread, at most one job per camera."""

    def __init__(self, upload: Uploader, prefix: str = "") -> None:
        self.upload = upload
        self.prefix = f"{prefix}/" if prefix else ""
        self._jobs: Dict[str, threading.Thread] = {}
        # camera ip -> time.monotonic() of the last patrol restart by a job
        self._patrol_restarted_at: Dict[str, float] = {}

    def trigger(
        self,
        client: PyroCameraAPIClient,
        ip: str,
        pose: int,
        name: str,
        frame: Image.Image,
        bboxes: Sequence[Sequence[float]],
        deadline: float,
        should_abort: Callable[[], bool] = lambda: False,
    ) -> None:
        """Encode the 4K frame now (so it can be dropped) and run the PTZ sequence + uploads in a thread.

        Skipped while the previous job of this camera is still running, so a slow uplink never
        piles up threads and crops on the device.
        """
        job = self._jobs.get(ip)
        if job is not None and job.is_alive():
            logger.warning(f"[{ip}] zoom capture: previous job still running, skipping pose {pose}")
            return
        best = sorted(bboxes, key=operator.itemgetter(4), reverse=True)[:MAX_BBOXES]
        if not best:
            return
        full = resize_hd(frame)
        crops = [crop_hd(frame, *_centre(b)) for b in best]
        job = threading.Thread(
            target=self._run,
            args=(client, ip, pose, name, best, full, crops, deadline, should_abort),
            daemon=True,
            name=f"zoom-{ip}",
        )
        self._jobs[ip] = job
        job.start()

    def images_stale(self, ip: str) -> bool:
        """True while a patrol restarted by a job has not yet refreshed every pose image.

        Analysing the camera before that would replay the previous loop's images: duplicated
        frames in the sliding window and in the alert.
        """
        return time.monotonic() - self._patrol_restarted_at.get(ip, float("-inf")) < PATROL_PASS_S

    def wait(self, timeout: float) -> None:
        """Block until every job is done or `timeout` seconds have elapsed."""
        end = time.monotonic() + timeout
        for job in list(self._jobs.values()):
            job.join(max(0.0, end - time.monotonic()))

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
        full: bytes,
        crops: List[bytes],
        deadline: float,
        should_abort: Callable[[], bool],
    ) -> None:
        ts = datetime.now(timezone.utc)
        base = f"{self.prefix}{name}/{ts:%Y-%m-%d}/{ts:%Y%m%dT%H%M%SZ}_pose{pose}"
        manifest: Dict[str, Any] = {
            "camera_ip": ip,
            "camera_name": name,
            "pose": pose,
            "timestamp": ts.isoformat(),
            "bboxes": [[round(float(v), 4) for v in b] for b in bboxes],
            "image_size": list(HD),
            "zoom_level": ZOOM_LEVEL,
            "crops": [],
            "timings": {},
            "errors": [],
            "status": "started",
        }
        shots: Dict[str, bytes] = {}
        try:
            self._ptz_sequence(client, ip, pose, bboxes[0], deadline, should_abort, manifest, shots)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{ip}] zoom capture failed: {e}")
            manifest["status"] = "error"
            manifest["errors"].append(str(e))
        # Uploaded only once the camera is back on patrol: S3 must never hold the camera.
        try:
            manifest["full"] = self._put(f"{base}_full.jpg", full, "image/jpeg")
            manifest["crops"] = [self._put(f"{base}_crop{i}.jpg", data, "image/jpeg") for i, data in enumerate(crops)]
            if "centred" in shots:
                manifest["centred"] = self._put(f"{base}_centred.jpg", shots["centred"], "image/jpeg")
            if "zoom" in shots:
                manifest["zoom_crop"] = self._put(f"{base}_zoom{ZOOM_LEVEL}.jpg", shots["zoom"], "image/jpeg")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{ip}] zoom capture upload failed: {e}")
            manifest["errors"].append(f"upload: {e}")
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
        deadline: float,
        should_abort: Callable[[], bool],
        manifest: Dict[str, Any],
        shots: Dict[str, bytes],
    ) -> None:
        """Drive the camera onto the bbox, adding each HD centre crop to `shots` as soon as it is taken."""
        timings = manifest["timings"]

        def step(label: str, fn: Callable[[], Any]) -> Any:  # noqa: ANN401
            if time.monotonic() > deadline:
                raise _AbortError(f"deadline before {label}")
            if should_abort():
                raise _AbortError(f"stream active before {label}")
            t0 = time.monotonic()
            result = fn()
            timings[label] = round(time.monotonic() - t0, 2)
            return result

        click_x, click_y = _centre(bbox)
        manifest["click"] = [round(click_x, 4), round(click_y, 4)]
        patrol_stopped = False
        try:
            step("stop_patrol", lambda: client.stop_patrol(ip))
            # stop_patrol returns once the patrol thread is gone, so no late patrol move follows.
            patrol_stopped = True
            # The patrol has moved on since the analysed frame: go back to its pose first.
            step("goto_preset", lambda: client.goto_preset(ip, pose))
            step("click_to_move", lambda: client.click_to_move(ip, click_x, click_y))
            shots["centred"] = crop_hd(step("capture_centred", lambda: client.capture_image(ip)), 0.5, 0.5)
            step("zoom", lambda: client.zoom(ip, ZOOM_LEVEL))
            shots["zoom"] = crop_hd(step("capture_zoom", lambda: client.capture_image(ip)), 0.5, 0.5)
            manifest["status"] = "done"
        except _AbortError as e:
            logger.warning(f"[{ip}] zoom capture: {e}")
            manifest["status"] = str(e)
        finally:
            if patrol_stopped:
                self._restore(client, ip, manifest)

    def _restore(self, client: PyroCameraAPIClient, ip: str, manifest: Dict[str, Any]) -> None:
        """Zoom back to 0 (retried: a zoomed patrol would blind the detector) and restart the patrol."""
        plan = [("zoom_reset", lambda: client.zoom(ip, 0))] * ZOOM_RESET_ATTEMPTS + [
            ("start_patrol", lambda: client.start_patrol(ip))
        ]
        done = set()
        for label, fn in plan:
            if label in done:
                continue
            t0 = time.monotonic()
            try:
                fn()
                manifest["timings"][label] = round(time.monotonic() - t0, 2)
                done.add(label)
                if label == "start_patrol":
                    self._patrol_restarted_at[ip] = time.monotonic()
            except Exception as e:  # noqa: BLE001
                logger.error(f"[{ip}] zoom capture {label} failed: {e}")
                manifest["errors"].append(f"{label}: {e}")
