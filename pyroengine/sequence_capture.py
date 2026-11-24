# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Temporary R&D: save the 4K frames of every new detection sequence to S3.

When a camera pose gets a box at MIN_CONF or more after IDLE_S without any, the next FRAMES frames of that
pose are uploaded as JPEG, the triggering one and the ones without boxes included, then a manifest of the
boxes the engine saw on each frame. Offline, each sequence can be replayed through the whole pipeline (HD
detection, crop refinement, temporal model) to measure which variant raises an alert. Nothing here changes
alerting: frames are encoded and uploaded in one background thread.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .crop_check import Uploader, _jpeg

__all__ = ["SequenceCapture"]

logger = logging.getLogger(__name__)

MIN_CONF = 0.05  # the engine's own confidence threshold: any box it outputs starts a sequence
FRAMES = 20  # frames per sequence, the triggering one included
IDLE_S = 600.0  # a box starts a new sequence only after this long without any box on the pose
MAX_SEQUENCES_PER_DAY = 30  # per camera pose, at most FRAMES x ~260 KB = ~5 MB each
# Frames waiting for the worker, the one being uploaded included; a 4K RGB frame holds 25 MB.
MAX_PENDING_FRAMES = 4


class SequenceCapture:
    """Tracks sequences on the inference thread and uploads their frames in one background thread."""

    def __init__(
        self,
        upload: Uploader,
        prefix: str = "",
        names: Optional[Dict[str, str]] = None,
        max_pending_frames: int = MAX_PENDING_FRAMES,
    ) -> None:
        self.upload = upload
        self.prefix = f"{prefix}/" if prefix else ""
        self.names = names or {}
        self.max_pending_frames = max_pending_frames
        self.dropped = 0  # frames skipped because the worker was behind
        self._pending_frames = 0
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        # Only touched by submit, i.e. by the inference thread
        self._last_box: Dict[str, float] = {}  # cam_id -> monotonic time of the last box
        self._active: Dict[str, Dict[str, Any]] = {}  # cam_id -> ongoing sequence
        self._started: Dict[str, Tuple[date, int]] = {}  # cam_id -> (UTC day, sequences started that day)
        threading.Thread(target=self._worker, name="sequence-capture", daemon=True).start()

    def submit(self, cam_id: str, frame: Image.Image, preds: np.ndarray) -> None:
        """Feed every scored frame of a pose, with the boxes the engine output (normalized to `frame`)."""
        boxes = np.reshape(preds, (-1, 5))
        boxes = boxes[boxes[:, 4] >= MIN_CONF]
        now = time.monotonic()
        ts = datetime.now(timezone.utc)
        seq = self._active.get(cam_id)
        if seq is None and len(boxes) and now - self._last_box.get(cam_id, -IDLE_S) >= IDLE_S:
            if self._take_slot(cam_id, ts.date()):
                seq = self._active[cam_id] = self._new_sequence(cam_id, ts)
            else:
                logger.debug("[%s] Sequence capture skipped, daily quota used", cam_id)
        if len(boxes):
            self._last_box[cam_id] = now
        if seq is None:
            return

        entry: Dict[str, Any] = {
            "ts": ts.isoformat(),
            "key": f"{seq['base']}/{len(seq['frames']):02d}.jpg",
            "boxes": [[round(float(v), 4) for v in box] for box in boxes],
        }
        seq["frames"].append(entry)
        with self._lock:
            full = self._pending_frames >= self.max_pending_frames
            if not full:
                self._pending_frames += 1
        if full:
            self.dropped += 1
            entry["key"] = None
            logger.debug("[%s] Sequence frame skipped, worker busy (%d skipped)", cam_id, self.dropped)
        else:
            self._queue.put_nowait((entry, frame))
        if len(seq["frames"]) >= FRAMES:
            del self._active[cam_id]
            self._queue.put_nowait((seq, None))  # after its frames, so their upload status is final

    def _new_sequence(self, cam_id: str, ts: datetime) -> Dict[str, Any]:
        ip, _, pose = cam_id.partition("_")
        name = self.names.get(ip, ip)
        base = f"{self.prefix}sequences/{name}/{ts:%Y-%m-%d}/{ts:%Y%m%dT%H%M%S}Z" + (f"_pose{pose}" if pose else "")
        logger.info("[%s] Sequence capture started: %s", cam_id, base)
        return {"camera": name, "cam_id": cam_id, "base": base, "frames": []}

    def _take_slot(self, cam_id: str, day: date) -> bool:
        """Count one sequence for this camera pose, unless it already used its daily quota."""
        last_day, count = self._started.get(cam_id, (day, 0))
        if last_day != day:
            count = 0
        if count >= MAX_SEQUENCES_PER_DAY:
            return False
        self._started[cam_id] = (day, count + 1)
        return True

    def _worker(self) -> None:
        while True:
            item, frame = self._queue.get()
            try:
                if frame is None:  # a finished sequence: its manifest
                    self.upload(f"{item['base']}/manifest.json", json.dumps(item).encode(), "application/json")
                else:
                    self.upload(item["key"], _jpeg(frame.convert("RGB")), "image/jpeg")
            except Exception as e:  # ruff: ignore[blind-except]
                logger.warning("Sequence capture upload failed: %s", e)
                if frame is not None:
                    item["key"] = None
            finally:
                if frame is not None:
                    with self._lock:
                        self._pending_frames -= 1
                self._queue.task_done()
