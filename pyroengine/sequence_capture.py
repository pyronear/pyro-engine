# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Temporary R&D: save the 4K frames of every detection sequence that can reach the temporal model to S3.

When a camera pose gets a box at MIN_CONF or more after IDLE_S without any, the next FRAMES frames of that
pose are written to a local folder as JPEG, the triggering one and the ones without boxes included. Once the
sequence is complete, it is uploaded with a manifest of the boxes the engine saw on each frame only when it
could have reached the temporal model: some `window` consecutive frames hold at least half as many boxes
(the engine's own count, see Predictor._update_states, without its overlap and score conditions, so a
sequence the crop refinement would raise is kept too). Otherwise it is deleted. Offline, each sequence can be replayed
through the whole pipeline (HD detection, crop refinement, temporal model) to measure which variant raises
an alert. Nothing here changes alerting: frames are encoded, written and uploaded in one background thread.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .crop_check import Uploader, _jpeg

__all__ = ["SequenceCapture", "can_reach_temporal"]

logger = logging.getLogger(__name__)

# Own top-level folder, apart from the crop check data whatever S3_PREFIX is
PREFIX = "sequence-capture"
MIN_CONF = 0.05  # the engine's own confidence threshold: any box it outputs starts a sequence
FRAMES = 20  # frames per sequence, the triggering one included
IDLE_S = 600.0  # a box starts a new sequence only after this long without any box on the pose
MAX_SEQUENCES_PER_DAY = 30  # uploads per camera pose, at most FRAMES x ~1.1 MB = ~23 MB each
# Frames waiting for the worker, the one being uploaded included; a 4K RGB frame holds 25 MB.
MAX_PENDING_FRAMES = 4
# Complete sequences waiting for upload on disk, ~23 MB each; newer ones are dropped beyond this.
MAX_PENDING_UPLOADS = 10


class SequenceCapture:
    """Tracks sequences on the inference thread, writes and uploads their frames in one background thread."""

    def __init__(
        self,
        upload: Uploader,
        local_dir: Path,
        names: Optional[Dict[str, str]] = None,
        window: int = 6,
        max_pending_frames: int = MAX_PENDING_FRAMES,
    ) -> None:
        self.upload = upload
        self.local_dir = local_dir
        shutil.rmtree(local_dir, ignore_errors=True)  # sequences left by a previous run are incomplete
        local_dir.mkdir(parents=True)
        self.names = names or {}
        self.window = window
        self.max_pending_frames = max_pending_frames
        self.dropped = 0  # frames skipped because the worker was behind
        self._pending_frames = 0
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        # Only touched by submit, i.e. by the inference thread
        self._last_box: Dict[str, float] = {}  # cam_id -> monotonic time of the last box
        self._active: Dict[str, Dict[str, Any]] = {}  # cam_id -> ongoing sequence
        # Complete sequences, uploaded in their own thread so a slow uplink never delays writing frames
        self._uploads: queue.Queue = queue.Queue()
        # Only touched by the upload thread
        self._uploaded: Dict[str, Tuple[date, int]] = {}  # cam_id -> (UTC day, sequences uploaded that day)
        threading.Thread(target=self._worker, name="sequence-capture", daemon=True).start()
        threading.Thread(target=self._uploader, name="sequence-upload", daemon=True).start()

    def submit(self, cam_id: str, frame: Image.Image, preds: np.ndarray) -> None:
        """Feed every scored frame of a pose, with the boxes the engine output (normalized to `frame`)."""
        boxes = np.reshape(preds, (-1, 5))
        boxes = boxes[boxes[:, 4] >= MIN_CONF]
        now = time.monotonic()
        ts = datetime.now(timezone.utc)
        seq = self._active.get(cam_id)
        if seq is None and len(boxes) and now - self._last_box.get(cam_id, -IDLE_S) >= IDLE_S:
            seq = self._active[cam_id] = self._new_sequence(cam_id, ts)
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
            self._queue.put_nowait((seq, None))  # after its frames, so they are all written

    def _new_sequence(self, cam_id: str, ts: datetime) -> Dict[str, Any]:
        ip, _, pose = cam_id.partition("_")
        name = self.names.get(ip, ip)
        base = f"{PREFIX}/{name}/{ts:%Y-%m-%d}/{ts:%Y%m%dT%H%M%S}Z" + (f"_pose{pose}" if pose else "")
        logger.debug("[%s] Sequence capture started: %s", cam_id, base)
        return {"camera": name, "cam_id": cam_id, "base": base, "frames": []}

    def _take_slot(self, cam_id: str, day: date) -> bool:
        """Count one upload for this camera pose, unless it already used its daily quota."""
        last_day, count = self._uploaded.get(cam_id, (day, 0))
        if last_day != day:
            count = 0
        if count >= MAX_SEQUENCES_PER_DAY:
            return False
        self._uploaded[cam_id] = (day, count + 1)
        return True

    def _finish(self, seq: Dict[str, Any]) -> None:
        """Upload a complete sequence when it could reach the temporal model, then delete its local frames."""
        try:
            if not self._take_slot(seq["cam_id"], datetime.now(timezone.utc).date()):
                logger.info("[%s] Sequence dropped, daily upload quota used: %s", seq["cam_id"], seq["base"])
            else:
                for f in seq["frames"]:
                    if f["key"] is None:
                        continue
                    try:
                        self.upload(f["key"], (self.local_dir / f["key"]).read_bytes(), "image/jpeg")
                    except Exception as e:  # ruff: ignore[blind-except]
                        logger.warning("Sequence frame upload failed: %s", e)
                        f["key"] = None
                self.upload(f"{seq['base']}/manifest.json", json.dumps(seq).encode(), "application/json")
                logger.info("[%s] Sequence uploaded: %s", seq["cam_id"], seq["base"])
        finally:
            shutil.rmtree(self.local_dir / seq["base"], ignore_errors=True)

    def _worker(self) -> None:
        while True:
            item, frame = self._queue.get()
            try:
                if frame is None and not can_reach_temporal([len(f["boxes"]) for f in item["frames"]], self.window):
                    logger.debug("[%s] Sequence dropped, too few boxes: %s", item["cam_id"], item["base"])
                    shutil.rmtree(self.local_dir / item["base"], ignore_errors=True)
                elif frame is None and self._uploads.qsize() >= MAX_PENDING_UPLOADS:
                    logger.warning("[%s] Sequence dropped, upload backlog full: %s", item["cam_id"], item["base"])
                    shutil.rmtree(self.local_dir / item["base"], ignore_errors=True)
                elif frame is None:
                    self._uploads.put_nowait(item)
                else:
                    path = self.local_dir / item["key"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(_jpeg(frame.convert("RGB")))
            except Exception as e:  # ruff: ignore[blind-except]
                logger.warning("Sequence capture failed: %s", e)
                if frame is not None:
                    item["key"] = None
            finally:
                if frame is not None:
                    with self._lock:
                        self._pending_frames -= 1
                self._queue.task_done()

    def _uploader(self) -> None:
        while True:
            seq = self._uploads.get()
            try:
                self._finish(seq)
            except Exception:
                logger.exception("[%s] Sequence upload failed", seq["cam_id"])
            finally:
                self._uploads.task_done()


def can_reach_temporal(box_counts: List[int], window: int) -> bool:
    """True when some `window` consecutive frames hold at least half as many boxes, like the engine requires."""
    need = (window + 1) // 2
    return any(sum(box_counts[i : i + window]) >= need for i in range(max(1, len(box_counts) - window + 1)))
