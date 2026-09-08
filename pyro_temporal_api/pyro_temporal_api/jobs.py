# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Validation jobs: one submitted window of frames + boxes, scored by the temporal model in a worker thread."""

from __future__ import annotations

import logging
import queue
import tempfile
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from temporal_model.core import Detection, FrameDetections

logger = logging.getLogger(__name__)

MAX_JOBS_KEPT = 100


@dataclass
class Job:
    job_id: str
    cam_id: str
    frames: list[tuple[str, bytes]]  # (frame_id, jpeg bytes), oldest first
    boxes: list[list[list[float]]]  # per frame: [[x1, y1, x2, y2, conf], ...] normalized xyxy
    status: str = "pending"  # pending | done | error
    verdict: dict[str, Any] | None = None
    error: str | None = None
    _done: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "cam_id": self.cam_id,
            "status": self.status,
            "verdict": self.verdict,
            "error": self.error,
        }


def _to_frame_detections(frames: list[Any], boxes: list[list[list[float]]]) -> dict[str, FrameDetections]:
    """Engine boxes (normalized xyxy + conf) -> the temporal model's per-frame detections."""
    out = {}
    for idx, (frame, frame_boxes) in enumerate(zip(frames, boxes, strict=True)):
        out[frame.frame_id] = FrameDetections(
            frame_idx=idx,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            detections=[
                Detection(
                    class_id=0,
                    cx=(x1 + x2) / 2,
                    cy=(y1 + y2) / 2,
                    w=x2 - x1,
                    h=y2 - y1,
                    confidence=float(conf),
                )
                for x1, y1, x2, y2, conf in frame_boxes
            ],
        )
    return out


def score(model: Any, job: Job) -> dict[str, Any]:
    """Run the temporal model on a job's window; frames live on disk only for the call."""
    with tempfile.TemporaryDirectory(prefix="temporal_job_") as td:
        paths = []
        for frame_id, data in job.frames:
            p = Path(td) / f"{frame_id}.jpg"
            p.write_bytes(data)
            paths.append(p)
        frames = model.load_sequence(paths)
        out = model.predict(frames, frame_detections=_to_frame_detections(frames, job.boxes))
    kept = out.details["tubes"]["kept"]
    probs = [t["probability"] for t in kept if t["probability"] is not None]
    return {
        "is_positive": bool(out.is_positive),
        "probability": max(probs) if probs else None,
        "n_tubes": len(kept),
    }


class JobStore:
    """FIFO of jobs scored one at a time by a daemon thread; keeps the last MAX_JOBS_KEPT results."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="temporal-worker", daemon=True)
        self._worker.start()

    def submit(self, cam_id: str, frames: list[tuple[str, bytes]], boxes: list[list[list[float]]]) -> Job:
        job = Job(job_id=uuid.uuid4().hex, cam_id=cam_id, frames=frames, boxes=boxes)
        with self._lock:
            self._jobs[job.job_id] = job
            while len(self._jobs) > MAX_JOBS_KEPT:
                self._jobs.popitem(last=False)
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def wait(self, job_id: str, timeout: float) -> bool:
        """Block until the job finishes (tests and synchronous callers)."""
        job = self.get(job_id)
        return job is not None and job._done.wait(timeout)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                job.verdict = score(self._model, job)
                job.status = "done"
                logger.info("job %s cam %s: %s", job.job_id, job.cam_id, job.verdict)
            except Exception as exc:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                logger.exception("job %s cam %s failed", job.job_id, job.cam_id)
            finally:
                job.frames = []  # release the JPEGs once scored
                job._done.set()
