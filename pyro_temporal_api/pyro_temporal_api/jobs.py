# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Validation jobs: one submitted window of frames + boxes, scored by the temporal model in a worker thread."""

from __future__ import annotations

import logging
import queue
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from temporal_model.core import Detection, FrameDetections

logger = logging.getLogger(__name__)

MAX_JOBS_KEPT = 100
# One job per camera pose per round is the normal load; more means a stuck worker or a flood.
MAX_PENDING = 16
# A job scores in seconds; one running longer than this means the worker is stuck (health goes red).
STUCK_AFTER = 120.0


class QueueFullError(RuntimeError):
    """Raised by submit() when too many jobs are still waiting to be scored."""


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


def _tube_summary(tube: dict[str, Any]) -> dict[str, Any] | None:
    """``{"probability", "box"}`` of a scored tube, the box being the union of its detections (normalized xyxy)."""
    boxes = [e["bbox"] for e in tube["entries"] if e.get("bbox") is not None]
    if tube.get("probability") is None or not boxes:
        return None
    x1 = min(cx - w / 2 for cx, _, w, _ in boxes)
    y1 = min(cy - h / 2 for _, cy, _, h in boxes)
    x2 = max(cx + w / 2 for cx, _, w, _ in boxes)
    y2 = max(cy + h / 2 for _, cy, _, h in boxes)
    return {"probability": round(float(tube["probability"]), 4), "box": [round(v, 4) for v in (x1, y1, x2, y2)]}


def score(model: Any, job: Job) -> dict[str, Any]:
    """Run the temporal model on a job's window; frames live on disk only for the call."""
    hits, misses = getattr(model, "cache_hits", 0), getattr(model, "cache_misses", 0)
    with tempfile.TemporaryDirectory(prefix="temporal_job_") as td:
        paths = []
        for frame_id, data in job.frames:
            p = Path(td) / f"{frame_id}.jpg"
            p.write_bytes(data)
            paths.append(p)
        frames = model.load_sequence(paths)
        out = model.predict(frames, frame_detections=_to_frame_detections(frames, job.boxes))
    kept = out.details["tubes"]["kept"]
    tubes: list[dict[str, Any]] = []
    for t in kept:
        summary = _tube_summary(t)
        if summary is not None:
            tubes.append(summary)
    probs = [t["probability"] for t in tubes]
    return {
        "is_positive": bool(out.is_positive),
        "probability": max(probs) if probs else None,
        "n_tubes": len(kept),
        # One entry per scored tube: the engine attaches each probability to the boxes it covers.
        "tubes": tubes,
        # ViT slots served from the embedding cache vs embedded for this job (0/0 without a cache)
        "embeddings_cached": getattr(model, "cache_hits", 0) - hits,
        "embeddings_new": getattr(model, "cache_misses", 0) - misses,
    }


class JobStore:
    """FIFO of jobs scored one at a time by a daemon thread; keeps the last MAX_JOBS_KEPT results."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        self._running_since: float | None = None
        self._worker = threading.Thread(target=self._run, name="temporal-worker", daemon=True)
        self._worker.start()

    def submit(self, cam_id: str, frames: list[tuple[str, bytes]], boxes: list[list[list[float]]]) -> Job:
        if self._queue.qsize() >= MAX_PENDING:
            raise QueueFullError(f"{MAX_PENDING} jobs already pending")
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

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        """The job as returned by the API, read under the lock the worker publishes results with."""
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job is not None else None

    def healthy(self) -> bool:
        """False when the worker thread died or has been scoring one job for longer than STUCK_AFTER."""
        if not self._worker.is_alive():
            return False
        since = self._running_since
        return since is None or time.monotonic() - since < STUCK_AFTER

    def wait(self, job_id: str, timeout: float) -> bool:
        """Block until the job finishes (tests and synchronous callers)."""
        job = self.get(job_id)
        return job is not None and job._done.wait(timeout)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            self._running_since = time.monotonic()
            try:
                verdict = score(self._model, job)
                with self._lock:  # publish verdict and status together: no half-updated reads
                    job.verdict, job.status = verdict, "done"
                logger.info("job %s cam %s: %s", job.job_id, job.cam_id, verdict)
            except Exception as exc:
                with self._lock:
                    job.error, job.status = f"{type(exc).__name__}: {exc}", "error"
                logger.exception("job %s cam %s failed", job.job_id, job.cam_id)
            finally:
                self._running_since = None
                job.frames = []  # release the JPEGs once scored
                job._done.set()
