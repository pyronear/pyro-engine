# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Temporal validation service.

The engine submits a window of frames + YOLO boxes for one camera pose
(``POST /jobs``) and reads the verdict on its next round (``GET /jobs/{id}``).
Scoring runs in a worker thread so requests return immediately. The model is
the torch-free ``model_onnx.zip`` runtime of pyronear/temporal-model.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from temporal_model.core.onnx_model import OnnxTemporalModel

from pyro_temporal_api.jobs import JobStore, QueueFullError

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.environ.get("TEMPORAL_MODEL_PATH", "data/model_onnx.zip"))
MAX_FRAMES = 20
# Inference frames are a few hundred KB; anything larger is not an engine frame.
MAX_FRAME_BYTES = 4 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = OnnxTemporalModel.from_package(MODEL_PATH)
    app.state.jobs = JobStore(model)
    app.state.model_version = model_version(MODEL_PATH)
    logger.info("Loaded temporal model %s (version %s)", MODEL_PATH, app.state.model_version)
    yield


def model_version(path: Path) -> str | None:
    import zipfile

    import yaml

    with zipfile.ZipFile(path) as zf:
        return yaml.safe_load(zf.read("manifest.yaml")).get("model_version")


app = FastAPI(title="Pyro Temporal API", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_version": getattr(request.app.state, "model_version", None)}


@app.post("/jobs", status_code=202)
async def submit_job(
    request: Request,
    cam_id: str = Form(...),
    boxes: str = Form(..., description="JSON list, one entry per frame: [[x1, y1, x2, y2, conf], ...]"),
    frames: list[UploadFile] = File(..., description="JPEG frames, oldest first; filename stem = frame id"),
):
    try:
        parsed = json.loads(boxes)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"boxes is not valid JSON: {exc}") from exc
    if not isinstance(parsed, list) or not all(isinstance(fb, list) for fb in parsed):
        raise HTTPException(status_code=422, detail="boxes must be a list of per-frame box lists")
    if not 1 <= len(frames) <= MAX_FRAMES:
        raise HTTPException(status_code=422, detail=f"expected 1 to {MAX_FRAMES} frames, got {len(frames)}")
    if len(parsed) != len(frames):
        raise HTTPException(status_code=422, detail=f"{len(frames)} frames but {len(parsed)} box lists")
    for fb in parsed:
        if any(not isinstance(b, list) or len(b) != 5 for b in fb):
            raise HTTPException(status_code=422, detail="each box must be [x1, y1, x2, y2, conf]")

    payload = []
    for f in frames:
        data = await f.read(MAX_FRAME_BYTES + 1)
        if len(data) > MAX_FRAME_BYTES:
            raise HTTPException(status_code=413, detail=f"frame {f.filename} exceeds {MAX_FRAME_BYTES} bytes")
        payload.append((Path(f.filename or "frame").stem, data))
    try:
        job = request.app.state.jobs.submit(cam_id, payload, parsed)
    except QueueFullError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"job_id": job.job_id}


@app.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str):
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job.to_dict()
