# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import io
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pyro_temporal_api import jobs, main
from pyro_temporal_api.jobs import JobStore, _to_frame_detections


class FakeModel:
    """Stands in for OnnxTemporalModel: records what it saw, answers from `verdicts`."""

    def __init__(self, positive=True, probability=0.8, fail=False):
        self.positive, self.probability, self.fail = positive, probability, fail
        self.calls = []

    def load_sequence(self, paths):
        return [SimpleNamespace(frame_id=p.stem, image_path=p, timestamp=None) for p in paths]

    def predict(self, frames, frame_detections):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append((frames, frame_detections))
        kept = [{"logit": 1.0, "probability": self.probability}] if frame_detections else []
        return SimpleNamespace(is_positive=self.positive, details={"tubes": {"kept": kept}})


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "gray").save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client():
    # No `with`: the lifespan (which loads the real model) must not run in tests.
    model = FakeModel()
    main.app.state.jobs = JobStore(model)
    main.app.state.model_version = "test"
    return TestClient(main.app), model


def _submit(client, n_frames=3, boxes=None, cam_id="192.168.1.10_1"):
    files = [("frames", (f"{cam_id}_2026-09-08T10-00-0{i}.jpg", _jpeg(), "image/jpeg")) for i in range(n_frames)]
    if boxes is None:
        boxes = [[[0.4, 0.4, 0.5, 0.5, 0.3]]] * n_frames
    return client.post("/jobs", data={"cam_id": cam_id, "boxes": json.dumps(boxes)}, files=files)


def test_health(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "ok", "model_version": "test"}


def test_job_lifecycle_and_conversion(client):
    c, model = client
    r = _submit(c, n_frames=3)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert c.app.state.jobs.wait(job_id, timeout=5)

    body = c.get(f"/jobs/{job_id}").json()
    assert body["status"] == "done"
    assert body["cam_id"] == "192.168.1.10_1"
    assert body["verdict"] == {"is_positive": True, "probability": 0.8, "n_tubes": 1}

    frames, dets = model.calls[0]
    assert [f.frame_id for f in frames] == [f"192.168.1.10_1_2026-09-08T10-00-0{i}" for i in range(3)]
    d = dets[frames[0].frame_id].detections[0]
    assert (d.cx, d.cy, d.w, d.h, d.confidence) == pytest.approx((0.45, 0.45, 0.1, 0.1, 0.3))
    assert dets[frames[2].frame_id].frame_idx == 2


def test_model_failure_is_reported_as_error(client):
    c, model = client
    model.fail = True
    job_id = _submit(c).json()["job_id"]
    assert c.app.state.jobs.wait(job_id, timeout=5)
    body = c.get(f"/jobs/{job_id}").json()
    assert body["status"] == "error"
    assert "RuntimeError" in body["error"]
    assert body["verdict"] is None


def test_unknown_job_is_404(client):
    c, _ = client
    assert c.get("/jobs/nope").status_code == 404


@pytest.mark.parametrize(
    ("kwargs", "detail"),
    [
        ({"n_frames": 2, "boxes": [[]]}, "box lists"),
        ({"n_frames": 1, "boxes": [[[0.1, 0.1, 0.2]]]}, "each box"),
        ({"n_frames": 21}, "expected 1 to 20"),
    ],
)
def test_rejects_malformed_submissions(client, kwargs, detail):
    c, _ = client
    r = _submit(c, **kwargs)
    assert r.status_code == 422
    assert detail in r.json()["detail"]


def test_frames_are_released_after_scoring(client):
    c, _ = client
    job_id = _submit(c).json()["job_id"]
    assert c.app.state.jobs.wait(job_id, timeout=5)
    assert c.app.state.jobs.get(job_id).frames == []


def test_to_frame_detections_keeps_empty_frames():
    frames = [SimpleNamespace(frame_id="a", timestamp=None), SimpleNamespace(frame_id="b", timestamp=None)]
    dets = _to_frame_detections(frames, [[], [[0, 0, 1, 1, 0.5]]])
    assert dets["a"].detections == []
    assert len(dets["b"].detections) == 1


def test_oversized_frame_is_rejected(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(main, "MAX_FRAME_BYTES", 100)
    r = _submit(c, n_frames=1)
    assert r.status_code == 413


def test_full_queue_returns_503(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(jobs, "MAX_PENDING", 0)
    r = _submit(c, n_frames=1)
    assert r.status_code == 503
    assert "pending" in r.json()["detail"]
