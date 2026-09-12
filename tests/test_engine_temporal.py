# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from unittest.mock import MagicMock

import numpy as np
import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError

from pyroengine.engine import Engine

CAM = "192.168.1.10_1"
# fake_pred is (cx, cy, w, h, conf) in model pixel space (imgsz 1024);
# post_process turns it into normalized xyxy [0.4, 0.4, 0.5, 0.5, 0.6]
PRED = np.array([[460.8, 460.8, 102.4, 102.4, 0.6]])


class FakeTemporal:
    """Scripted temporal service: `results` are returned in order by result(); submit() records windows."""

    def __init__(self, results=()) -> None:
        self.results = list(results)
        self.submitted = []
        self.n = 0

    def submit(self, cam_id, window):
        self.n += 1
        self.submitted.append((cam_id, list(window)))
        return f"job{self.n}"

    def result(self, _job_id):
        return self.results.pop(0)


def _engine(tmp_path, temporal, **kwargs):
    engine = Engine(
        cache_folder=str(tmp_path),
        conf_thresh=0.1,
        nb_consecutive_frames=2,
        temporal_api_url="http://localhost:8082",
        cam_creds={CAM: ("token", 7)},
        **kwargs,
    )
    engine.temporal = temporal
    engine.api_client = {"192.168.1.10": MagicMock()}  # no api_url: a real Client would call the API
    return engine


def _fire(engine, image, n=1):
    for _ in range(n):
        engine.predict(image, CAM, fake_pred=PRED.T.copy())
    return engine._states[CAM]


def test_window_collects_frames_and_boxes(tmp_path, mock_wildfire_image):
    engine = _engine(tmp_path, FakeTemporal([{"status": "pending"}] * 3), temporal_window=3)
    state = _fire(engine, mock_wildfire_image, n=4)
    window = list(state["temporal_frames"])
    assert len(window) == 3  # bounded ring buffer
    frame_id, jpeg, boxes = window[-1]
    assert frame_id.startswith(f"{CAM}_20")
    assert jpeg[:2] == b"\xff\xd8"  # JPEG magic
    assert boxes == [[0.4, 0.4, 0.5, 0.5, 0.6]]


def test_alert_held_until_positive_verdict(tmp_path, mock_wildfire_image):
    done = lambda positive: {"status": "done", "verdict": {"is_positive": positive, "probability": 0.5, "n_tubes": 1}}
    temporal = FakeTemporal([{"status": "pending"}, done(False), done(True)])
    engine = _engine(tmp_path, temporal)

    state = _fire(engine, mock_wildfire_image)  # ongoing on the first frame -> submit
    assert state["ongoing"]
    assert len(temporal.submitted) == 1
    assert state["temporal_job"] == "job1"
    assert len(engine._alerts) == 0

    _fire(engine, mock_wildfire_image)  # pending -> held
    assert len(engine._alerts) == 0
    assert state["temporal_job"] == "job1"

    _fire(engine, mock_wildfire_image)  # negative -> resubmitted with a longer window
    assert len(engine._alerts) == 0
    assert state["temporal_job"] == "job2"
    assert len(temporal.submitted[1][1]) == 3

    _fire(engine, mock_wildfire_image)  # positive -> validated, window staged
    assert state["temporal_validated"] is True
    assert state["temporal_job"] is None
    assert len(engine._alerts) > 0
    assert not temporal.results  # every scripted verdict was consumed

    n_staged = len(engine._alerts)
    _fire(engine, mock_wildfire_image)  # validated event: no more calls, keeps staging
    assert len(temporal.submitted) == 2
    assert len(engine._alerts) > n_staged


@pytest.mark.parametrize(
    "temporal",
    [
        FakeTemporal([{"status": "error", "verdict": None, "error": "RuntimeError: boom"}]),
        MagicMock(submit=MagicMock(side_effect=RequestsConnectionError("down"))),
    ],
    ids=["job_error", "service_down"],
)
def test_service_failure_fails_open(tmp_path, mock_wildfire_image, temporal):
    engine = _engine(tmp_path, temporal)
    state = _fire(engine, mock_wildfire_image)
    if isinstance(temporal, FakeTemporal):
        assert len(engine._alerts) == 0  # round N: submitted, held
        _fire(engine, mock_wildfire_image)  # round N+1: job error
    assert state["temporal_validated"] is True
    assert len(engine._alerts) > 0


def test_end_of_event_resets_verdict_but_keeps_window(tmp_path, mock_wildfire_image, mock_forest_image):
    temporal = FakeTemporal([{"status": "done", "verdict": {"is_positive": True, "probability": 0.9, "n_tubes": 1}}])
    engine = _engine(tmp_path, temporal)
    state = _fire(engine, mock_wildfire_image, n=3)
    assert state["temporal_validated"] is True
    n_frames = len(state["temporal_frames"])

    for _ in range(3):
        engine.predict(mock_forest_image, CAM, fake_pred=np.empty((5, 0)))
    assert state["ongoing"] is False
    assert state["temporal_validated"] is False
    assert state["temporal_job"] is None
    assert len(state["temporal_frames"]) >= n_frames


def test_disabled_without_url(tmp_path, mock_wildfire_image):
    engine = Engine(cache_folder=str(tmp_path), conf_thresh=0.1, nb_consecutive_frames=2)
    assert engine.temporal is None
    state = _fire(engine, mock_wildfire_image, n=2)
    assert state["ongoing"]
    assert len(state["temporal_frames"]) == 0


EMPTY = np.empty((5, 0))
DONE = lambda positive: {"status": "done", "verdict": {"is_positive": positive, "probability": 0.5, "n_tubes": 1}}


def test_event_ending_during_validation_still_uploads_on_positive(tmp_path, mock_wildfire_image, mock_forest_image):
    temporal = FakeTemporal([{"status": "pending"}, {"status": "pending"}, DONE(True)])
    engine = _engine(tmp_path, temporal)
    state = _fire(engine, mock_wildfire_image)  # submitted, held
    engine.predict(mock_forest_image, CAM, fake_pred=EMPTY)  # still ongoing (hysteresis), pending
    engine.predict(mock_forest_image, CAM, fake_pred=EMPTY)  # event over, verdict still pending: kept
    assert state["ongoing"] is False
    assert state["temporal_job"] == "job1"
    assert len(engine._alerts) == 0

    engine.predict(mock_forest_image, CAM, fake_pred=EMPTY)  # positive verdict: held frames go out
    assert len(engine._alerts) > 0
    assert state["temporal_job"] is None
    assert state["temporal_validated"] is False  # event closed after staging
    assert not temporal.results


def test_event_ending_during_validation_dropped_on_negative(tmp_path, mock_wildfire_image, mock_forest_image):
    temporal = FakeTemporal([{"status": "pending"}, {"status": "pending"}, DONE(False)])
    engine = _engine(tmp_path, temporal)
    state = _fire(engine, mock_wildfire_image)
    for _ in range(3):
        engine.predict(mock_forest_image, CAM, fake_pred=EMPTY)
    assert state["ongoing"] is False
    assert state["temporal_job"] is None  # no resubmission for an ended event
    assert len(temporal.submitted) == 1
    assert len(engine._alerts) == 0
