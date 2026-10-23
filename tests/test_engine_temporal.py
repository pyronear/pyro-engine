# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from unittest.mock import MagicMock

import numpy as np
import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError

from pyroengine.engine import Engine

CAM = "192.168.1.10_1"
# fake_pred is (cx, cy, w, h, conf) in model pixel space (imgsz 1024);
# post_process turns it into normalized xyxy [0.4, 0.4, 0.5, 0.5, 0.6]
PRED = np.array([[460.8, 460.8, 102.4, 102.4, 0.6]])
EMPTY = np.empty((5, 0))
BOX = [0.4, 0.4, 0.5, 0.5]
FAR_BOX = [0.8, 0.8, 0.9, 0.9]


def done(*tubes):
    """A done verdict with one (box, probability) tube each."""
    tubes = [{"box": box, "probability": prob} for box, prob in tubes]
    probs = [t["probability"] for t in tubes]
    return {
        "status": "done",
        "verdict": {
            "is_positive": any(p > 0.5 for p in probs),
            "probability": max(probs, default=None),
            "n_tubes": len(tubes),
            "tubes": tubes,
        },
    }


PENDING = {"status": "pending"}


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


def _scores(engine):
    return [a["temporal_scores"] for a in engine._alerts]


def test_window_is_the_prediction_window(tmp_path, mock_wildfire_image):
    engine = _engine(tmp_path, FakeTemporal([PENDING] * 3), temporal_window=3)
    state = _fire(engine, mock_wildfire_image, n=4)
    assert state["last_predictions"].maxlen == 3  # widened from nb_consecutive_frames=2
    window = engine._temporal_window(CAM)
    assert len(window) == 3
    frame_id, jpeg, boxes = window[-1]
    assert frame_id.startswith(f"{CAM}_20")
    assert len({f for f, _, _ in window}) == 3  # ids unique at microsecond precision
    assert jpeg[:2] == b"\xff\xd8"  # JPEG magic
    assert boxes == [[0.4, 0.4, 0.5, 0.5, 0.6]]
    # resubmission keeps the ids of frames already sent (embedding cache hits on the service)
    assert [f for f, _, _ in engine._temporal_window(CAM)] == [f for f, _, _ in window]


def test_frames_scored_every_round_and_sent_with_tube_scores(tmp_path, mock_wildfire_image):
    temporal = FakeTemporal([PENDING, done((BOX, 0.05)), done((BOX, 0.3)), done((BOX, 0.9))])
    engine = _engine(tmp_path, temporal)

    state = _fire(engine, mock_wildfire_image)  # ongoing on the first frame -> submit
    assert state["ongoing"]
    assert len(temporal.submitted) == 1
    assert state["temporal_job"][0] == "job1"
    assert len(engine._alerts) == 0

    _fire(engine, mock_wildfire_image)  # pending -> held, no new job
    assert len(engine._alerts) == 0
    assert state["temporal_job"][0] == "job1"

    _fire(engine, mock_wildfire_image)  # below temporal_min_score -> nothing staged, resubmitted
    assert len(engine._alerts) == 0
    assert state["temporal_job"][0] == "job2"
    assert len(temporal.submitted[1][1]) == 3

    _fire(engine, mock_wildfire_image)  # 0.3 >= 0.1 -> the 3 frames job2 scored go out, the 4th waits
    assert len(engine._alerts) == 3
    assert all(s == [0.3] for s in _scores(engine))
    assert state["temporal_job"][0] == "job3"

    _fire(engine, mock_wildfire_image)  # next verdict (job3 scored frames 1-4): frame 4 goes out, frame 5 waits
    assert len(engine._alerts) == 4
    assert _scores(engine)[-1] == [0.9]
    assert len(temporal.submitted) == 4  # one job per round while ongoing
    assert not temporal.results


def test_only_frames_of_the_scored_job_are_staged(tmp_path, mock_wildfire_image):
    """Regression: with a full window, a verdict must not stage the frame captured after its job
    (unscored) nor lose the oldest scored frame."""
    temporal = FakeTemporal([PENDING, done((BOX, 0.5)), done((BOX, 0.6))])
    engine = _engine(tmp_path, temporal, temporal_window=3)
    ts = []
    for _ in range(4):
        state = _fire(engine, mock_wildfire_image)
        ts.append(state["last_predictions"][-1][3])
    # job1 = {f1}, job2 = {f1, f2, f3}; f4 arrived after job2 and f1 left the window before its verdict
    assert [a["ts"] for a in engine._alerts] == ts[:3]
    assert _scores(engine) == [[0.5], [0.6], [0.6]]
    assert state["last_predictions"][-1][4] is False  # f4 waits for job3


def test_boxes_outside_scored_tubes_are_dropped(tmp_path, mock_wildfire_image):
    engine = _engine(tmp_path, FakeTemporal([done((FAR_BOX, 0.9))] * 2))
    _fire(engine, mock_wildfire_image, n=3)
    assert len(engine._alerts) == 2  # frames still go out (continuity) ...
    assert all(a["bboxes"] == [] and a["temporal_scores"] == [] for a in engine._alerts)  # ... without the box

    engine = _engine(tmp_path, FakeTemporal([done((FAR_BOX, 0.9), (BOX, 0.2))] * 2))
    _fire(engine, mock_wildfire_image, n=3)
    assert [[b[:4] for b in a["bboxes"]] for a in engine._alerts] == [[BOX]] * 2
    assert _scores(engine) == [[0.2], [0.2]]


@pytest.mark.parametrize(
    "temporal",
    [
        FakeTemporal([{"status": "error", "verdict": None, "error": "RuntimeError: boom"}]),
        MagicMock(submit=MagicMock(side_effect=RequestsConnectionError("down"))),
    ],
    ids=["job_error", "service_down"],
)
def test_service_failure_fails_open_without_scores(tmp_path, mock_wildfire_image, temporal):
    engine = _engine(tmp_path, temporal)
    state = _fire(engine, mock_wildfire_image)
    if isinstance(temporal, FakeTemporal):
        assert len(engine._alerts) == 0  # round N: submitted, held
        _fire(engine, mock_wildfire_image)  # round N+1: job error
    assert state["temporal_bypass"] is True
    assert len(engine._alerts) > 0
    assert all(a["bboxes"] and a["temporal_scores"] is None for a in engine._alerts)


def test_end_of_event_resets_verdict_but_keeps_window(tmp_path, mock_wildfire_image, mock_forest_image):
    temporal = FakeTemporal([done((BOX, 0.9))] * 6)
    engine = _engine(tmp_path, temporal)
    state = _fire(engine, mock_wildfire_image, n=3)
    assert state["temporal_tubes"]
    n_frames = len(state["last_predictions"])

    for _ in range(3):
        engine.predict(mock_forest_image, CAM, fake_pred=EMPTY)
    assert state["ongoing"] is False
    assert state["temporal_tubes"] == []
    assert state["temporal_bypass"] is False
    assert state["temporal_job"] is None
    assert len(state["last_predictions"]) >= n_frames


def test_disabled_without_url(tmp_path, mock_wildfire_image):
    engine = Engine(cache_folder=str(tmp_path), conf_thresh=0.1, nb_consecutive_frames=2, cam_creds={CAM: ("token", 7)})
    engine.api_client = {"192.168.1.10": MagicMock()}
    assert engine.temporal is None
    state = _fire(engine, mock_wildfire_image, n=2)
    assert state["ongoing"]
    assert state["last_predictions"].maxlen == 2  # window untouched
    assert len(engine._alerts) > 0  # staged on the spot, no gate
    assert all(a["temporal_scores"] is None for a in engine._alerts)


def test_lost_job_fails_open_and_event_can_end(tmp_path, mock_wildfire_image, mock_forest_image):
    """Regression: a 404 (service restarted) used to leave temporal_job set, so the event never ended
    and every later frame of the pose was staged as an alert."""
    temporal = FakeTemporal()
    temporal.result = MagicMock(side_effect=HTTPError("404 unknown job"))
    engine = _engine(tmp_path, temporal)
    state = _fire(engine, mock_wildfire_image)  # submitted
    _fire(engine, mock_wildfire_image)  # poll fails -> fail open
    assert state["temporal_bypass"] is True
    assert state["temporal_job"] is None
    assert len(engine._alerts) > 0

    for _ in range(3):
        engine.predict(mock_forest_image, CAM, fake_pred=EMPTY)
    assert state["ongoing"] is False
    assert state["temporal_bypass"] is False  # event ended and was reset
    n_alerts = len(engine._alerts)
    engine.predict(mock_forest_image, CAM, fake_pred=EMPTY)
    assert len(engine._alerts) == n_alerts  # nothing staged once the event is over


def test_job_pending_too_long_fails_open(tmp_path, mock_wildfire_image):
    temporal = FakeTemporal([PENDING])
    engine = _engine(tmp_path, temporal, temporal_job_timeout=0)
    state = _fire(engine, mock_wildfire_image)  # submitted
    _fire(engine, mock_wildfire_image)  # still pending past the deadline -> fail open
    assert state["temporal_bypass"] is True
    assert state["temporal_job"] is None
    assert len(engine._alerts) > 0


def test_event_ending_during_validation_still_uploads_on_positive(tmp_path, mock_wildfire_image, mock_forest_image):
    temporal = FakeTemporal([PENDING, PENDING, done((BOX, 0.7))])
    engine = _engine(tmp_path, temporal)
    state = _fire(engine, mock_wildfire_image)  # submitted, held
    engine.predict(mock_forest_image, CAM, fake_pred=EMPTY)  # still ongoing (hysteresis), pending
    engine.predict(mock_forest_image, CAM, fake_pred=EMPTY)  # event over, verdict still pending: kept
    assert state["ongoing"] is False
    assert state["temporal_job"][0] == "job1"
    assert len(engine._alerts) == 0

    engine.predict(mock_forest_image, CAM, fake_pred=EMPTY)  # verdict in: held frames go out, no new job
    assert len(engine._alerts) > 0
    assert _scores(engine)[0] == [0.7]
    assert state["temporal_job"] is None
    assert len(temporal.submitted) == 1
    assert state["temporal_tubes"] == []  # event closed after staging
    assert not temporal.results


def test_event_ending_during_validation_dropped_on_negative(tmp_path, mock_wildfire_image, mock_forest_image):
    temporal = FakeTemporal([PENDING, PENDING, done((BOX, 0.02))])
    engine = _engine(tmp_path, temporal)
    state = _fire(engine, mock_wildfire_image)
    for _ in range(3):
        engine.predict(mock_forest_image, CAM, fake_pred=EMPTY)
    assert state["ongoing"] is False
    assert state["temporal_job"] is None  # no resubmission for an ended event
    assert len(temporal.submitted) == 1
    assert len(engine._alerts) == 0
    rejected_ts = [e[3] for e in state["last_predictions"]]
    assert all(e[4] for e in state["last_predictions"])  # held frames consumed, never uploaded

    # Regression: a new event on the same pose must not upload the rejected frames under its own score
    temporal.results.extend([PENDING, done((BOX, 0.9))])
    _fire(engine, mock_wildfire_image, n=3)
    assert len(engine._alerts) > 0
    assert not {a["ts"] for a in engine._alerts} & set(rejected_ts)
