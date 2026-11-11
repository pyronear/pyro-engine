# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from itertools import count
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
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
_SEEDS = count()


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
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def windows(self):
        return [[f for f, _, _ in window] for _, window in self.submitted]


def _engine(tmp_path, temporal, **kwargs):
    # Scored from the first detection unless a test checks the pyro-api minimum (4 frames)
    kwargs.setdefault("temporal_min_frames", 1)
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


def _image():
    """A new capture: its JPEG differs from every other one."""
    noise = np.random.default_rng(next(_SEEDS)).integers(0, 256, (72, 128, 3), dtype=np.uint8)
    return Image.fromarray(noise)


def _fire(engine, n=1, pred=PRED, image=None):
    """`n` captures with a detection; returns their frame ids."""
    ids = []
    for _ in range(n):
        engine.predict(image or _image(), CAM, fake_pred=pred.T.copy())
        ids.append(_last_id(engine))
    return ids


def _quiet(engine, n=1):
    """`n` captures without a detection; returns their frame ids."""
    ids = []
    for _ in range(n):
        engine.predict(_image(), CAM, fake_pred=EMPTY)
        ids.append(_last_id(engine))
    return ids


def _last_id(engine):
    return engine._frame_id(CAM, engine._states[CAM]["last_predictions"][-1][3])


def _state(engine):
    return engine._states[CAM]


def _event(engine):
    return _state(engine)["temporal_events"][-1]


def _scores(engine):
    return [a["temporal_scores"] for a in engine._alerts]


def _sent(engine):
    return [engine._frame_id(CAM, a["ts"]) for a in engine._alerts]


def test_window_holds_the_detection_frames_with_boxes_from_0_1(tmp_path):
    """As pyro-api scores a sequence: frames without a detection stay out of the window (they carried
    weak boxes that inflated the score), and boxes below the service's detection threshold are dropped."""
    engine = _engine(tmp_path, FakeTemporal([PENDING] * 3))
    weak = np.array([[460.8, 460.8, 102.4, 102.4, 0.6], [870.4, 870.4, 102.4, 102.4, 0.07]])  # + FAR_BOX at 0.07
    lead_up = _quiet(engine)
    f1 = _fire(engine, pred=weak)
    gap = _quiet(engine)  # still ongoing (hysteresis), no detection
    assert _state(engine)["ongoing"]
    f3 = _fire(engine, pred=weak)
    event = _event(engine)
    assert [engine._frame_id(CAM, e[3]) for e in event["frames"]] == lead_up + f1 + gap + f3
    assert [f for f, _, _ in event["window"]] == f1 + f3
    assert all(jpeg[:2] == b"\xff\xd8" for _, jpeg, _ in event["window"])
    assert all(boxes == [[0.4, 0.4, 0.5, 0.5, 0.6]] for _, _, boxes in event["window"])
    assert _state(engine)["last_predictions"].maxlen == 2  # the detection window is not widened


def test_scored_from_the_4th_detection_then_the_full_window_verdict_holds(tmp_path):
    temporal = FakeTemporal([done((BOX, 0.9)), done((BOX, 0.9))])
    engine = _engine(tmp_path, temporal, temporal_min_frames=4, temporal_window=5)
    ids = _fire(engine, n=3)
    assert not temporal.submitted  # held until the 4th detection
    assert not engine._alerts
    ids += _fire(engine)
    assert temporal.windows() == [ids]
    ids += _fire(engine)  # job1 verdict: frames 1-4 go out, the grown window (5) is submitted
    assert _sent(engine) == ids[:4]
    ids += _fire(engine, n=2)  # job2 is on the full window: final, frames 5-7 go out under it
    assert temporal.windows() == [ids[:4], ids[:5]]
    assert _sent(engine) == ids
    assert _scores(engine) == [[0.9]] * 7
    assert all(e[0] is None and e[5] is None for e in _event(engine)["frames"])  # sent frames release their JPEG


def test_verdict_below_min_score_holds_frames_until_a_positive_one(tmp_path):
    temporal = FakeTemporal([PENDING, done((BOX, 0.2)), done((BOX, 0.3)), done((BOX, 0.9))])
    engine = _engine(tmp_path, temporal)
    ids = _fire(engine)  # submitted
    ids += _fire(engine)  # pending: held, no new job
    assert not engine._alerts
    assert len(temporal.submitted) == 1
    ids += _fire(engine)  # 0.2 is not above temporal_min_score: nothing staged, window 3 submitted
    assert not engine._alerts
    ids += _fire(engine)  # 0.3: the 3 frames job2 scored go out, the 4th waits
    assert _sent(engine) == ids[:3]
    assert _scores(engine) == [[0.3]] * 3
    _fire(engine)  # job3 scored frames 1-4: frame 4 goes out, frame 5 waits
    assert _sent(engine) == ids
    assert _scores(engine)[-1] == [0.9]
    assert temporal.windows() == [ids[:1], ids[:3], ids[:4], [*ids, _last_id(engine)]]


def test_final_negative_verdict_drops_the_event(tmp_path):
    temporal = FakeTemporal([done((BOX, 0.1)), done((BOX, 0.1))])
    engine = _engine(tmp_path, temporal, temporal_min_frames=4, temporal_window=5)
    _fire(engine, n=8)
    assert len(temporal.submitted) == 2  # 4 and 5 frames only
    assert not engine._alerts
    assert all(e[4] for e in _event(engine)["frames"])  # dropped, never uploaded
    _quiet(engine, n=2)
    assert _state(engine)["temporal_events"] == []


def test_repeated_jpeg_is_scored_once_and_still_sent(tmp_path):
    temporal = FakeTemporal([done((BOX, 0.8))])
    engine = _engine(tmp_path, temporal)
    image = _image()
    # The pose is read twice before the patrol refreshed it: the same JPEG comes back
    ids = _fire(engine, image=image)
    ids += _fire(engine, image=image)  # verdict on the first frame, the repeat does not grow the window
    assert temporal.windows() == [ids[:1]]
    ids += _quiet(engine, n=2)  # event over: its last verdict is final and covers the repeat
    assert _sent(engine) == ids[:3]
    assert _scores(engine) == [[0.8]] * 3


def test_boxes_outside_scored_tubes_are_dropped(tmp_path):
    engine = _engine(tmp_path, FakeTemporal([done((FAR_BOX, 0.9))] * 2))
    _fire(engine, n=3)
    assert len(engine._alerts) == 2  # frames still go out (continuity) ...
    assert all(a["bboxes"] == [] and a["temporal_scores"] == [] for a in engine._alerts)  # ... without the box

    engine = _engine(tmp_path, FakeTemporal([done((FAR_BOX, 0.9), (BOX, 0.3))] * 2))
    _fire(engine, n=3)
    assert [[b[:4] for b in a["bboxes"]] for a in engine._alerts] == [[BOX]] * 2
    assert _scores(engine) == [[0.3], [0.3]]

    # a tube merely touching the box does not lend it its score, a tube covering the box does
    touching = [0.48, 0.48, 0.9, 0.9]
    engine = _engine(tmp_path, FakeTemporal([done((touching, 0.9))] * 2))
    _fire(engine, n=3)
    assert all(a["bboxes"] == [] for a in engine._alerts)
    covering = [0.3, 0.3, 0.9, 0.9]
    engine = _engine(tmp_path, FakeTemporal([done((covering, 0.9))] * 2))
    _fire(engine, n=3)
    assert _scores(engine) == [[0.9], [0.9]]


@pytest.mark.parametrize(
    "temporal",
    [
        FakeTemporal([{"status": "error", "verdict": None, "error": "RuntimeError: boom"}]),
        MagicMock(submit=MagicMock(side_effect=RequestsConnectionError("down"))),
    ],
    ids=["job_error", "service_down"],
)
def test_service_failure_fails_open_without_scores(tmp_path, temporal):
    engine = _engine(tmp_path, temporal)
    _fire(engine)
    if isinstance(temporal, FakeTemporal):
        assert not engine._alerts  # round N: submitted, held
        temporal.submit = MagicMock(side_effect=RequestsConnectionError("down"))  # retries fail too
        _fire(engine)  # round N+1: job error
    assert _event(engine)["bypass"] is True
    assert engine._alerts
    assert all(a["bboxes"] and a["temporal_scores"] is None for a in engine._alerts)


def test_fail_open_retries_and_resumes_scoring(tmp_path):
    """Regression: one transient failure used to leave the rest of the event unscored, and pyro-api
    keeps the last device verdict without scoring the sequence itself."""
    temporal = FakeTemporal([RequestsConnectionError("down"), done((BOX, 0.9)), done((BOX, 0.9))])
    engine = _engine(tmp_path, temporal)
    _fire(engine)  # job1 submitted
    _fire(engine)  # job1 poll fails: fail open, frames go out unscored, job2 submitted on the same round
    assert _event(engine)["bypass"] is True
    assert _event(engine)["job"][0] == "job2"
    assert all(s is None for s in _scores(engine))
    _fire(engine)  # job2 verdict: scoring resumed, job3 submitted
    assert _event(engine)["bypass"] is False
    _fire(engine)  # job3 verdict scores the frame captured after job2
    assert _scores(engine)[-1] == [0.9]


def test_lost_job_fails_open_and_event_can_end(tmp_path):
    """Regression: a 404 (service restarted) used to leave the job set, so the event never ended
    and every later frame of the pose was staged as an alert."""
    temporal = FakeTemporal()
    temporal.result = MagicMock(side_effect=HTTPError("404 unknown job"))
    engine = _engine(tmp_path, temporal)
    _fire(engine)  # submitted
    _fire(engine)  # poll fails -> fail open
    assert _event(engine)["bypass"] is True
    assert engine._alerts
    _quiet(engine, n=3)
    assert not _state(engine)["ongoing"]
    assert _state(engine)["temporal_events"] == []  # event ended and resolved
    n_alerts = len(engine._alerts)
    _quiet(engine)
    assert len(engine._alerts) == n_alerts  # nothing staged once the event is over


def test_job_pending_too_long_fails_open(tmp_path):
    temporal = FakeTemporal([PENDING])
    engine = _engine(tmp_path, temporal, temporal_job_timeout=0)
    _fire(engine)  # submitted
    _fire(engine)  # still pending past the deadline -> fail open
    assert _event(engine)["bypass"] is True
    assert engine._alerts


def test_event_ending_while_its_verdict_is_pending_still_uploads_on_positive(tmp_path):
    temporal = FakeTemporal([PENDING, PENDING, done((BOX, 0.7))])
    engine = _engine(tmp_path, temporal)
    ids = _fire(engine)  # submitted, held
    ids += _quiet(engine)  # still ongoing (hysteresis), pending
    _quiet(engine)  # event over, verdict still pending: kept
    assert not _state(engine)["ongoing"]
    assert _event(engine)["ended"]
    assert not engine._alerts
    _quiet(engine)  # verdict in: final for the ended event, its frames go out, no new job
    assert _sent(engine) == ids
    assert _scores(engine)[0] == [0.7]
    assert len(temporal.submitted) == 1
    assert _state(engine)["temporal_events"] == []


def test_event_ending_while_its_verdict_is_pending_dropped_on_negative(tmp_path):
    temporal = FakeTemporal([PENDING, PENDING, done((BOX, 0.02))])
    engine = _engine(tmp_path, temporal)
    rejected = _fire(engine) + _quiet(engine)  # A: the detection and the hysteresis frame
    _quiet(engine, n=2)  # A over; the last quiet frame is a lead-up of whatever comes next
    assert not _state(engine)["ongoing"]
    assert len(temporal.submitted) == 1  # no resubmission, the ended event's window did not grow
    assert not engine._alerts
    assert _state(engine)["temporal_events"] == []  # dropped for good

    # A new event on the same pose must not upload the rejected frames under its own score
    temporal.results.extend([done((BOX, 0.9))] * 3)
    _fire(engine, n=3)
    assert engine._alerts
    assert not set(_sent(engine)) & set(rejected)


def test_pending_verdict_of_an_ended_event_does_not_validate_the_next(tmp_path):
    """Codex review: event B inherited A's detection count and was sent under A's verdict."""
    temporal = FakeTemporal([PENDING, PENDING, PENDING, done((BOX, 0.9)), PENDING])
    engine = _engine(tmp_path, temporal)
    a = _fire(engine) + _quiet(engine)  # A submits job1, pending
    _quiet(engine)  # A over, job1 pending
    b = _fire(engine)  # B starts: its own event, job2
    b += _fire(engine)  # job1 validates A; job2 pending
    events = _state(engine)["temporal_events"]
    assert len(events) == 1  # A resolved, B ongoing
    assert _sent(engine) == a
    assert temporal.windows() == [a[:1], b[:1]]
    assert all(not e[4] for e in events[0]["frames"])  # B waits for its own verdict


def test_frames_of_a_rejected_event_do_not_leak_into_the_next(tmp_path):
    """Codex review: a detection captured while the verdict was pending survived the rejection,
    counted in the next event's window and went out under its score."""
    temporal = FakeTemporal([PENDING, PENDING, done((BOX, 0.02)), done((BOX, 0.02))] + [done((BOX, 0.9))] * 4)
    engine = _engine(tmp_path, temporal)
    a = _fire(engine, n=2)  # job1 on A1; A2 captured while it is pending
    a += _quiet(engine)  # hysteresis frame, still A
    _quiet(engine)  # A over; job1 rejects A, its grown window (A1, A2) is still scored
    b = _fire(engine)  # A's last verdict rejects it for good; B submits its own window
    b += _fire(engine, n=3)  # B is validated
    assert temporal.windows()[:3] == [a[:1], a[:2], b[:1]]
    assert all(set(w) <= set(b) for w in temporal.windows()[3:])
    assert engine._alerts
    assert not set(_sent(engine)) & set(a)


def test_window_grown_during_a_slow_verdict_is_still_submitted(tmp_path):
    """Codex review: the job of detection 9 still pending at detection 10 made the engine skip the
    10-frame window and keep the 9-frame verdict."""
    temporal = FakeTemporal([PENDING, done((BOX, 0.1)), done((BOX, 0.9))])
    engine = _engine(tmp_path, temporal, temporal_window=3)
    ids = _fire(engine, n=4)  # job1 on 1 frame pending at frames 2 and 3, verdict at frame 4
    assert temporal.windows() == [ids[:1], ids[:3]]  # the full window is submitted after the late verdict
    _fire(engine)  # verdict on the full window: final
    assert len(engine._alerts) == 5


def test_carried_boxes_do_not_turn_a_quiet_frame_into_a_detection(tmp_path):
    """Codex review: staging wrote carried/backfilled boxes into the frame, which then counted as a
    detection and sent an unrelated weak box to the temporal model."""
    temporal = FakeTemporal([done((BOX, 0.9))] * 3)
    engine = _engine(tmp_path, temporal)
    unrelated = np.array([[870.4, 870.4, 102.4, 102.4, 0.12]])  # FAR_BOX, not a detection of the event
    ids = _fire(engine)
    ids += _fire(engine, pred=unrelated)  # quiet for the predictor (no output), still ongoing
    assert _state(engine)["ongoing"]
    ids += _fire(engine, n=2)  # frame 2 staged with the carried box under job2's verdict
    assert ids[1] in _sent(engine)
    assert all(ids[1] not in w for w in temporal.windows())
    assert temporal.windows()[-1] == [ids[0], ids[2], ids[3]]


def test_window_keeps_the_first_detections_across_quiet_frames(tmp_path):
    """Codex review: with quiet frames in between, the first detections left the 11-capture buffer
    before the 10th, and the last window had 7 frames."""
    temporal = FakeTemporal([done((BOX, 0.1))] * 7)
    engine = _engine(tmp_path, temporal, temporal_min_frames=4)
    detections = []
    for _ in range(10):
        detections += _fire(engine)
        _quiet(engine)  # ongoing through hysteresis
    assert temporal.windows()[-1] == detections


def test_final_positive_verdict_on_a_quiet_frame_sends_the_end_of_the_event(tmp_path):
    """Codex review: the full-window verdict arriving on a quiet frame only applied from the next
    detection, so the trailing frames were lost when the event ended."""
    temporal = FakeTemporal([done((BOX, 0.1)), done((BOX, 0.9))])
    engine = _engine(tmp_path, temporal, temporal_window=2)
    ids = _fire(engine, n=2)  # job1 rejected at frame 2, the full window (2) submitted
    ids += _quiet(engine)  # job2 validates it: final, every frame goes out, the quiet one included
    _quiet(engine)  # event over
    assert _sent(engine) == ids
    assert _scores(engine)[-1] == [0.9]


def test_event_too_short_to_score_is_dropped_and_does_not_count_in_the_next(tmp_path):
    temporal = FakeTemporal([PENDING])
    engine = _engine(tmp_path, temporal, temporal_min_frames=4)
    short = _fire(engine, n=2)
    _quiet(engine, n=3)
    assert not _state(engine)["ongoing"]
    assert _state(engine)["temporal_events"] == []  # dropped
    ids = _fire(engine, n=3)
    assert not temporal.submitted  # 3 detections in this event, not 5
    ids += _fire(engine)
    assert temporal.windows() == [ids]
    assert not engine._alerts
    assert not set(short) & set(ids)


def test_disabled_without_url(tmp_path, mock_wildfire_image):
    engine = Engine(cache_folder=str(tmp_path), conf_thresh=0.1, nb_consecutive_frames=2, cam_creds={CAM: ("token", 7)})
    engine.api_client = {"192.168.1.10": MagicMock()}
    assert engine.temporal is None
    for _ in range(2):
        engine.predict(mock_wildfire_image, CAM, fake_pred=PRED.T.copy())
    state = engine._states[CAM]
    assert state["ongoing"]
    assert state["last_predictions"].maxlen == 2  # window untouched
    assert engine._alerts  # staged on the spot, no gate
    assert all(a["temporal_scores"] is None for a in engine._alerts)


def test_upload_keeps_at_most_five_boxes(tmp_path):
    """Backfill at tracked locations can push a frame past the 5 boxes pyroclient accepts, which
    used to drop the whole frame, detections included."""
    temporal = FakeTemporal([done(([0.0, 0.0, 1.0, 1.0], 0.9))] * 3)
    engine = _engine(tmp_path, temporal)
    backfilled = [[0.05 + 0.15 * i, 0.8, 0.1 + 0.15 * i, 0.85, 0.0] for i in range(5)]
    engine._backfill_bboxes = lambda bboxes, _preds, _tracked: list(bboxes) + backfilled
    _fire(engine, n=3)
    assert engine._alerts
    for alert in engine._alerts:
        assert len(alert["bboxes"]) == 5
        assert len(alert["temporal_scores"]) == 5
        assert alert["bboxes"][0][:4] == BOX  # the real detection is kept, a backfilled box is left out


def test_event_memory_is_bounded(tmp_path):
    """Codex review: every frame of an event stayed in RAM, staged ones included, and a camera API
    serving the same capture again and again held its JPEGs forever (the window never grows)."""
    engine = _engine(tmp_path, FakeTemporal([done((BOX, 0.9))] * 2), temporal_window=2)
    _fire(engine, n=60)  # final verdict at frame 3, every later frame goes out on capture
    assert len(engine._alerts) == 60
    assert len(_event(engine)["frames"]) <= 3  # the tracked-boxes lookback only

    engine = _engine(tmp_path, FakeTemporal(), temporal_min_frames=4, temporal_window=2)
    image = _image()
    _fire(engine, n=80, image=image)  # one distinct detection: never scored
    assert not engine._alerts
    assert sum(not e[4] for e in _event(engine)["frames"]) <= 6  # held frames capped at 3 windows


def test_a_later_event_goes_out_after_an_earlier_one_still_pending(tmp_path):
    """Codex review: B was validated first and uploaded before A, whose verdict came later."""
    temporal = FakeTemporal([PENDING, PENDING, PENDING, PENDING, done((BOX, 0.9))] + [done((BOX, 0.9))] * 4)
    engine = _engine(tmp_path, temporal)
    a = _fire(engine) + _quiet(engine)  # A submits job1, pending
    _quiet(engine)  # A over
    b = _fire(engine)  # B submits job2; job1 pending
    b += _fire(engine)  # job2 validates B, held behind A; job1 pending
    assert not engine._alerts
    _fire(engine)  # job1 validates A: A goes out, then B
    sent = _sent(engine)
    assert sent[: len(a)] == a
    assert set(b) <= set(sent[len(a) :])


def test_a_later_capture_does_not_change_the_boxes_sent_under_an_earlier_verdict(tmp_path):
    """Codex review: a box detected after the submission became a tracked location and was
    backfilled onto the frames of the earlier verdict, with its score."""
    temporal = FakeTemporal([done(([0.0, 0.0, 1.0, 1.0], 0.9))] * 2)
    engine = _engine(tmp_path, temporal)
    _fire(engine, pred=np.array([[460.8, 460.8, 102.4, 102.4, 0.6], [870.4, 870.4, 102.4, 102.4, 0.07]]))
    _fire(engine, pred=np.array([[460.8, 460.8, 102.4, 102.4, 0.6], [870.4, 870.4, 102.4, 102.4, 0.6]]))
    assert [b[:4] for b in engine._alerts[0]["bboxes"]] == [BOX]  # frame 1 without frame 2's FAR_BOX


@pytest.mark.parametrize(
    "reply",
    [
        done(([0.4, 0.4], 0.9)),  # box of 2 coordinates
        done((["a", 0.4, 0.5, 0.5], 0.9)),
        done(([0.4, 0.4, float("nan"), 0.5], 0.9)),
        {"status": "done", "verdict": None},
        {"status": "done", "verdict": []},
        [],
    ],
    ids=["short_box", "text_box", "nan_box", "null_verdict", "list_verdict", "list_reply"],
)
def test_malformed_verdict_fails_open(tmp_path, reply):
    """Codex review: a malformed verdict raised out of predict, kept the job, and blocked the pose."""
    temporal = FakeTemporal([reply, done((BOX, 0.9))])
    engine = _engine(tmp_path, temporal, temporal_window=1)
    _fire(engine)  # submitted (full window)
    _fire(engine)  # malformed verdict: fail open, frames go out unscored, the job is retried
    assert _event(engine)["bypass"] is True
    assert engine._alerts
    assert all(s is None for s in _scores(engine))
    _fire(engine)  # the retried job's verdict resumes scoring
    assert _event(engine)["bypass"] is False
    assert _scores(engine)[-1] == [0.9]


def test_malformed_submission_reply_fails_open(tmp_path):
    temporal = MagicMock(submit=MagicMock(side_effect=TypeError("list indices must be integers")))
    engine = _engine(tmp_path, temporal)
    _fire(engine)
    assert _event(engine)["bypass"] is True
    assert engine._alerts


def test_crop_boxes_are_frozen_on_capture_for_an_event_held_behind_another(tmp_path):
    """Codex review: an event waiting behind an earlier one had no crop box yet, so the context crops
    of its quiet frames did not cover its fire location."""
    engine = _engine(tmp_path, FakeTemporal([PENDING] * 10))
    _fire(engine)  # A: job pending
    _quiet(engine, n=2)  # A over
    _fire(engine)  # B, held behind A
    _quiet(engine)  # B's quiet frame (hysteresis)
    b = _event(engine)
    assert b["crop_boxes"]
    crop = b["frames"][-1][0]  # context crop of the quiet frame
    left, top, right, bottom = b["crop_boxes"][0]
    assert crop is not None
    assert crop.left <= left
    assert crop.top <= top
    assert right <= crop.right
    assert bottom <= crop.bottom
