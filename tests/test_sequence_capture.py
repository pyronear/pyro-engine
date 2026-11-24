import json
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from pyroengine import sequence_capture as sc
from pyroengine.sequence_capture import SequenceCapture

CAM = "10.0.0.1_2"
BOX = np.array([[0.45, 0.45, 0.55, 0.55, 0.3]])
NO_BOX = np.zeros((0, 5))
FRAME = Image.new("RGB", (64, 36))


class FakeUpload:
    def __init__(self, fail_on=None) -> None:
        self.calls = {}
        self.manifests = []  # every manifest, also those started in the same second (same key)
        self.fail_on = fail_on

    def __call__(self, key, data, content_type) -> None:
        if self.fail_on and key.endswith(self.fail_on):
            raise OSError("S3 down")
        self.calls[key] = (data, content_type)
        if key.endswith("manifest.json"):
            self.manifests.append(json.loads(data))


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def make(monkeypatch, frames=3, upload=None, **kwargs):
    monkeypatch.setattr(sc, "FRAMES", frames)
    clock = Clock()
    monkeypatch.setattr(sc.time, "monotonic", clock)
    upload = upload or FakeUpload()
    return SequenceCapture(upload, "pfx", {"10.0.0.1": "cam-a"}, **kwargs), upload, clock


def feed(cap, clock, preds_list):
    for preds in preds_list:
        cap.submit(CAM, FRAME, preds)
        clock.t += 30
    cap._queue.join()


def manifests(upload):
    return upload.manifests


def test_sequence_keeps_the_next_frames_and_ignores_weak_boxes(monkeypatch):
    cap, upload, clock = make(monkeypatch)
    weak = np.array([[0.45, 0.45, 0.55, 0.55, 0.04]])
    feed(cap, clock, [NO_BOX, weak, BOX, NO_BOX, BOX, NO_BOX])
    jpgs = sorted(k for k in upload.calls if k.endswith(".jpg"))
    assert len(jpgs) == 3  # the triggering frame, a frame without box, then one more
    assert all(k.startswith("pfx/sequences/cam-a/") and "_pose2/" in k for k in jpgs)
    assert upload.calls[jpgs[0]][0][:2] == b"\xff\xd8"
    (man,) = manifests(upload)
    assert man["camera"] == "cam-a"
    assert [f["key"] for f in man["frames"]] == jpgs
    assert [len(f["boxes"]) for f in man["frames"]] == [1, 0, 1]


def test_new_sequence_only_after_idle_time(monkeypatch):
    cap, upload, clock = make(monkeypatch)
    feed(cap, clock, [BOX] * 10)  # one long detection: one sequence
    assert len(manifests(upload)) == 1
    clock.t += sc.IDLE_S - 31  # last box less than IDLE_S ago
    feed(cap, clock, [BOX])
    assert len(manifests(upload)) == 1
    assert cap._active == {}
    clock.t += sc.IDLE_S
    feed(cap, clock, [BOX, NO_BOX, NO_BOX])
    assert len(manifests(upload)) == 2


def test_daily_quota_per_pose(monkeypatch):
    monkeypatch.setattr(sc, "MAX_SEQUENCES_PER_DAY", 2)
    cap, upload, clock = make(monkeypatch, frames=1)
    for _ in range(3):
        feed(cap, clock, [BOX])
        clock.t += sc.IDLE_S
    assert len(manifests(upload)) == 2


def test_busy_worker_or_failed_upload_marks_the_frame(monkeypatch):
    cap, upload, clock = make(monkeypatch, upload=FakeUpload(fail_on="/01.jpg"), max_pending_frames=0)
    feed(cap, clock, [BOX, NO_BOX, NO_BOX])
    (man,) = manifests(upload)
    assert [f["key"] for f in man["frames"]] == [None, None, None]
    assert cap.dropped == 3

    cap, upload, clock = make(monkeypatch, upload=FakeUpload(fail_on="/01.jpg"))
    feed(cap, clock, [BOX, NO_BOX, NO_BOX])
    (man,) = manifests(upload)
    assert [f["key"] is None for f in man["frames"]] == [False, True, False]


def test_engine_feeds_every_frame_and_survives_capture_errors(tmp_path):
    from pyroengine.engine import Engine

    engine = Engine(cache_folder=str(tmp_path), frame_size=(720, 1280))
    engine.model = MagicMock(return_value=NO_BOX)
    engine.sequence_capture = MagicMock()
    frame = Image.new("RGB", (3840, 2160))
    engine.predict(frame, CAM)
    cam_id, sent_frame, sent_preds = engine.sequence_capture.submit.call_args.args
    assert (cam_id, sent_frame) == (CAM, frame)
    assert sent_preds.shape == (0, 5)  # frames without boxes are fed too

    engine.sequence_capture.submit.side_effect = RuntimeError("boom")
    engine.predict(frame, CAM)  # does not raise
