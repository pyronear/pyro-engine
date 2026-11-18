import io
import json
import threading

import numpy as np
from PIL import Image

from pyroengine.crop_check import CropCheck, crop_score, hd_window

FRAME_4K = (3840, 2160)


class FakeUpload:
    def __init__(self) -> None:
        self.calls = {}

    def __call__(self, key, data, content_type) -> None:
        self.calls[key] = (data, content_type)


def test_hd_window_centred_and_kept_inside():
    assert hd_window(FRAME_4K, [0.45, 0.45, 0.55, 0.55]) == (1280, 720, 2560, 1440)
    # Near the top-left and bottom-right corners the window shifts but keeps its HD size
    assert hd_window(FRAME_4K, [0.0, 0.0, 0.02, 0.02]) == (0, 0, 1280, 720)
    assert hd_window(FRAME_4K, [0.98, 0.98, 1.0, 1.0]) == (2560, 1440, 3840, 2160)
    # A frame smaller than HD is clipped
    assert hd_window((640, 480), [0.4, 0.4, 0.6, 0.6]) == (0, 0, 640, 480)


def test_crop_score_only_counts_overlapping_boxes():
    box = [0.45, 0.45, 0.55, 0.55]
    win = hd_window(FRAME_4K, box)  # the box covers the crop centre, 0.2-0.8 in crop coords
    preds = np.array([[0.4, 0.4, 0.6, 0.6, 0.3], [0.0, 0.0, 0.1, 0.1, 0.9]])
    assert crop_score(preds, box, win, FRAME_4K) == 0.3
    assert crop_score(np.zeros((0, 5)), box, win, FRAME_4K) == 0.0


def run_checks(full_conf, crop_conf):
    upload = FakeUpload()
    cc = CropCheck(lambda _img: np.array([[0.4, 0.4, 0.6, 0.6, crop_conf]]), upload, "pfx", {"10.0.0.1": "cam-a"})
    frame = Image.new("RGB", FRAME_4K)
    cc.submit("10.0.0.1_2", frame, b"hd", np.array([[0.45, 0.45, 0.55, 0.55, full_conf]]))
    cc._queue.join()
    cc.flush()
    return upload


def records(upload):
    (data, ctype), *_ = [v for k, v in upload.calls.items() if k.endswith(".jsonl")]
    assert ctype == "application/x-ndjson"
    return [json.loads(line) for line in data.decode().splitlines()]


def test_agreement_uploads_record_only():
    upload = run_checks(full_conf=0.4, crop_conf=0.5)
    assert all(k.startswith("pfx/records/") for k in upload.calls)
    (rec,) = records(upload)
    assert rec["camera"] == "cam-a"
    assert rec["boxes"][0] | {"window": None} == {
        "bbox": [0.45, 0.45, 0.55, 0.55],
        "full": 0.4,
        "crop": 0.5,
        "window": None,
        "disagree": False,
    }
    assert rec["keys"] == {}


def test_disagreement_uploads_full_and_crop_both_ways():
    for full_conf, crop_conf in [(0.4, 0.1), (0.1, 0.4)]:
        upload = run_checks(full_conf, crop_conf)
        (rec,) = records(upload)
        assert rec["boxes"][0]["disagree"]
        assert set(rec["keys"]) == {"full", "crop0"}
        assert rec["keys"]["full"].startswith("pfx/cam-a/")
        assert rec["keys"]["full"].endswith("_pose2_full.jpg")
        assert upload.calls[rec["keys"]["full"]][0] == b"hd"
        crop = Image.open(io.BytesIO(upload.calls[rec["keys"]["crop0"]][0]))
        assert crop.size == (1280, 720)


def test_full_queue_drops_without_blocking():
    release = threading.Event()

    def slow_model(img):
        release.wait(5)
        return np.zeros((0, 5))

    cc = CropCheck(slow_model, FakeUpload(), max_pending_crops=2)
    frame = Image.new("RGB", (1280, 720))
    preds = np.array([[0.4, 0.4, 0.6, 0.6, 0.3]])
    for _ in range(4):
        cc.submit("10.0.0.1", frame, b"hd", preds)
    assert cc.dropped == 2
    # A frame with more boxes than the budget is skipped too
    cc.submit("10.0.0.1", frame, b"hd", np.repeat(preds, 3, axis=0))
    assert cc.dropped == 3
    release.set()
    cc._queue.join()


def test_failed_records_upload_is_retried():
    calls = []

    def flaky(key, data, content_type):
        calls.append(key)
        if len(calls) == 1:
            raise OSError("offline")

    cc = CropCheck(lambda _img: np.zeros((0, 5)), flaky)
    cc.submit("10.0.0.1", Image.new("RGB", FRAME_4K), b"hd", np.array([[0.4, 0.4, 0.6, 0.6, 0.1]]))
    cc._queue.join()
    cc.flush()
    assert len(cc._records) == 1
    cc.flush()
    assert cc._records == []
    assert len(calls) == 2


def test_engine_submits_original_frame_and_boxes(tmpdir_factory):
    from unittest.mock import MagicMock

    from pyroengine.engine import Engine

    engine = Engine(cache_folder=str(tmpdir_factory.mktemp("crop_check_cache")), frame_size=(720, 1280))
    preds = np.array([[0.4, 0.4, 0.6, 0.6, 0.3]])
    engine.model = MagicMock(return_value=preds)
    engine.crop_check = MagicMock()
    frame = Image.new("RGB", FRAME_4K)
    engine.predict(frame, "10.0.0.1_0")
    cam_id, sent_frame, hd_jpeg, sent_preds = engine.crop_check.submit.call_args.args
    assert cam_id == "10.0.0.1_0"
    assert sent_frame is frame
    assert Image.open(io.BytesIO(hd_jpeg)).size == (1280, 720)
    np.testing.assert_array_equal(sent_preds, preds)

    # No box, no check
    engine.crop_check.reset_mock()
    engine.model.return_value = np.zeros((0, 5))
    engine.predict(frame, "10.0.0.1_0")
    engine.crop_check.submit.assert_not_called()
