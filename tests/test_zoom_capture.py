import io
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pyroengine.core import SystemController
from pyroengine.zoom_capture import ZoomCapture, crop_square


@pytest.fixture(autouse=True)
def no_settle(monkeypatch):
    monkeypatch.setattr("pyroengine.zoom_capture.time.sleep", lambda *_a, **_k: None)


@pytest.fixture
def events():
    """Upload keys and PTZ calls in the order they happened."""
    return []


@pytest.fixture
def uploads():
    return {}


@pytest.fixture
def capture(uploads, events):
    def upload(key, data, _ct):
        uploads[key] = data
        events.append(key)

    return ZoomCapture(upload, prefix="rnd")


@pytest.fixture
def client(events):
    c = MagicMock()
    c.capture_image.return_value = Image.new("RGB", (3840, 2160), (10, 20, 30))
    c.start_patrol.side_effect = lambda _ip: events.append("start_patrol")
    return c


def _manifest(uploads):
    key = next(k for k in uploads if k.endswith("_manifest.json"))
    return key, json.loads(uploads[key])


def test_crop_square_stays_inside_frame_near_corner():
    frame = Image.new("RGB", (3840, 2160))
    data = crop_square(frame, [0.0, 0.0, 0.02, 0.02, 0.9])
    assert Image.open(io.BytesIO(data)).size == (1024, 1024)


def test_crop_square_grows_with_large_bbox():
    frame = Image.new("RGB", (3840, 2160))
    data = crop_square(frame, [0.1, 0.1, 0.6, 0.3, 0.9])  # 1920 px wide bbox
    w, h = Image.open(io.BytesIO(data)).size
    assert w == h == 2160  # 1920 * 1.2 capped by the frame height


def test_full_sequence_uploads_and_restores_patrol(capture, client, uploads, events):
    frame = Image.new("RGB", (3840, 2160))
    bboxes = [[0.1, 0.1, 0.2, 0.2, 0.5], [0.6, 0.6, 0.7, 0.8, 0.9]]
    capture.trigger(client, "10.0.0.1", 2, "cam", frame, bboxes, time.monotonic() + 60)
    capture.wait(5)

    names = [c[0] for c in client.method_calls]
    assert names == ["stop_patrol", "goto_preset", "click_to_move", "zoom", "capture_image", "zoom", "start_patrol"]
    client.goto_preset.assert_called_once_with("10.0.0.1", 2)
    _, cx, cy = client.click_to_move.call_args.args
    assert (round(cx, 3), round(cy, 3)) == (0.65, 0.7)  # best bbox centre
    assert [c.args[1] for c in client.zoom.call_args_list] == [10, 0]

    key, manifest = _manifest(uploads)
    assert key.startswith("rnd/cam/")
    assert manifest["status"] == "done"
    assert len(manifest["crops"]) == 2
    assert manifest["bboxes"][0][4] == 0.9
    assert manifest["zoom_crop"].endswith("_zoom10.jpg")
    assert Image.open(io.BytesIO(uploads[manifest["zoom_crop"]])).size == (1024, 1024)
    # The zoomed image is uploaded only once the camera is back on patrol
    assert events.index("start_patrol") < events.index(manifest["zoom_crop"])
    # The restarted patrol has not refreshed the pose images yet
    assert capture.images_stale("10.0.0.1")
    assert not capture.images_stale("10.0.0.9")


def test_deadline_passed_uploads_crops_only(capture, client, uploads):
    frame = Image.new("RGB", (1920, 1080))
    capture.trigger(client, "10.0.0.1", 1, "cam", frame, [[0.4, 0.4, 0.5, 0.5, 0.7]], time.monotonic() - 1)
    capture.wait(5)

    assert not client.method_calls  # deadline hit before stop_patrol: nothing to restore
    _, manifest = _manifest(uploads)
    assert manifest["status"] == "deadline before stop_patrol"
    assert len(manifest["crops"]) == 1


def test_stream_aborts_sequence_and_restores(capture, client, uploads):
    frame = Image.new("RGB", (1920, 1080))
    stream = iter([False, False, True])  # stream shows up before click_to_move
    capture.trigger(
        client, "10.0.0.1", 1, "cam", frame, [[0.4, 0.4, 0.5, 0.5, 0.7]], time.monotonic() + 60, lambda: next(stream)
    )
    capture.wait(5)

    names = [c[0] for c in client.method_calls]
    assert names == ["stop_patrol", "goto_preset", "zoom", "start_patrol"]
    _, manifest = _manifest(uploads)
    assert manifest["status"] == "stream active before click_to_move"


def test_error_mid_sequence_still_resets_zoom_and_patrol(capture, client, uploads):
    client.click_to_move.side_effect = RuntimeError("409 busy")
    frame = Image.new("RGB", (1920, 1080))
    capture.trigger(client, "10.0.0.1", 1, "cam", frame, [[0.4, 0.4, 0.5, 0.5, 0.7]], time.monotonic() + 60)
    capture.wait(5)

    names = [c[0] for c in client.method_calls]
    assert names == ["stop_patrol", "goto_preset", "click_to_move", "zoom", "start_patrol"]
    client.zoom.assert_called_once_with("10.0.0.1", 0)
    _, manifest = _manifest(uploads)
    assert manifest["status"] == "error"
    assert "409 busy" in manifest["errors"][0]


def test_zoom_reset_is_retried_before_patrol_restart(capture, client, uploads):
    client.zoom.side_effect = [None, RuntimeError("timeout"), None]
    frame = Image.new("RGB", (1920, 1080))
    capture.trigger(client, "10.0.0.1", 1, "cam", frame, [[0.4, 0.4, 0.5, 0.5, 0.7]], time.monotonic() + 60)
    capture.wait(5)

    assert [c.args[1] for c in client.zoom.call_args_list] == [10, 0, 0]
    names = [c[0] for c in client.method_calls]
    assert names[-1] == "start_patrol"
    _, manifest = _manifest(uploads)
    assert manifest["status"] == "done"
    assert manifest["errors"] == ["zoom_reset: timeout"]


def test_trigger_skipped_while_previous_job_runs(capture, client, uploads):
    gate = threading.Event()
    client.stop_patrol.side_effect = lambda _ip: gate.wait(5)
    frame = Image.new("RGB", (1920, 1080))
    capture.trigger(client, "10.0.0.1", 1, "cam", frame, [[0.4, 0.4, 0.5, 0.5, 0.7]], time.monotonic() + 60)
    capture.trigger(client, "10.0.0.1", 2, "cam", frame, [[0.4, 0.4, 0.5, 0.5, 0.7]], time.monotonic() + 60)
    gate.set()
    capture.wait(5)

    assert client.stop_patrol.call_count == 1
    assert len([k for k in uploads if k.endswith("_manifest.json")]) == 1
    assert len([k for k in uploads if "_crop" in k]) == 1


@patch("pyroengine.core.time.sleep", lambda *_a, **_k: None)
@patch("pyroengine.core.PyroCameraAPIClient")
def test_inference_loop_triggers_zoom_capture_for_reolink_only(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.get_latest_image.return_value = Image.new("RGB", (100, 100), (255, 200, 200))
    mock_client.get_stream_status.return_value = {"active_streams": 0}

    engine = MagicMock()
    engine._states = {
        "10.0.0.1_1": {
            "ongoing": True,
            "last_predictions": [(None, None, [[0.1, 0.1, 0.2, 0.2, 0.8]], "", False, b"")],
        },
        "10.0.0.1_2": {"ongoing": False, "last_predictions": [(None, None, [], "", False, b"")]},
        "10.0.0.2_1": {
            "ongoing": True,
            "last_predictions": [(None, None, [[0.1, 0.1, 0.2, 0.2, 0.8]], "", False, b"")],
        },
    }
    camera_data = {
        "10.0.0.1": {"name": "reo", "type": "ptz", "adapter": "reolink", "poses": [1, 2]},
        "10.0.0.2": {"name": "lino", "type": "ptz", "adapter": "linovision", "poses": [1]},
    }
    zoom_capture = MagicMock()
    zoom_capture.images_stale.return_value = False
    controller = SystemController(engine, camera_data, "http://fake.url", zoom_capture=zoom_capture)

    controller.inference_loop(deadline=time.monotonic() + 30)

    zoom_capture.trigger.assert_called_once()
    args = zoom_capture.trigger.call_args.args
    assert args[1:4] == ("10.0.0.1", 1, "reo")
    assert args[5] == [[0.1, 0.1, 0.2, 0.2, 0.8]]
    assert args[7] == controller._any_stream_active

    zoom_capture.trigger.reset_mock()
    controller.inference_loop()  # no deadline: zoom capture disabled
    zoom_capture.trigger.assert_not_called()

    # A camera whose patrol just restarted is skipped for the whole loop
    engine.predict.reset_mock()
    zoom_capture.images_stale.side_effect = lambda ip: ip == "10.0.0.1"
    controller.inference_loop(deadline=time.monotonic() + 30)
    assert [c.args[1] for c in engine.predict.call_args_list] == ["10.0.0.2_1"]
    zoom_capture.trigger.assert_not_called()
