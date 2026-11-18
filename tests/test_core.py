import itertools
import logging
import pathlib
from collections import defaultdict
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

from pyroengine.core import NIGHT_FRAMES_BEFORE_RESET, NIGHT_PASSES_BEFORE_SLEEP, SystemController, is_day_time


# Disable sleeps in SystemController during tests to make them fast
@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    # SystemController uses time.sleep from pyroengine.core
    monkeypatch.setattr("pyroengine.core.time.sleep", lambda *_args, **_kwargs: None)


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.predict.return_value = 0.0
    engine.conf_thresh = 0.25
    # The summary reads the predictor's per-camera verdict, not the raw confidence.
    engine._states = defaultdict(lambda: {"ongoing": False})
    return engine


@pytest.fixture
def mock_camera_data():
    return {"192.168.1.1": {"name": "cam1", "type": "ptz", "poses": [1, 2]}}


def test_is_day_time_ir_strategy():
    day_img = Image.new("RGB", (100, 100), (255, 200, 200))
    assert is_day_time(None, day_img, "ir")

    night_img = Image.new("RGB", (100, 100), (255, 255, 255))
    assert not is_day_time(None, night_img, "ir")


def test_is_day_time_time_strategy(tmp_path):
    cache = tmp_path
    pathlib.Path(cache / "sunset_sunrise.txt").write_text("06:00\n18:00\n")

    with patch("pyroengine.core.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 6, 17, 10, 0)
        mock_datetime.strptime = datetime.strptime
        assert is_day_time(cache, None, "time")

        mock_datetime.now.return_value = datetime(2024, 6, 17, 20, 0)
        assert not is_day_time(cache, None, "time")


@patch("pyroengine.core.PyroCameraAPIClient")
def test_focus_finder_runs_hourly(mock_client_class, mock_engine, mock_camera_data):
    mock_client = mock_client_class.return_value
    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url")
    controller.is_day = True
    controller.last_autofocus = datetime.now().replace(hour=0)

    controller.focus_finder()

    assert mock_client.run_focus_optimization.called
    assert mock_client.stop_patrol.called
    assert mock_client.start_patrol.called


@patch("pyroengine.core.PyroCameraAPIClient")
def test_inference_loop_triggers_predict(mock_client_class, mock_engine, mock_camera_data):
    mock_client = mock_client_class.return_value
    dummy_img = Image.new("RGB", (100, 100), (255, 200, 200))
    mock_client.get_latest_image.return_value = dummy_img
    # New behavior, no active streams means inference should run
    mock_client.get_stream_status.return_value = {"active_streams": 0}

    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url")
    controller.is_day = True

    controller.inference_loop()

    assert mock_engine.predict.called
    mock_client.get_latest_image.assert_called()


@patch("pyroengine.core.PyroCameraAPIClient")
def test_inference_loop_quiet_round_logs_single_line(mock_client_class, mock_engine, mock_camera_data, caplog):
    """A round with no detection reports one INFO summary, not one line per pose."""
    mock_client = mock_client_class.return_value
    mock_client.get_latest_image.return_value = Image.new("RGB", (100, 100), (255, 200, 200))
    mock_client.get_stream_status.return_value = {"active_streams": 0}

    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url")
    controller._last_frame_digest.clear()  # the warm-up pass in __init__ already read these frames

    with caplog.at_level(logging.INFO, logger="pyroengine.core"):
        controller.inference_loop()

    info_lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_lines) == 1
    assert "analyzed=2" in info_lines[0]
    assert "positive=0" in info_lines[0]
    assert "max_confidence" in info_lines[0]


@patch("pyroengine.core.PyroCameraAPIClient")
def test_inference_loop_summary_counts_failures(mock_client_class, mock_engine, mock_camera_data, caplog):
    """Poses that fail to capture are reported in the round summary."""
    mock_client = mock_client_class.return_value
    mock_client.get_latest_image.side_effect = Exception("camera down")
    mock_client.get_stream_status.return_value = {"active_streams": 0}

    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url")
    controller._last_frame_digest.clear()  # the warm-up pass in __init__ already read these frames

    with caplog.at_level(logging.INFO, logger="pyroengine.core"):
        controller.inference_loop()

    summary = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO][-1]
    assert "analyzed=0" in summary
    assert "failed=2" in summary


@patch("pyroengine.core.PyroCameraAPIClient")
def test_inference_loop_handles_http_error(mock_client_class, mock_engine, mock_camera_data):
    mock_client = mock_client_class.return_value
    mock_error = requests.HTTPError(response=MagicMock(text="404 Not Found"))
    mock_client.get_latest_image.side_effect = mock_error
    # New behavior, force no active streams so the loop reaches get_latest_image
    mock_client.get_stream_status.return_value = {"active_streams": 0}

    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url")

    controller.inference_loop()

    assert mock_client.get_latest_image.called
    assert not mock_engine.predict.called


@patch("pyroengine.core.PyroCameraAPIClient")
def test_inference_loop_handles_generic_error(mock_client_class, mock_engine, mock_camera_data):
    mock_client = mock_client_class.return_value
    mock_client.get_latest_image.side_effect = Exception("Something went wrong")
    # New behavior, force no active streams so the loop reaches get_latest_image
    mock_client.get_stream_status.return_value = {"active_streams": 0}

    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url")

    controller.inference_loop()

    assert mock_client.get_latest_image.called
    assert not mock_engine.predict.called


@patch("pyroengine.core.PyroCameraAPIClient")
def test_heartbeat_file_is_written(mock_client_class, mock_engine, mock_camera_data, tmp_path):
    """The healthcheck relies on this file, so it must be refreshed independently of log level."""
    mock_client = mock_client_class.return_value
    mock_client.get_latest_image.return_value = Image.new("RGB", (100, 100), (255, 200, 200))
    mock_client.get_stream_status.return_value = {"active_streams": 0}
    heartbeat = tmp_path / "heartbeat"

    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url", heartbeat_file=str(heartbeat))
    heartbeat.unlink(missing_ok=True)

    controller.inference_loop()

    assert heartbeat.exists()
    assert heartbeat.read_text()


@patch("pyroengine.core.PyroCameraAPIClient")
def test_heartbeat_written_when_no_capture_succeeds(mock_client_class, mock_engine, mock_camera_data, tmp_path):
    """Offline cameras must not get the engine killed: a restart would not bring them back."""
    mock_client = mock_client_class.return_value
    mock_client.get_latest_image.return_value = None
    mock_client.get_stream_status.return_value = {"active_streams": 0}
    heartbeat = tmp_path / "heartbeat"

    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url", heartbeat_file=str(heartbeat))
    heartbeat.unlink(missing_ok=True)

    controller.inference_loop()

    assert heartbeat.exists()


@patch("pyroengine.core.PyroCameraAPIClient")
def test_summary_counts_ongoing_events_as_positive(mock_client_class, mock_engine, mock_camera_data, caplog):
    """The positive count follows the predictor's hysteresis verdict, not the raw threshold."""
    mock_client = mock_client_class.return_value
    mock_client.get_latest_image.return_value = Image.new("RGB", (100, 100), (255, 200, 200))
    mock_client.get_stream_status.return_value = {"active_streams": 0}
    # Ongoing event whose confidence dipped below conf_thresh: still a positive.
    mock_engine.predict.return_value = 0.22
    mock_engine._states = defaultdict(lambda: {"ongoing": True})

    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url")
    controller._last_frame_digest.clear()  # the warm-up pass in __init__ already read these frames

    with caplog.at_level(logging.INFO, logger="pyroengine.core"):
        controller.inference_loop()

    summary = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO][-1]
    assert "positive=2" in summary


@patch("pyroengine.core.PyroCameraAPIClient")
def test_heartbeat_write_failure_does_not_raise(mock_client_class, mock_engine, mock_camera_data, tmp_path):
    """A broken heartbeat path must not stop detection."""
    mock_client = mock_client_class.return_value
    mock_client.get_stream_status.return_value = {"active_streams": 0}
    unwritable = tmp_path / "missing_dir" / "heartbeat"

    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url", heartbeat_file=str(unwritable))

    controller._write_heartbeat()  # must not raise
    assert not unwritable.exists()


@patch("pyroengine.core.PyroCameraAPIClient")
def test_inference_loop_skips_when_stream_active(mock_client_class, mock_engine, mock_camera_data):
    mock_client = mock_client_class.return_value
    mock_client.get_stream_status.return_value = {"active_streams": 1}

    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url")
    controller.inference_loop()

    assert not mock_client.get_latest_image.called
    assert not mock_engine.predict.called


@patch("pyroengine.core.PyroCameraAPIClient")
def test_night_stops_running_patrols(mock_client_class, mock_engine, mock_camera_data, monkeypatch):
    mock_client = mock_client_class.return_value
    mock_client.get_patrol_status.return_value = {"patrol_running": True}
    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url")
    controller.is_day = False

    class NightSleepError(Exception):
        pass

    def sleep(_s):
        raise NightSleepError

    monkeypatch.setattr("pyroengine.core.time.sleep", sleep)
    with pytest.raises(NightSleepError):
        controller.main_loop(period=30)

    mock_client.stop_patrol.assert_called_once_with("192.168.1.1")


@patch("pyroengine.core.PyroCameraAPIClient")
def test_night_frames_are_skipped_and_site_sleeps_only_when_all_are_night(mock_client_class, mock_engine):
    camera_data = {
        "192.168.1.1": {"name": "cam1", "type": "ptz", "poses": [1]},
        "192.168.1.2": {"name": "cam2", "type": "static"},
    }
    day_img = Image.new("RGB", (100, 100), (255, 200, 200))
    night_img = Image.new("RGB", (100, 100), (255, 255, 255))
    mock_client = mock_client_class.return_value
    mock_client.get_stream_status.return_value = {"active_streams": 0}
    controller = SystemController(mock_engine, camera_data, "http://fake.url")

    # One camera in infrared: it is not analyzed, the other one keeps the site awake
    mock_client.get_latest_image.side_effect = [night_img, day_img]
    controller.inference_loop()
    mock_engine.predict.assert_called_once_with(day_img, "192.168.1.2")
    assert controller.is_day

    # Every camera at night: the site sleeps after NIGHT_PASSES_BEFORE_SLEEP passes, not one
    mock_client.get_latest_image.side_effect = lambda *_: night_img
    for _ in range(NIGHT_PASSES_BEFORE_SLEEP - 1):
        controller.inference_loop()
        assert controller.is_day
    controller.inference_loop()
    assert mock_engine.predict.call_count == 1
    assert not controller.is_day

    # No frame at all (stale cache): keep the previous state
    mock_client.get_latest_image.side_effect = lambda *_: None
    controller.inference_loop()
    assert not controller.is_day

    # One day frame wakes the site and restarts the count
    controller.is_day = True
    mock_client.get_latest_image.side_effect = lambda *_: night_img
    controller.inference_loop()
    controller.inference_loop()
    mock_client.get_latest_image.side_effect = [night_img, day_img]
    controller.inference_loop()
    assert controller.is_day
    mock_client.get_latest_image.side_effect = lambda *_: night_img
    controller.inference_loop()
    assert controller.is_day

    # One camera down (204) says nothing about the light: the other one puts the site to sleep
    controller.is_day = True
    mock_client.get_latest_image.side_effect = lambda _ip, pose: night_img if pose == 1 else None
    for _ in range(NIGHT_PASSES_BEFORE_SLEEP):
        controller.inference_loop()
    assert not controller.is_day


@patch("pyroengine.core.PyroCameraAPIClient")
def test_camera_in_infrared_during_day_is_warned_once(mock_client_class, mock_engine, caplog):
    camera_data = {
        "192.168.1.1": {"name": "cam1", "type": "static"},
        "192.168.1.2": {"name": "cam2", "type": "static"},
    }
    day_img = Image.new("RGB", (100, 100), (255, 200, 200))
    night_levels = itertools.count(10)  # a new infrared capture (R == G == B) on every read
    mock_client = mock_client_class.return_value
    mock_client.get_stream_status.return_value = {"active_streams": 0}
    mock_client.get_latest_image.side_effect = lambda ip, _pose: (
        Image.new("RGB", (100, 100), (next(night_levels),) * 3) if ip == "192.168.1.1" else day_img
    )
    controller = SystemController(mock_engine, camera_data, "http://fake.url")

    with caplog.at_level(logging.WARNING):
        for _ in range(NIGHT_FRAMES_BEFORE_RESET + 2):
            controller.inference_loop()

    warnings = [r for r in caplog.records if "infrared while others" in r.message]
    assert len(warnings) == 1
    assert "192.168.1.1" in warnings[0].message
    assert controller.is_day


@patch("pyroengine.core.PyroCameraAPIClient")
def test_wake_up_probes_every_camera(mock_client_class, mock_engine):
    camera_data = {
        "192.168.1.1": {"name": "cam1", "type": "static"},
        "192.168.1.2": {"name": "cam2", "type": "static"},
    }
    day_img = Image.new("RGB", (100, 100), (255, 200, 200))
    night_img = Image.new("RGB", (100, 100), (255, 255, 255))
    mock_client = mock_client_class.return_value
    controller = SystemController(mock_engine, camera_data, "http://fake.url")

    # The first camera stays in infrared (or is down), the second one sees daylight
    mock_client.capture_image.side_effect = [night_img, day_img]
    assert controller._probe_day()
    mock_client.capture_image.side_effect = [RuntimeError("down"), day_img]
    assert controller._probe_day()

    # Every camera reports night: stay asleep
    mock_client.capture_image.side_effect = [night_img, night_img]
    assert not controller._probe_day()

    # One night frame and one camera down: the camera down says nothing, stay asleep
    mock_client.capture_image.side_effect = [night_img, RuntimeError("down")]
    assert not controller._probe_day()

    # Nobody answers: wake up and let the inference loop decide
    mock_client.capture_image.side_effect = [RuntimeError("down"), None]
    assert controller._probe_day()


@patch("pyroengine.core.PyroCameraAPIClient")
def test_wake_up_restarts_the_night_counts(mock_client_class, mock_engine):
    camera_data = {
        "192.168.1.1": {"name": "cam1", "type": "static"},
        "192.168.1.2": {"name": "cam2", "type": "static"},
    }
    night_img = Image.new("RGB", (100, 100), (255, 255, 255))
    mock_client = mock_client_class.return_value
    mock_client.get_stream_status.return_value = {"active_streams": 0}
    controller = SystemController(mock_engine, camera_data, "http://fake.url")

    # Every camera at night for the whole evening, then the site slept
    mock_client.get_latest_image.side_effect = lambda *_: night_img
    for _ in range(NIGHT_PASSES_BEFORE_SLEEP + 2):
        controller.inference_loop()
    assert not controller.is_day

    # Dawn: nobody answers the probe, the site wakes up. One all-night pass must not put it
    # back to sleep, and the camera still in infrared is warned once others see daylight.
    mock_client.capture_image.side_effect = [RuntimeError("down"), None]
    assert controller._probe_day()
    controller.is_day = True
    mock_client.get_latest_image.side_effect = lambda *_: Image.new("RGB", (100, 100), (200, 200, 200))
    controller.inference_loop()
    assert controller.is_day
    assert controller._night_frames["192.168.1.1"] == 1


@patch("pyroengine.core.PyroCameraAPIClient")
def test_identical_frame_is_not_analyzed_twice(mock_client_class, mock_engine):
    camera_data = {"192.168.1.2": {"name": "cam2", "type": "static"}}
    mock_client = mock_client_class.return_value
    mock_client.get_stream_status.return_value = {"active_streams": 0}
    mock_client.get_latest_image.side_effect = [
        Image.new("RGB", (100, 100), (255, 200, 200)),
        Image.new("RGB", (100, 100), (255, 200, 200)),
        Image.new("RGB", (100, 100), (255, 201, 200)),
    ]
    controller = SystemController(mock_engine, camera_data, "http://fake.url")

    for _ in range(3):
        controller.inference_loop()

    assert mock_engine.predict.call_count == 2
    assert controller.is_day


@patch("pyroengine.core.PyroCameraAPIClient")
def test_failed_prediction_keeps_day_and_is_retried(mock_client_class, mock_engine):
    camera_data = {"192.168.1.2": {"name": "cam2", "type": "static"}}
    day_img = Image.new("RGB", (100, 100), (255, 200, 200))
    mock_client = mock_client_class.return_value
    mock_client.get_stream_status.return_value = {"active_streams": 0}
    mock_client.get_latest_image.return_value = day_img
    mock_engine.predict.side_effect = [RuntimeError("upload failed"), 0.0]
    controller = SystemController(mock_engine, camera_data, "http://fake.url")
    controller.is_day = True

    controller.inference_loop()
    assert controller.is_day

    controller.inference_loop()
    assert mock_engine.predict.call_count == 2


@patch("pyroengine.core.PyroCameraAPIClient")
def test_night_frame_drops_detection_history(mock_client_class, mock_engine):
    camera_data = {"192.168.1.2": {"name": "cam2", "type": "static"}}
    mock_client = mock_client_class.return_value
    mock_client.get_stream_status.return_value = {"active_streams": 0}
    # Distinct infrared frames (R == G == B), as the patrol stores a new capture each cycle
    mock_client.get_latest_image.side_effect = [Image.new("RGB", (100, 100), (v, v, v)) for v in (10, 20, 30, 40)]
    mock_engine._states = {"192.168.1.2": {"ongoing": True}}
    controller = SystemController(mock_engine, camera_data, "http://fake.url")
    controller._night_frames.clear()  # __init__ already ran one pass

    # Not on one frame: cameras flicker at dusk and an ongoing event must survive it
    for _ in range(NIGHT_FRAMES_BEFORE_RESET - 1):
        controller.inference_loop()
    assert "192.168.1.2" in mock_engine._states

    controller.inference_loop()
    assert "192.168.1.2" not in mock_engine._states


@patch("pyroengine.core.PyroCameraAPIClient")
def test_same_cached_night_frame_counts_once(mock_client_class, mock_engine):
    """The loop runs faster than the patrol refreshes the cache: one infrared capture read on
    several passes is still one isolated frame and must not drop an ongoing event."""
    camera_data = {"192.168.1.2": {"name": "cam2", "type": "static"}}
    mock_client = mock_client_class.return_value
    mock_client.get_stream_status.return_value = {"active_streams": 0}
    mock_client.get_latest_image.return_value = Image.new("RGB", (100, 100), (255, 255, 255))
    mock_engine._states = {"192.168.1.2": {"ongoing": True}}
    controller = SystemController(mock_engine, camera_data, "http://fake.url")

    for _ in range(NIGHT_FRAMES_BEFORE_RESET + 1):
        controller.inference_loop()

    assert controller._night_frames["192.168.1.2"] == 1
    assert "192.168.1.2" in mock_engine._states


@patch("pyroengine.core.PyroCameraAPIClient")
def test_unchanged_frame_still_sends_heartbeat(mock_client_class, mock_engine):
    """A URL camera serving the same JPEG for minutes still answers and must not look offline."""
    camera_data = {"192.168.1.2": {"name": "cam2", "type": "static"}}
    mock_client = mock_client_class.return_value
    mock_client.get_stream_status.return_value = {"active_streams": 0}
    mock_client.get_latest_image.return_value = Image.new("RGB", (100, 100), (255, 200, 200))
    controller = SystemController(mock_engine, camera_data, "http://fake.url")

    controller.inference_loop()  # same frame as the warm-up pass in __init__

    assert mock_engine.predict.call_count == 1
    mock_engine.maybe_heartbeat.assert_called_once_with("192.168.1.2")


@patch("pyroengine.core.PyroCameraAPIClient")
def test_heartbeat_written_while_stream_active(mock_client_class, mock_engine, mock_camera_data, tmp_path):
    """A stream longer than the healthcheck window must not mark the engine unhealthy."""
    mock_client_class.return_value.get_stream_status.return_value = {"active_streams": 1}
    heartbeat = tmp_path / "heartbeat"

    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url", heartbeat_file=str(heartbeat))
    heartbeat.unlink(missing_ok=True)

    controller.inference_loop()

    assert heartbeat.exists()
