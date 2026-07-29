import logging
import pathlib
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

from pyroengine.core import SystemController, is_day_time


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

    with caplog.at_level(logging.INFO, logger="pyroengine.core"):
        controller.inference_loop()

    info_lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_lines) == 1
    assert "analyzed=2" in info_lines[0]
    assert "positive=0" in info_lines[0]
    # The engine healthcheck greps the log for "confidence"
    assert "max_confidence" in info_lines[0]


@patch("pyroengine.core.PyroCameraAPIClient")
def test_inference_loop_summary_counts_failures(mock_client_class, mock_engine, mock_camera_data, caplog):
    """Poses that fail to capture are reported in the round summary."""
    mock_client = mock_client_class.return_value
    mock_client.get_latest_image.side_effect = Exception("camera down")
    mock_client.get_stream_status.return_value = {"active_streams": 0}

    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url")

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
def test_inference_loop_skips_when_stream_active(mock_client_class, mock_engine, mock_camera_data):
    mock_client = mock_client_class.return_value
    mock_client.get_stream_status.return_value = {"active_streams": 1}

    controller = SystemController(mock_engine, mock_camera_data, "http://fake.url")
    controller.inference_loop()

    assert not mock_client.get_latest_image.called
    assert not mock_engine.predict.called
