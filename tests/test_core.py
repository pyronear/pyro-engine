from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

from pyroengine.core import SystemController, is_day_time


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.predict.return_value = None
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
    with open(cache / "sunset_sunrise.txt", "w") as f:
        f.write("06:00\n18:00\n")

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
