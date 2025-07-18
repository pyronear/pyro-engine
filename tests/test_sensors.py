from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from requests.exceptions import ConnectTimeout

from pyroengine.sensors import ReolinkCamera


def test_reolinkcamera_connect_timeout():
    # Mock the requests.get method to raise a ConnectTimeout exception
    with patch("requests.get", side_effect=ConnectTimeout), patch("requests.post"):
        camera = ReolinkCamera("192.168.99.99", "login", "pwd", "static")
        result = camera.capture()
        # Assert that the capture method returns None when a ConnectTimeout occurs
        assert result is None


def test_reolinkcamera_success(mock_wildfire_stream):
    mock_get = MagicMock()
    mock_get.status_code = 200
    mock_get.content = mock_wildfire_stream

    mock_post = MagicMock()
    mock_post.status_code = 200
    mock_post.json.return_value = [{"code": 0}]

    with patch("requests.get", return_value=mock_get), patch("requests.post", return_value=mock_post):
        camera = ReolinkCamera("192.168.99.99", "login", "pwd", "static")
        result = camera.capture()
        assert isinstance(result, Image.Image)


def test_move_camera_success():
    # Mock the response of requests.post to return a successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"code": 0}]

    with patch("requests.post", return_value=mock_response):
        camera = ReolinkCamera("192.168.99.99", "login", "pwd", "ptz")
        camera.move_camera("Left", speed=2, idx=1)
        # Assert that a successful operation logs the correct message
        assert mock_response.json.call_count == 1


def test_move_camera_failure():
    # Mock the response of requests.post to return a failed response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"code": 1, "error": "Some error"}]

    with patch("requests.post", return_value=mock_response):
        camera = ReolinkCamera("192.168.99.99", "login", "pwd", "ptz")
        camera.move_camera("Left", speed=2, idx=1)
        # Assert that a failed operation logs an error message
        assert mock_response.json.call_count == 1


def test_get_ptz_preset_success():
    # Mock the response of requests.post to return a successful response with preset data
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"code": 0, "value": {"PtzPreset": [{"id": 1, "name": "preset1", "enable": 1}]}}]

    with patch("requests.post", return_value=mock_response):
        camera = ReolinkCamera("192.168.99.99", "login", "pwd", "ptz")
        presets = camera.get_ptz_preset()
        # Assert that the get_ptz_preset method returns the correct presets
        assert presets == [{"id": 1, "name": "preset1", "enable": 1}]


def test_set_ptz_preset_success():
    # Mock the response of requests.post to return a successful response for setting preset
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"code": 0}]

    with patch("requests.post", return_value=mock_response):
        camera = ReolinkCamera("192.168.99.99", "login", "pwd", "ptz")
        camera.set_ptz_preset(idx=1)
        # Assert that the set_ptz_preset method was called successfully
        assert mock_response.json.call_count == 1


def test_set_ptz_preset_no_slots():
    # Mock the response of requests.post to return no available slots for presets
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"code": 0, "value": {"PtzPreset": [{"id": 1, "name": "preset1", "enable": 1}]}}]

    with patch("requests.post", return_value=mock_response):
        camera = ReolinkCamera("192.168.99.99", "login", "pwd", "ptz")
        with pytest.raises(ValueError, match="No available slots for new presets."):
            camera.set_ptz_preset()


def test_move_in_seconds():
    # Mock the move_camera method and requests.post to avoid real HTTP call in __init__
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"code": 0}]

    with (
        patch("requests.post", return_value=mock_response),
        patch.object(ReolinkCamera, "move_camera") as mock_move_camera,
    ):
        camera = ReolinkCamera("192.168.99.99", "login", "pwd", "ptz")
        camera.move_in_seconds(1, operation="Right", speed=2)
        # Assert that the move_camera method was called with the correct arguments
        mock_move_camera.assert_any_call("Right", 2)
        mock_move_camera.assert_any_call("Stop")


def test_reboot_camera_success():
    # Mock the response of requests.post to return a successful response for reboot
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"code": 0}]

    with patch("requests.post", return_value=mock_response):
        camera = ReolinkCamera("192.168.99.99", "login", "pwd", "static")
        response = camera.reboot_camera()
        # Assert that the reboot_camera method was called successfully
        assert mock_response.json.call_count == 1
        assert response == mock_response.json.return_value


def test_get_auto_focus_success():
    # Mock the response of requests.post to return a successful response with autofocus data
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"code": 0, "value": {"AutoFocus": [{"channel": 0, "disable": 0}]}}]

    with patch("requests.post", return_value=mock_response):
        camera = ReolinkCamera("192.168.99.99", "login", "pwd", "static")
        response = camera.get_auto_focus()
        # Assert that the get_auto_focus method returns the correct data
        assert response == mock_response.json.return_value


def test_set_auto_focus_success():
    # Mock the response of requests.post to return a successful response for setting autofocus
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"code": 0}]

    with patch("requests.post", return_value=mock_response):
        camera = ReolinkCamera("192.168.99.99", "login", "pwd", "static")
        response = camera.set_auto_focus(disable=True)
        # Assert that the set_auto_focus method was called successfully
        assert mock_response.json.call_count == 1
        assert response == mock_response.json.return_value


def test_start_zoom_focus_success():
    # Mock the response of requests.post to return a successful response for starting zoom focus
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"code": 0}]

    with patch("requests.post", return_value=mock_response):
        camera = ReolinkCamera("192.168.99.99", "login", "pwd", "ptz")
        response = camera.start_zoom_focus(position=100)
        # Assert that the start_zoom_focus method was called successfully
        assert mock_response.json.call_count == 1
        assert response == mock_response.json.return_value


def test_set_manual_focus_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"code": 0}]

    with patch("requests.post", return_value=mock_response) as mock_post:
        camera = ReolinkCamera("192.168.99.99", "user", "pass")
        response = camera.set_manual_focus(position=300)

        assert mock_post.called
        assert mock_response.json.call_count == 1
        assert response == mock_response.json.return_value


def test_get_focus_level_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "code": 0,
            "value": {"ZoomFocus": {"focus": {"pos": 150}, "zoom": {"pos": 80}}},
        }
    ]

    with patch("requests.post", return_value=mock_response):
        camera = ReolinkCamera("192.168.99.99", "user", "pass")
        result = camera.get_focus_level()

        assert result == {"focus": 150, "zoom": 80}


def test_focus_finder_success():
    from unittest.mock import MagicMock

    import numpy as np
    from PIL import Image

    # Mapping position -> sharpness
    sharpness_map = {
        720: 5.0,
        721: 10.0,
        722: 15.0,
        723: 20.0,
        724: 25.0,  # Peak
        725: 24.0,
        726: 22.0,
    }

    # Keep track of focus positions requested
    called_positions = []

    def mock_set_manual_focus(pos):
        called_positions.append(pos)

    def mock_capture():
        # Dummy image, content doesn't matter because we mock sharpness
        return Image.fromarray((np.random.rand(100, 100) * 255).astype(np.uint8))

    def mock_sharpness(image):
        pos = called_positions[-1]
        return sharpness_map.get(pos, 0.0)

    camera = ReolinkCamera("192.168.1.1", "user", "pass", "ptz")
    camera.focus_position = 720
    camera.cam_type = "ptz"
    camera.capture = MagicMock(side_effect=mock_capture)
    camera.set_manual_focus = MagicMock(side_effect=mock_set_manual_focus)
    camera._measure_sharpness = mock_sharpness  # must be a method

    best_focus = camera.focus_finder()

    assert isinstance(best_focus, int)
    assert best_focus == 724  # Peak sharpness
