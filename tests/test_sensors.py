from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from requests.exceptions import ConnectTimeout

from pyroengine.sensors import ReolinkCamera


def test_reolinkcamera_connect_timeout():
    # Mock the requests.get method to raise a ConnectTimeout exception
    with patch("requests.get", side_effect=ConnectTimeout):
        camera = ReolinkCamera("192.168.1.1", "login", "pwd", "typeA")
        result = camera.capture()
        # Assert that the capture method returns None when a ConnectTimeout occurs
        assert result is None


def test_reolinkcamera_success(mock_wildfire_stream):
    # Mock the response of requests.get to return a successful response with image data
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = mock_wildfire_stream

    with patch("requests.get", return_value=mock_response):
        camera = ReolinkCamera("192.168.1.1", "login", "pwd", "typeA")
        result = camera.capture()
        # Assert that the capture method returns an Image object
        assert isinstance(result, Image.Image)
