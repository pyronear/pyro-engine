import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pyroengine.core import SystemController, capture_camera_image, is_day_time


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.predict.return_value = None  # Mock predict method
    return engine


@pytest.fixture
def mock_cameras():
    camera = MagicMock()
    camera.capture.return_value = Image.new("RGB", (100, 100))  # Mock captured image
    camera.cam_type = "static"
    camera.ip_address = "192.168.1.1"
    return [camera]


@pytest.fixture
def mock_cameras_ptz():
    camera = MagicMock()
    camera.capture.return_value = Image.new("RGB", (100, 100))  # Mock captured image
    camera.cam_type = "ptz"
    camera.cam_poses = [1, 2]
    camera.ip_address = "192.168.1.1"
    return [camera]


@pytest.fixture
def system_controller(mock_engine, mock_cameras):
    return SystemController(engine=mock_engine, cameras=mock_cameras)


@pytest.fixture
def system_controller_ptz(mock_engine, mock_cameras_ptz):
    return SystemController(engine=mock_engine, cameras=mock_cameras_ptz)


def test_is_day_time_ir_strategy(mock_wildfire_image):
    # Use day image
    assert is_day_time(None, mock_wildfire_image, "ir")

    # Create a grayscale image to simulate night image
    frame = Image.new("RGB", (100, 100), (255, 255, 255))
    assert not is_day_time(None, frame, "ir")


def test_is_day_time_time_strategy(tmp_path):
    cache = tmp_path
    with open(cache / "sunset_sunrise.txt", "w") as f:
        f.write("06:00\n18:00\n")

    # Mock datetime to return a specific time within day hours
    with patch("pyroengine.core.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 6, 17, 10, 0, 0)
        mock_datetime.strptime = datetime.strptime  # Ensure strptime works as expected
        assert is_day_time(cache, None, "time")

    # Mock datetime to return a specific time outside day hours
    with patch("pyroengine.core.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 6, 17, 20, 0, 0)
        mock_datetime.strptime = datetime.strptime  # Ensure strptime works as expected
        assert not is_day_time(cache, None, "time")


@pytest.mark.asyncio
async def test_capture_images(system_controller):
    queue = asyncio.Queue(maxsize=10)
    await system_controller.capture_images(queue)

    assert queue.qsize() == 1
    cam_id, frame = await queue.get()  # Use timeout to wait for the item
    assert cam_id == "192.168.1.1"
    assert isinstance(frame, Image.Image)


@pytest.mark.asyncio
async def test_capture_images_ptz(system_controller_ptz):
    queue = asyncio.Queue(maxsize=10)
    await system_controller_ptz.capture_images(queue)

    assert queue.qsize() == 2
    cam_id, frame = await queue.get()  # Use timeout to wait for the item
    assert cam_id == "192.168.1.1_1"
    assert isinstance(frame, Image.Image)


@pytest.mark.asyncio
async def test_analyze_stream(system_controller):
    queue = asyncio.Queue()
    mock_frame = Image.new("RGB", (100, 100))
    await queue.put(("192.168.1.1", mock_frame))

    analyze_task = asyncio.create_task(system_controller.analyze_stream(queue))
    await queue.put(None)  # Signal the end of the stream
    await analyze_task

    system_controller.engine.predict.assert_called_once_with(mock_frame, "192.168.1.1")


@pytest.mark.asyncio
async def test_capture_images_method(system_controller):
    with patch("pyroengine.core.capture_camera_image", new_callable=AsyncMock) as mock_capture:
        queue = asyncio.Queue()
        await system_controller.capture_images(queue)

        for camera in system_controller.cameras:
            mock_capture.assert_any_call(camera, queue)
        assert mock_capture.call_count == len(system_controller.cameras)


@pytest.mark.asyncio
async def test_analyze_stream_method(system_controller):
    queue = asyncio.Queue()
    mock_frame = Image.new("RGB", (100, 100))
    await queue.put(("192.168.1.1", mock_frame))
    await queue.put(None)  # Signal the end of the stream

    await system_controller.analyze_stream(queue)

    system_controller.engine.predict.assert_called_once_with(mock_frame, "192.168.1.1")


def test_check_day_time(system_controller):
    with patch("pyroengine.core.is_day_time", return_value=True) as mock_is_day_time:
        system_controller.check_day_time()
        assert system_controller.day_time is True
        mock_is_day_time.assert_called_once()

    with patch("pyroengine.core.is_day_time", return_value=False) as mock_is_day_time:
        system_controller.check_day_time()
        assert system_controller.day_time is False
        mock_is_day_time.assert_called_once()

    with patch("pyroengine.core.is_day_time", side_effect=Exception("Error in is_day_time")) as mock_is_day_time, patch(
        "pyroengine.core.logging.exception"
    ) as mock_logging_exception:
        system_controller.check_day_time()
        mock_is_day_time.assert_called_once()
        mock_logging_exception.assert_called_once_with("Exception during initial day time check: Error in is_day_time")


def test_repr_method(system_controller):
    repr_str = repr(system_controller)
    # Check if the representation is a string
    assert isinstance(repr_str, str)


def test_repr_method_no_cameras(mock_engine):
    system_controller = SystemController(engine=mock_engine, cameras=[])
    repr_str = repr(system_controller)
    # Check if the representation is a string
    assert isinstance(repr_str, str)
