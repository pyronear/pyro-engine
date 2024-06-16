import time
from multiprocessing import Queue
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pyroengine.core import SystemController, capture_camera_image


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


def test_capture_images(system_controller):
    queue = Queue(maxsize=10)
    for camera in system_controller.cameras:
        capture_camera_image((camera, queue))

    print(f"Queue size after capture: {queue.qsize()}")
    assert queue.qsize() == 1
    cam_id, frame = queue.get(timeout=1)  # Use timeout to wait for the item
    print(f"Camera ID: {cam_id}, Frame size: {frame.size}")
    assert cam_id == "192.168.1.1"
    assert isinstance(frame, Image.Image)


def test_capture_images_ptz(system_controller_ptz):
    queue = Queue(maxsize=10)
    for camera in system_controller_ptz.cameras:
        capture_camera_image((camera, queue))

    # Retry logic to account for potential timing issues
    retries = 10
    while retries > 0 and queue.qsize() < 2:
        time.sleep(0.1)
        retries -= 1

    print(f"Queue size after capture: {queue.qsize()}")
    assert queue.qsize() == 2
    cam_id, frame = queue.get(timeout=1)  # Use timeout to wait for the item
    print(f"Camera ID: {cam_id}, Frame size: {frame.size}")
    assert cam_id == "192.168.1.1_1"
    assert isinstance(frame, Image.Image)


def test_analyze_stream(system_controller):
    mock_frame = Image.new("RGB", (100, 100))
    cam_id = "192.168.1.1"
    system_controller.analyze_stream(mock_frame, cam_id)
    system_controller.engine.predict.assert_called_once_with(mock_frame, cam_id)


def test_run(system_controller):
    with patch.object(system_controller, "capture_images", return_value=Queue()), patch.object(
        system_controller, "analyze_stream"
    ), patch.object(system_controller.engine, "_process_alerts"), patch("signal.signal"), patch("signal.alarm"), patch(
        "time.sleep", side_effect=InterruptedError
    ):  # Mock sleep to exit the loop

        try:
            system_controller.run(period=2)
        except InterruptedError:
            pass


def test_run_no_images(system_controller):
    with patch.object(system_controller, "capture_images", return_value=Queue()), patch.object(
        system_controller, "analyze_stream"
    ) as mock_analyze_stream, patch.object(system_controller.engine, "_process_alerts"), patch("signal.signal"), patch(
        "signal.alarm"
    ), patch(
        "time.sleep", side_effect=InterruptedError
    ):  # Mock sleep to exit the loop

        try:
            system_controller.run(period=2)
        except InterruptedError:
            pass

        mock_analyze_stream.assert_not_called()


def test_run_capture_exception(system_controller):
    with patch.object(system_controller, "capture_images", side_effect=Exception("Capture error")), patch.object(
        system_controller.engine, "_process_alerts"
    ), patch("signal.signal"), patch("signal.alarm"), patch(
        "time.sleep", side_effect=InterruptedError
    ):  # Mock sleep to exit the loop

        try:
            system_controller.run(period=2)
        except InterruptedError:
            pass


def test_capture_camera_image_exception():
    queue = Queue(maxsize=10)
    camera = MagicMock()
    camera.cam_type = "static"
    camera.ip_address = "192.168.1.1"
    camera.capture.side_effect = Exception("Capture error")

    capture_camera_image((camera, queue))

    assert queue.empty()


def test_repr_method(system_controller):
    repr_str = repr(system_controller)
    # Check if the representation is a string
    assert isinstance(repr_str, str)


def test_repr_method_no_cameras(mock_engine):
    system_controller = SystemController(engine=mock_engine, cameras=[])
    repr_str = repr(system_controller)
    # Check if the representation is a string
    assert isinstance(repr_str, str)


def test_capture_camera_image():
    queue = Queue(maxsize=10)
    camera = MagicMock()
    camera.cam_type = "static"
    camera.ip_address = "192.168.1.1"
    camera.capture.return_value = Image.new("RGB", (100, 100))  # Mock captured image

    capture_camera_image((camera, queue))

    print(f"Queue size after capture: {queue.qsize()}")
    assert not queue.empty()
    cam_id, frame = queue.get(timeout=1)  # Use timeout to wait for the item
    print(f"Camera ID: {cam_id}, Frame size: {frame.size}")
    assert cam_id == "192.168.1.1"
    assert isinstance(frame, Image.Image)


def test_capture_camera_image_ptz():
    queue = Queue(maxsize=10)
    camera = MagicMock()
    camera.cam_type = "ptz"
    camera.cam_poses = [1, 2]
    camera.ip_address = "192.168.1.1"
    camera.capture.return_value = Image.new("RGB", (100, 100))  # Mock captured image

    capture_camera_image((camera, queue))

    # Retry logic to account for potential timing issues
    retries = 10
    while retries > 0 and queue.qsize() < 2:
        time.sleep(0.1)
        retries -= 1

    print(f"Queue size after capture: {queue.qsize()}")
    assert queue.qsize() == 2
    cam_id, frame = queue.get(timeout=1)  # Use timeout to wait for the item
    print(f"Camera ID: {cam_id}, Frame size: {frame.size}")
    assert cam_id == "192.168.1.1_1"
    assert isinstance(frame, Image.Image)
