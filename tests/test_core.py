from multiprocessing import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest
from PIL import Image

from pyroengine.core import SystemController


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.9]])  # Mock prediction output
    return engine


@pytest.fixture
def mock_cameras():
    camera = MagicMock()
    camera.capture.return_value = Image.new("RGB", (100, 100))  # Mock captured image
    camera.cam_type = "static"
    camera.ip_address = "192.168.1.1"
    return [camera]


@pytest.fixture
def system_controller(mock_engine, mock_cameras):
    return SystemController(engine=mock_engine, cameras=mock_cameras)


def test_capture_images(system_controller):
    capture_queue = Queue(maxsize=10)
    system_controller.day_time = True  # Ensure it's day time

    with patch("pyroengine.core.is_day_time", return_value=True):
        # Run the capture_images method for a limited time
        system_controller.capture_images(capture_queue, capture_interval=1, run_for_seconds=2)

        # Check the size of the queue
        print(f"Queue size: {capture_queue.qsize()}")

        # Check if the image was captured and put in the queue
        assert not capture_queue.empty()
        frame, cam_id = capture_queue.get_nowait()
        assert cam_id == "192.168.1.1"
        assert isinstance(frame, Image.Image)


def test_run_predictions(system_controller):
    capture_queue = Queue(maxsize=10)
    prediction_queue = Queue(maxsize=10)

    # Put a mock frame in the capture queue
    mock_frame = Image.new("RGB", (100, 100))
    capture_queue.put((mock_frame, "192.168.1.1"))

    # Run the run_predictions method for a limited time
    system_controller.run_predictions(capture_queue, prediction_queue, run_for_seconds=2)

    # Check if the prediction was made and put in the prediction queue
    assert not prediction_queue.empty()
    preds, frame, cam_id = prediction_queue.get_nowait()
    assert preds.shape == (1, 5)
    assert np.array_equal(preds, np.array([[0.1, 0.2, 0.3, 0.4, 0.9]]))
    assert isinstance(frame, Image.Image)
    assert cam_id == "192.168.1.1"


def test_process_alerts(system_controller, mock_engine):
    prediction_queue = Queue(maxsize=10)

    # Put a mock prediction in the prediction queue
    mock_frame = Image.new("RGB", (100, 100))
    mock_preds = np.array([[0.1, 0.2, 0.3, 0.4, 0.9]])
    prediction_queue.put((mock_preds, mock_frame, "192.168.1.1"))

    # Run the process_alerts method for a limited time
    system_controller.process_alerts(prediction_queue, run_for_seconds=2)

    # Check if the alert was processed
    assert mock_engine.process_prediction.called
    call_args = mock_engine.process_prediction.call_args[0]
    npt.assert_array_equal(call_args[0], mock_preds)  # Compare numpy arrays
    assert call_args[1] == mock_frame
    assert call_args[2] == "192.168.1.1"


def test_process_alerts_empty_queue(system_controller, mock_engine):
    prediction_queue = Queue(maxsize=10)

    # Run the process_alerts method with an empty queue
    system_controller.process_alerts(prediction_queue, run_for_seconds=2)

    # Ensure process_prediction was never called
    mock_engine.process_prediction.assert_not_called()


def test_process_alerts_exception_handling(system_controller, mock_engine):
    prediction_queue = Queue(maxsize=10)

    # Put a mock prediction in the prediction queue that will cause an exception
    mock_frame = Image.new("RGB", (100, 100))
    mock_preds = np.array([[0.1, 0.2, 0.3, 0.4, 0.9]])
    prediction_queue.put((mock_preds, mock_frame, "192.168.1.1"))

    # Make process_prediction raise an exception
    mock_engine.process_prediction.side_effect = Exception("Test exception")

    # Run the process_alerts method for a limited time
    system_controller.process_alerts(prediction_queue, run_for_seconds=2)

    # Ensure process_prediction was called and the exception was handled
    assert mock_engine.process_prediction.called


def test_repr_method(system_controller):
    repr_str = repr(system_controller)
    # Check if the representation is a string
    assert isinstance(repr_str, str)
