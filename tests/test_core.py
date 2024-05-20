from unittest.mock import MagicMock, patch

from pyroengine.core import SystemController
from pyroengine.engine import Engine


def test_systemcontroller_with_mock_camera(tmpdir_factory):
    # Setup
    folder = str(tmpdir_factory.mktemp("engine_cache"))
    engine = Engine(cache_folder=folder)

    # Creating a mock camera
    mock_camera = MagicMock()
    mock_camera.capture.return_value = "Mock Image"
    mock_camera.ip_address = "192.168.1.1"
    cams = [mock_camera]
    controller = SystemController(engine, cams)

    # Mocking the methods of SystemController that would run long processes
    with patch.object(controller, "capture_images", return_value=None) as mock_capture_images, patch.object(
        controller, "run_predictions", return_value=None
    ) as mock_run_predictions, patch.object(controller, "process_alerts", return_value=None) as mock_process_alerts:

        # Invoke the methods
        controller.capture_images(MagicMock(), 1)
        controller.run_predictions(MagicMock(), MagicMock())
        controller.process_alerts(MagicMock())

        # Assert the methods were called
        mock_capture_images.assert_called_once()
        mock_run_predictions.assert_called_once()
        mock_process_alerts.assert_called_once()
