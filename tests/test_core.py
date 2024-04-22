from unittest.mock import MagicMock

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

    # This should not raise an error as the camera is mocked
    controller.capture_and_predict(0)
