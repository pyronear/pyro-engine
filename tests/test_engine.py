import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import onnx
import pytest
from dotenv import load_dotenv
from PIL import Image

from pyroengine.engine import Engine


def test_engine_offline(tmpdir_factory, mock_wildfire_image, mock_forest_image):
    # Cache
    folder = str(tmpdir_factory.mktemp("engine_cache"))

    engine = Engine(cache_folder=folder)

    # Cache saving
    _ts = datetime.now().isoformat()
    engine._stage_alert(mock_wildfire_image, 0, datetime.now().isoformat(), bboxes="dummy")
    assert len(engine._alerts) == 1
    assert engine._alerts[0]["ts"] < datetime.now().isoformat() and _ts < engine._alerts[0]["ts"]
    assert engine._alerts[0]["media_id"] is None
    assert engine._alerts[0]["alert_id"] is None

    # inference
    engine = Engine(nb_consecutive_frames=4, cache_folder=folder, save_captured_frames=True)
    out = engine.predict(mock_forest_image)
    assert isinstance(out, float) and 0 <= out <= 1
    assert len(engine._states["-1"]["last_predictions"]) == 1
    assert engine._states["-1"]["ongoing"] is False
    assert isinstance(engine._states["-1"]["last_predictions"][0][0], Image.Image)
    assert engine._states["-1"]["last_predictions"][0][1].shape[0] == 0
    assert engine._states["-1"]["last_predictions"][0][1].shape[1] == 5
    assert engine._states["-1"]["last_predictions"][0][3] < datetime.now().isoformat()
    assert engine._states["-1"]["last_predictions"][0][4] is False

    out = engine.predict(mock_wildfire_image)
    assert isinstance(out, float) and 0 <= out <= 1
    assert len(engine._states["-1"]["last_predictions"]) == 2
    assert engine._states["-1"]["ongoing"] is False
    assert isinstance(engine._states["-1"]["last_predictions"][0][0], Image.Image)
    assert engine._states["-1"]["last_predictions"][1][1].shape[0] > 0
    assert engine._states["-1"]["last_predictions"][1][1].shape[1] == 5
    assert engine._states["-1"]["last_predictions"][1][3] < datetime.now().isoformat()
    assert engine._states["-1"]["last_predictions"][1][4] is False

    out = engine.predict(mock_wildfire_image)
    assert isinstance(out, float) and 0 <= out <= 1
    assert len(engine._states["-1"]["last_predictions"]) == 3
    assert engine._states["-1"]["ongoing"] is True
    assert isinstance(engine._states["-1"]["last_predictions"][0][0], Image.Image)
    assert engine._states["-1"]["last_predictions"][2][1].shape[0] > 0
    assert engine._states["-1"]["last_predictions"][2][1].shape[1] == 5
    assert len(engine._states["-1"]["last_predictions"][-1][2][0]) == 5
    assert engine._states["-1"]["last_predictions"][2][3] < datetime.now().isoformat()
    assert engine._states["-1"]["last_predictions"][2][4] is False


def create_dummy_onnx_model(model_path):
    """Creates a small dummy ONNX model."""
    x = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])
    y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])

    node = onnx.helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = onnx.helper.make_graph([node], "dummy_model", [x], [y])

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 10)])
    model.ir_version = 10

    onnx.save(model, model_path)


@pytest.fixture
def dummy_onnx_file():
    """Fixture to create a temporary ONNX file."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmpfile:
        create_dummy_onnx_model(tmpfile.name)
        yield tmpfile.name  # returns file path


def test_valid_model_path(dummy_onnx_file):
    """Tests Engine instanciation with a valid input model_path"""
    instance = Engine(model_path=dummy_onnx_file)
    assert instance.model.format == "onnx"


@pytest.fixture
def invalid_onnx_file():
    """Fixture to create a temporary invalid ONNX file."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmpfile:
        with open(tmpfile.name, "wb") as f:
            f.write(b"Invalid content")
        yield tmpfile.name  # returns file path


def test_invalid_model_content(invalid_onnx_file):
    """Tests Engine instantiation with an invalid ONNX model content."""
    with pytest.raises(RuntimeError, match="Failed to load the ONNX model"):
        # Engine instantiation with an invalid model : Classifier instnaciation should raise an error
        Engine(model_path=invalid_onnx_file)


# mock_isfile is a mock of the os.path.isfile() function which allows to simulate file existence
@patch("os.path.isfile")
def test_nonexistent_model(mock_isfile):
    """Tests Engine instanciation with a non-existent input model_path"""
    mock_isfile.return_value = False  # Simulates file non-existence
    with pytest.raises(ValueError, match=r"Model file not found: .*"):
        Engine(model_path="nonexistent.onnx")


@patch("pathlib.Path.is_file", return_value=True)
def test_invalid_extension(mock_isfile):
    """Tests Engine instanciation with a file format different than .onnx"""
    mock_isfile.return_value = True  # Simulates file existence
    with pytest.raises(
        ValueError,
        match=r"Input model_path should point to an ONNX export but currently is *",
    ):
        Engine(model_path="model.ncnn")


def test_engine_online(tmpdir_factory, mock_wildfire_stream, mock_wildfire_image):
    # Cache
    folder = str(tmpdir_factory.mktemp("engine_cache"))
    # With API
    load_dotenv(Path(__file__).parent.parent.joinpath(".env").absolute())
    api_url = os.environ.get("API_URL")
    cam_creds = {"dummy_cam": (os.environ.get("API_TOKEN"), 0, None)}
    # Skip the API-related tests if the URL is not specified

    if isinstance(api_url, str):
        engine = Engine(
            api_url=api_url,
            conf_thresh=0.01,
            cam_creds=cam_creds,
            nb_consecutive_frames=4,
            frame_saving_period=3,
            cache_folder=folder,
        )
        # Heartbeat
        start_ts = datetime.now(timezone.utc).isoformat()
        response = engine.heartbeat("dummy_cam")
        assert response.status_code // 100 == 2
        json_respone = response.json()
        time.sleep(0.1)
        ts = datetime.now(timezone.utc).isoformat()

        assert start_ts < json_respone["last_active_at"] < ts
        # Send an alert
        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 1
        assert len(engine._alerts) == 0
        assert engine._states["dummy_cam"]["ongoing"] is False

        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 2

        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 3

        assert engine._states["dummy_cam"]["ongoing"] is True
        # Check that a media and an alert have been registered
        engine._process_alerts()
        assert len(engine._alerts) == 0


def test_engine_occlusion(tmpdir_factory, mock_wildfire_stream, mock_wildfire_image):
    # Cache
    folder = str(tmpdir_factory.mktemp("engine_cache"))
    # With API
    load_dotenv(Path(__file__).parent.parent.joinpath(".env").absolute())
    api_url = os.environ.get("API_URL")
    bbox_mask_url = "https://github.com/pyronear/pyro-engine/releases/download/v0.1.1/test_occlusion_bboxes"
    cam_creds = {"dummy_cam": (os.environ.get("API_TOKEN"), 0, bbox_mask_url)}
    # Skip the API-related tests if the URL is not specified

    if isinstance(api_url, str):
        engine = Engine(
            api_url=api_url,
            conf_thresh=0.01,
            cam_creds=cam_creds,
            nb_consecutive_frames=4,
            frame_saving_period=3,
            cache_folder=folder,
        )
        # Heartbeat
        start_ts = datetime.now(timezone.utc).isoformat()
        response = engine.heartbeat("dummy_cam")
        assert response.status_code // 100 == 2
        json_respone = response.json()
        time.sleep(0.1)
        ts = datetime.now(timezone.utc).isoformat()

        assert start_ts < json_respone["last_active_at"] < ts
        # Send an alert
        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 1
        assert len(engine._alerts) == 0
        assert engine._states["dummy_cam"]["ongoing"] is False

        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 2

        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 3

        assert engine._states["dummy_cam"]["ongoing"] is False
