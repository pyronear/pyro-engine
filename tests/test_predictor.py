import logging

import numpy as np
from pyro_predictor import Classifier, Predictor


def test_predictor_direct_import():
    """Predictor and Classifier are importable directly from pyro_predictor."""
    assert Predictor is not None
    assert Classifier is not None


def test_predictor_offline(mock_wildfire_image, mock_forest_image):
    predictor = Predictor(nb_consecutive_frames=4, verbose=False)

    out = predictor.predict(mock_forest_image)
    assert isinstance(out, float)
    assert 0 <= out <= 1
    assert len(predictor._states["-1"]["last_predictions"]) == 1
    assert predictor._states["-1"]["ongoing"] is False

    out = predictor.predict(mock_wildfire_image)
    assert isinstance(out, float)
    assert 0 <= out <= 1
    assert len(predictor._states["-1"]["last_predictions"]) == 2

    out = predictor.predict(mock_wildfire_image)
    assert isinstance(out, float)
    assert 0 <= out <= 1
    assert predictor._states["-1"]["ongoing"]


def test_predictor_per_camera_state(mock_wildfire_image, mock_forest_image):
    """Each cam_id maintains independent state."""
    predictor = Predictor(nb_consecutive_frames=4, verbose=False)

    predictor.predict(mock_wildfire_image, cam_id="cam_a")
    predictor.predict(mock_forest_image, cam_id="cam_b")

    assert len(predictor._states["cam_a"]["last_predictions"]) == 1
    assert len(predictor._states["cam_b"]["last_predictions"]) == 1
    # cam_a saw wildfire, cam_b saw forest — states are independent
    assert predictor._states["cam_a"]["last_predictions"][0][1].shape[0] > 0
    assert predictor._states["cam_b"]["last_predictions"][0][1].shape[0] == 0


def test_predictor_fake_pred(mock_wildfire_image):
    """fake_pred bypasses model and goes through state update."""
    predictor = Predictor(nb_consecutive_frames=4, verbose=False)

    fake = np.empty((0,))
    out = predictor.predict(mock_wildfire_image, fake_pred=fake)
    assert isinstance(out, float)

    fake = np.array([[0.1, 0.1, 0.2, 0.2, 0.9], [0.3, 0.3, 0.4, 0.4, 0.8]]).T
    out = predictor.predict(mock_wildfire_image, fake_pred=fake)
    assert isinstance(out, float)


def test_predictor_quiet_frame_no_info_logs(mock_wildfire_image, caplog):
    """A frame with no detection stays out of INFO, so a quiet round is silent."""
    predictor = Predictor(nb_consecutive_frames=2)
    with caplog.at_level(logging.INFO, logger="pyro_predictor"):
        predictor.predict(mock_wildfire_image, fake_pred=np.empty((0,)))
    assert caplog.records == []


def test_predictor_detection_emits_info_log(mock_wildfire_image, caplog):
    """A detection is always reported at INFO."""
    predictor = Predictor(nb_consecutive_frames=2)
    fake = np.array([[0.1, 0.1, 0.2, 0.2, 0.9], [0.3, 0.3, 0.4, 0.4, 0.8]]).T
    with caplog.at_level(logging.INFO, logger="pyro_predictor"):
        predictor.predict(mock_wildfire_image, cam_id="cam_a", fake_pred=fake)
    assert any(r.levelno == logging.INFO and "cam_a" in r.getMessage() for r in caplog.records)


def test_predictor_frame_details_at_debug(mock_wildfire_image, caplog):
    """Per-frame details remain available when DEBUG is enabled."""
    predictor = Predictor(nb_consecutive_frames=2)
    with caplog.at_level(logging.DEBUG, logger="pyro_predictor"):
        predictor.predict(mock_wildfire_image, fake_pred=np.empty((0,)))
    assert any(r.levelno == logging.DEBUG for r in caplog.records)


def test_classifier_does_not_configure_root_logger(tmpdir_factory):
    """Importing/constructing the library must not install root handlers."""
    folder = str(tmpdir_factory.mktemp("cls_cache"))
    root_handlers = list(logging.root.handlers)
    Classifier(model_folder=folder, format="onnx", verbose=False)
    assert logging.root.handlers == root_handlers
