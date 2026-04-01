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


def test_predictor_verbose_false_no_logs(mock_wildfire_image, caplog):
    """verbose=False suppresses pyro_predictor log output."""
    predictor = Predictor(nb_consecutive_frames=2, verbose=False)
    with caplog.at_level(logging.INFO, logger="pyro_predictor"):
        predictor.predict(mock_wildfire_image)
    assert caplog.records == []


def test_predictor_verbose_true_emits_logs(mock_wildfire_image, caplog):
    """verbose=True (default) emits INFO logs."""
    predictor = Predictor(nb_consecutive_frames=2, verbose=True)
    with caplog.at_level(logging.INFO, logger="pyro_predictor"):
        predictor.predict(mock_wildfire_image)
    assert any(r.levelno == logging.INFO for r in caplog.records)


def test_classifier_verbose_false_no_logs(tmpdir_factory, caplog):
    """Classifier verbose=False suppresses log output during init."""
    folder = str(tmpdir_factory.mktemp("cls_cache"))
    with caplog.at_level(logging.INFO, logger="pyro_predictor"):
        Classifier(model_folder=folder, format="onnx", verbose=False)
    assert caplog.records == []
