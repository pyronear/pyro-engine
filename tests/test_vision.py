import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyroengine.vision import Classifier

METADATA_PATH = "data/model_metadata.json"
model_path = "data/model.onnx"
sha = "12b9b5728dfa2e60502dcde2914bfdc4e9378caa57611c567a44cdd6228838c2"


def custom_isfile_false(path):
    if path == model_path:
        return False  # or True based on your test case
    return True  # Default behavior for other paths


def custom_isfile_true(path):
    if path == model_path:
        return True  # or True based on your test case
    return True  # Default behavior for other paths


# Test for the case : the model doesn't exist
def test_classifier(mock_wildfire_image):
    print("test_classifier")
    with patch("os.path.isfile", side_effect=custom_isfile_false):
        # Instantiate the ONNX model
        model = Classifier()
        # Check preprocessing
        out = model.preprocess_image(mock_wildfire_image, (1024, 576))
        assert isinstance(out, np.ndarray) and out.dtype == np.float32
        assert out.shape == (1, 3, 576, 1024)
        # Check inference
        out = model(mock_wildfire_image)
        assert out.shape == (1, 5)
        conf = np.max(out[:, 4])
        assert conf >= 0 and conf <= 1

        # Test mask
        mask = np.ones((1024, 576))
        out = model(mock_wildfire_image, mask)
        assert out.shape == (1, 5)

        mask = np.zeros((1024, 1024))
        out = model(mock_wildfire_image, mask)
        assert out.shape == (0, 5)
        os.remove(model_path)
        os.remove(METADATA_PATH)


# Test that the model is not loaded
def test_no_download():
    print("test_no_download")
    data = {"sha256": sha}
    with patch("os.path.isfile", side_effect=custom_isfile_true):
        with patch("pyroengine.vision.Classifier.load_metadata", return_value=data):
            with patch("onnxruntime.InferenceSession", return_value=None):
                Classifier()
    assert os.path.isfile(model_path) is False


# Test if sha are not the same
@patch("pyroengine.vision.urlretrieve")
@patch("pyroengine.vision.DownloadProgressBar")
def test_sha_inequality(mock_download_progress, mock_urlretrieve):
    print("test_sha_inequality")
    data = {"sha256": "falsesha"}

    # Mock urlretrieve to create a fake file
    def fake_urlretrieve(url, filename, reporthook=None):
        with open(filename, "w") as f:
            f.write("fake model content")

    mock_urlretrieve.side_effect = fake_urlretrieve
    # Mock the DownloadProgressBar context manager
    mock_progress_bar_instance = MagicMock()
    mock_download_progress.return_value.__enter__.return_value = mock_progress_bar_instance

    with patch("os.path.isfile", side_effect=custom_isfile_true):
        with patch("pyroengine.vision.Classifier.load_metadata", return_value=data):
            with patch(
                "pyroengine.vision.Classifier.get_sha",
                return_value=sha,
            ):
                with patch("onnxruntime.InferenceSession", return_value=None):
                    with patch("os.remove", return_value=True):
                        model = Classifier()

    assert os.path.isfile(model_path) is True
    assert model.load_metadata("non_existent_metadata.json") is None
    os.remove(model_path)
    os.remove(METADATA_PATH)


# Test for raising ValueError if expected_sha256 is not found
def test_raise_value_error_if_no_sha256():
    print("test_raise_value_error_if_no_sha256")
    with patch("pyroengine.vision.Classifier.get_sha", return_value=""):
        with pytest.raises(
            ValueError, match="SHA256 hash for the model file not found in the Hugging Face model metadata."
        ):
            Classifier(model_path="non_existent_model.onnx")
