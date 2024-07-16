import datetime
import os
from unittest.mock import patch

import numpy as np

from pyroengine.vision import Classifier


def get_creation_date(file_path):
    if os.path.exists(file_path):

        # For Unix-like systems
        stat = os.stat(file_path)
        try:
            creation_time = stat.st_birthtime
        except AttributeError:
            # On Unix, use the last modification time as a fallback
            creation_time = stat.st_mtime

        creation_date = datetime.datetime.fromtimestamp(creation_time)
        return creation_date
    else:
        return None


def test_classifier(tmpdir_factory, mock_wildfire_image):
    print("test_classifier")
    folder = str(tmpdir_factory.mktemp("engine_cache"))

    # Instantiate the ONNX model
    model = Classifier(model_folder=folder)
    # Check inference
    out = model(mock_wildfire_image)
    assert out.shape[1] == 5
    conf = np.max(out[:, 4])
    assert 0 <= conf <= 1

    # Test onnx model
    model = Classifier(model_folder=folder, format="onnx")
    model_path = os.path.join(folder, "yolov8s.onnx")
    assert os.path.isfile(model_path)

    # Test mask
    mask = np.ones((384, 640))
    out = model(mock_wildfire_image, mask)
    assert out.shape == (1, 5)

    mask = np.zeros((384, 640))
    out = model(mock_wildfire_image, mask)
    assert out.shape == (0, 5)

    # Test dl pt model
    _ = Classifier(model_folder=folder, format="pt")
    model_path = os.path.join(folder, "yolov8s.pt")
    assert os.path.isfile(model_path)

    # Test dl ncnn model
    with patch.object(Classifier, "is_arm_architecture", return_value=True):
        _ = Classifier(model_folder=folder)
        model_path = os.path.join(folder, "yolov8s_ncnn_model")
        assert os.path.isdir(model_path)


def test_download(tmpdir_factory):
    print("test_classifier")
    folder = str(tmpdir_factory.mktemp("engine_cache"))

    # Instantiate the ONNX model
    _ = Classifier(model_folder=folder)

    model_path = os.path.join(folder, "yolov8s.onnx")
    model_creation_date = get_creation_date(model_path)

    # No download if exist
    _ = Classifier(model_folder=folder)
    model_creation_date2 = get_creation_date(model_path)
    assert model_creation_date == model_creation_date2

    # Download if does not exist
    os.remove(model_path)
    _ = Classifier(model_folder=folder)
    model_creation_date3 = get_creation_date(model_path)
    print(model_creation_date, model_creation_date3)
    assert model_creation_date != model_creation_date3

    # Download if sha is not the same
    with patch.object(Classifier, "get_sha", return_value="sha12"):
        _ = Classifier(model_folder=folder)
        model_creation_date4 = get_creation_date(model_path)
        print(model_creation_date, model_creation_date3)
        assert model_creation_date4 != model_creation_date3
