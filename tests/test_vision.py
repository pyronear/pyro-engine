import datetime
import os
import pathlib
from unittest.mock import patch

import numpy as np

from pyroengine.vision import Classifier


def get_creation_date(file_path):
    if pathlib.Path(file_path).exists():
        # For Unix-like systems
        stat = os.stat(file_path)
        try:
            creation_time = stat.st_birthtime
        except AttributeError:
            # On Unix, use the last modification time as a fallback
            creation_time = stat.st_mtime

        creation_date = datetime.datetime.fromtimestamp(creation_time)
        return creation_date
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
    model_path = os.path.join(folder, "yolov11s.onnx")
    assert pathlib.Path(model_path).is_file()

    # Test occlusion mask
    out = model(mock_wildfire_image, {})
    assert out.shape == (1, 5)

    occlusion_bboxes = {"2025-05-28 15:49:17": [0.2, 0.3, 0.4, 0.5]}
    out = model(mock_wildfire_image, occlusion_bboxes)
    assert out.shape == (1, 5)

    occlusion_bboxes = {"2025-05-28 15:49:17": [0.00621796, 0.5494792, 0.02899933, 0.6085069]}
    out = model(mock_wildfire_image, occlusion_bboxes)
    assert out.shape == (0, 5)


def test_download(tmpdir_factory):
    print("test_classifier")
    folder = str(tmpdir_factory.mktemp("engine_cache"))
    # Instantiate ncnn model
    _ = Classifier(model_folder=folder, format="ncnn")

    # Instantiate the ONNX model
    _ = Classifier(model_folder=folder, format="onnx")

    model_path = os.path.join(folder, "yolov11s.onnx")
    model_creation_date = get_creation_date(model_path)

    # No download if exist
    _ = Classifier(model_folder=folder, format="onnx")
    model_creation_date2 = get_creation_date(model_path)
    assert model_creation_date == model_creation_date2

    # Download if does not exist
    pathlib.Path(model_path).unlink()
    _ = Classifier(model_folder=folder, format="onnx")
    model_creation_date3 = get_creation_date(model_path)
    assert model_creation_date != model_creation_date3

    # Download if sha is not the same
    with patch.object(Classifier, "get_sha", return_value="sha12"):
        _ = Classifier(model_folder=folder, format="onnx")
        model_creation_date4 = get_creation_date(model_path)
        assert model_creation_date4 != model_creation_date3
