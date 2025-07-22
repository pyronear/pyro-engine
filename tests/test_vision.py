import hashlib
import os
import shutil

import numpy as np

from pyroengine.vision import Classifier


def test_classifier(tmpdir_factory, mock_wildfire_image):
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
    model_path = os.path.join(folder, "onnx_cpu_yolo11s", "model.onnx")
    assert os.path.isfile(model_path)

    # Test occlusion mask
    out = model(mock_wildfire_image, {})
    assert out.shape == (1, 5)

    occlusion_bboxes = {"2025-05-28 15:49:17": [0.2, 0.3, 0.4, 0.5]}
    out = model(mock_wildfire_image, occlusion_bboxes)
    assert out.shape == (1, 5)

    occlusion_bboxes = {"2025-05-28 15:49:17": [0.00621796, 0.5494792, 0.02899933, 0.6085069]}
    out = model(mock_wildfire_image, occlusion_bboxes)
    assert out.shape == (0, 5)


def sha256sum(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_download(tmpdir_factory):
    folder = str(tmpdir_factory.mktemp("engine_cache"))

    # First download
    _ = Classifier(model_folder=folder, format="onnx")
    model_path = os.path.join(folder, "onnx_cpu_yolo11s/model.onnx")
    assert os.path.isfile(model_path)

    hash1 = sha256sum(model_path)

    # Delete and download again
    os.remove(model_path)
    shutil.rmtree(os.path.dirname(model_path), ignore_errors=True)
    _ = Classifier(model_folder=folder, format="onnx")

    hash2 = sha256sum(model_path)

    # Test that the model was re-downloaded (at least once more)
    assert hash1 == hash2  # optional if content is static
    assert os.path.exists(model_path)
