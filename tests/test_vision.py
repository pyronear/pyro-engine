import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyroengine.vision import Classifier

METADATA_PATH = "data/model_metadata.json"
model_path = "data/yolov8s.onnx"
sha = "9f1b1c2654d98bbed91e514ce20ea73a0a5fbd1111880f230d516ed40ea2dc58"


# Test for the case : the model doesn't exist
def test_classifier(mock_wildfire_image):
    print("test_classifier")

    # Instantiate the ONNX model
    model = Classifier(model_folder="data")
    # Check inference
    out = model(mock_wildfire_image)
    assert out.shape[1] == 5
    conf = np.max(out[:, 4])
    assert 0 <= conf <= 1

    model = Classifier(model_folder="data", format="onnx")

    # Test mask
    mask = np.ones((384, 640))
    out = model(mock_wildfire_image, mask)
    assert out.shape == (1, 5)

    mask = np.zeros((384, 640))
    out = model(mock_wildfire_image, mask)
    assert out.shape == (0, 5)
