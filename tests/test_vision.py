import numpy as np

from pyroengine.vision import Classifier


def test_classifier(mock_wildfire_image):

    # Instantiate the ONNX model
    model = Classifier()
    # Check preprocessing
    out = model.preprocess_image(mock_wildfire_image)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert out.shape == (1, 3, 384, 640)
    # Check inference
    out = model(mock_wildfire_image)
    assert out >= 0 and out <= 1
