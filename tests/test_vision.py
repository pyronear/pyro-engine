import numpy as np

from pyroengine.vision import Classifier


def test_classifier(mock_wildfire_image):
    # Instantiate the ONNX model
    model = Classifier()
    # Check preprocessing
    out, pad = model.preprocess_image(mock_wildfire_image)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert out.shape == (1, 3, 1024, 1024)
    assert isinstance(pad, tuple)
    # Check inference
    out = model(mock_wildfire_image)
    assert out.shape == (1, 5)
    conf = np.max(out[:, 4])
    assert conf >= 0 and conf <= 1

    # Test mask
    mask = np.ones((1024, 640))
    out = model(mock_wildfire_image, mask)
    assert out.shape == (1, 5)

    mask = np.zeros((1024, 1024))
    out = model(mock_wildfire_image, mask)
    assert out.shape == (0, 5)
