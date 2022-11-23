import numpy as np
from huggingface_hub import hf_hub_download

from pyroengine.vision import Classifier


def test_classifier(mock_wildfire_image, tmpdir_factory):

    # Cache
    folder = str(tmpdir_factory.mktemp("vision_cache"))

    # Instantiate the ONNX model
    model = Classifier(
        [
            "pyronear/rexnet1_0x",
        ],
        folder,
    )
    # Check preprocessing
    out = model.preprocess_image(0, mock_wildfire_image)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert out.shape == (1, 3, 256, 384)
    # Check inference
    out = model(mock_wildfire_image)
    assert isinstance(out, np.float32) and out.dtype == np.float32
    assert out >= 0 and out <= 1

    # Test model ensemble

    # Instantiate the ONNX model
    model = Classifier(
        [
            "pyronear/rexnet1_0x",
            "pyronear/rexnet1_3x",
        ],
        folder,
    )
    # Check preprocessing
    out = model.preprocess_image(0, mock_wildfire_image)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert out.shape == (1, 3, 256, 384)
    out = model.preprocess_image(1, mock_wildfire_image)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert out.shape == (1, 3, 256, 384)
    # Check inference
    out = model(mock_wildfire_image)
    assert isinstance(out, np.float32) and out.dtype == np.float32
    assert out >= 0 and out <= 1
